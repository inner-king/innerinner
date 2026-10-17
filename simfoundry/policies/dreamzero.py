# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Client for DreamZero's public inference server.

DreamZero-DROID (GEAR-Dreams/DreamZero-DROID) is served by ``socket_test_optimized_AR.py`` from
github.com/dreamzero0/dreamzero, which speaks RoboArena's websocket protocol rather than openpi's. This client
follows that repository's own simulation client (``DreamZeroJointPosClient`` in ``eval_utils/run_sim_eval.py``):

- The two external cameras go under ``exterior_image_0_left`` and ``exterior_image_1_left``, which the server files
  as the model's ``exterior_image_1_left`` and ``exterior_image_2_left`` (DROID's names). Sent under DROID's names,
  the first camera lands in the second slot and the second camera is dropped.
- The server is stateful: it keeps one frame per call and the world model's KV cache across calls, and each
  request advances that context by one block. So a request goes out exactly once per action chunk, and every
  episode gets a new ``session_id`` (``reset``), which makes the server start over rather than continue the
  previous episode.
- The reply is the (N, 8) action array itself (7 absolute joint positions, then the gripper); a dict with
  ``actions``, as a hosted endpoint returns, is accepted too.
- The server runs the forward pass inside its event loop and answers no ping meanwhile: seconds per call on one
  GPU, minutes for the first, compiling calls. The websockets default (20 s ping timeout) would drop the
  connection, so the client waits as long as DreamZero's own does.
"""

import time
import uuid

import numpy as np
import websockets.sync.client
from openpi_client import msgpack_numpy

from simfoundry.utils.processing_utils import resize_with_pad
from .abstract_client import InferenceClient

# DreamZero's eval_utils/policy_client.py.
PING_INTERVAL_S = 60
PING_TIMEOUT_S = 600


class DreamZeroClient(InferenceClient):
    def __init__(self,
                 host: str = "localhost",
                 port: int = 8000,
                 open_loop_horizon: int = 24,
                 ) -> None:
        self.open_loop_horizon = open_loop_horizon
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        print(f"Initializing DreamZero client with host: {host} and port: {port}")
        self._ws, self.server_config = self._connect()
        # The server's first message is its RoboArena config: image size, cameras, action space.
        if self.server_config.get("action_space", "joint_position") != "joint_position":
            raise ValueError(f"The eval stage executes joint positions; server config: {self.server_config}")
        self.image_hw = tuple(self.server_config.get("image_resolution") or (180, 320))
        self.session_id = str(uuid.uuid4())
        print(f"DreamZero client initialized, server config: {self.server_config}")

    def _connect(self):
        while True:
            try:
                ws = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None,
                    ping_interval=PING_INTERVAL_S, ping_timeout=PING_TIMEOUT_S,
                )
                return ws, msgpack_numpy.unpackb(ws.recv())
            except ConnectionRefusedError:
                print(f"Waiting for the DreamZero server at {self._uri}...")
                time.sleep(5)

    def reset(self):
        self.session_id = str(uuid.uuid4())

    def infer(self, obs: dict, instruction: str) -> dict:
        """Sends one observation and returns the server's action chunk.

        Args:
            obs: The eval stage's observation: ``exterior_image_1_left``, ``exterior_image_2_left`` and
                ``wrist_image_left`` (H, W, 3), ``joint_position`` (7,), ``gripper_position`` (1,).
            instruction: The task prompt.

        Returns:
            ``action``: the (N, 8) chunk; ``viz``: the first external and the wrist image side by side.
        """
        h, w = self.image_hw
        ext_1_image = resize_with_pad(obs["exterior_image_1_left"], h, w)
        ext_2_image = resize_with_pad(obs["exterior_image_2_left"], h, w)
        wrist_image = resize_with_pad(obs["wrist_image_left"], h, w)
        request = {
            "observation/exterior_image_0_left": ext_1_image,
            "observation/exterior_image_1_left": ext_2_image,
            "observation/wrist_image_left": wrist_image,
            "observation/joint_position": np.asarray(obs["joint_position"], dtype=np.float64),
            "observation/gripper_position": np.asarray(obs["gripper_position"], dtype=np.float64),
            "prompt": instruction,
            "session_id": self.session_id,
            "endpoint": "infer",
        }
        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, str):
            # The server sends its traceback as text, then closes the connection.
            raise RuntimeError(f"Error in DreamZero server:\n{response}")
        result = msgpack_numpy.unpackb(response)
        actions = np.asarray(result["actions"] if isinstance(result, dict) else result)
        if actions.ndim != 2 or actions.shape[1] != 8:
            raise ValueError(f"Expected an (N, 8) joint-position chunk, got shape {actions.shape}")
        return {"action": actions, "viz": np.concatenate([ext_1_image, wrist_image], axis=1)}
