# SPDX-License-Identifier: Apache-2.0
"""Check simfoundry/policies/dreamzero.py against DreamZero's server code, with no model and no GPU.

The server is DreamZero's own RoboArena transport (eval_utils/policy_server.py of a dreamzero0/dreamzero checkout,
loaded by path) around a stub policy whose observation conversion is the real server's
ARDroidRoboarenaPolicy._convert_observation, lifted out of socket_test_optimized_AR.py, and whose session handling
mirrors that class's infer(). The client is driven the way the eval stage drives it: reset(), then one infer() per
action chunk, for two episodes. Checked:

- each camera reaches the model input it should (exterior 1 -> video.exterior_image_1_left, and so on);
- one server call per infer(), a new session per episode, and 1 frame then 4 frames per call within an episode;
- an (N, 8) chunk comes back whether the server replies with the array or with {"actions": ...};
- the connection survives a call that blocks the server's event loop for --delay seconds, as the real forward
  pass does (seconds per call on one GPU, minutes for the first, compiling calls).

    python repro/policy_eval/check_dreamzero_client.py --dreamzero /content/dreamzero [--pkgs DIR_WITH_openpi_client]
"""
import argparse
import ast
import importlib.util
import os
import socket
import sys
import threading
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIDEO_KEYS = ("video.exterior_image_1_left", "video.exterior_image_2_left", "video.wrist_image_left")
PROMPT = "pick up the marker, and put the marker in the cup"
# One flat colour per camera, so the server side can tell which camera landed where.
OBS = {
    "exterior_image_1_left": np.full((720, 1280, 3), 10, np.uint8),
    "exterior_image_2_left": np.full((720, 1280, 3), 20, np.uint8),
    "wrist_image_left": np.full((720, 1280, 3), 30, np.uint8),
    "joint_position": np.full(7, 0.1, np.float32),
    "gripper_position": np.array([0.5], np.float32),
}
EXPECTED_VALUE = {"video.exterior_image_1_left": 10, "video.exterior_image_2_left": 20, "video.wrist_image_left": 30}


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def server_conversion(dreamzero_dir):
    """ARDroidRoboarenaPolicy._convert_observation and FRAMES_PER_CHUNK from the server script, without its imports."""
    tree = ast.parse(open(os.path.join(dreamzero_dir, "socket_test_optimized_AR.py")).read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ARDroidRoboarenaPolicy")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_convert_observation")
    frames = next(ast.literal_eval(n.value) for n in cls.body
                  if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "FRAMES_PER_CHUNK")
    ns = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "_convert_observation", "exec"), ns)
    return ns["_convert_observation"], frames


def make_stub(convert, frames_per_chunk):
    class StubARPolicy:
        FRAMES_PER_CHUNK = frames_per_chunk
        _convert_observation = convert

        def __init__(self, slow_call=None, delay_s=0.0, reply_dict=False):
            self._frame_buffers = {k: [] for k in VIDEO_KEYS}
            self._is_first_call = True
            self._current_session_id = None
            self.calls, self.resets = [], 0
            self.slow_call, self.delay_s, self.reply_dict = slow_call, delay_s, reply_dict

        def _reset_state(self):
            for k in self._frame_buffers:
                self._frame_buffers[k] = []
            self._is_first_call = True
            self.resets += 1

        def reset(self, reset_info):
            self._reset_state()

        def infer(self, obs):
            session_id = obs.get("session_id")
            if session_id is not None and session_id != self._current_session_id:
                if self._current_session_id is not None:
                    self._reset_state()
                self._current_session_id = session_id
            conv = self._convert_observation(obs)
            self.calls.append({"session": session_id, "prompt": conv["annotation.language.action_text"],
                               "state": conv["state.joint_position"].shape,
                               "video": {k: (v.shape[0], int(v.min()), int(v.max())) for k, v in conv.items()
                                         if k.startswith("video.")}})
            missing = [k for k in VIDEO_KEYS if k not in conv]
            if missing:
                raise KeyError(f"model inputs missing: {missing}")
            if len(self.calls) == self.slow_call:
                time.sleep(self.delay_s)  # blocks the event loop, as the real forward pass does
            self._is_first_call = False
            actions = np.tile(np.arange(8, dtype=np.float32), (24, 1))
            return {"actions": actions} if self.reply_dict else actions

    return StubARPolicy


def serve(policy_server, policy):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    # The config socket_test_optimized_AR.py serves DreamZero-DROID with.
    cfg = policy_server.PolicyServerConfig(image_resolution=(180, 320), needs_wrist_camera=True,
                                           n_external_cameras=2, needs_stereo_camera=False,
                                           needs_session_id=True, action_space="joint_position")
    server = policy_server.WebsocketPolicyServer(policy, cfg, host="127.0.0.1", port=port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(1.0)
    return port


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dreamzero", required=True, help="a dreamzero0/dreamzero checkout")
    ap.add_argument("--pkgs", default=None, help="directory to put on sys.path for openpi_client, if not installed")
    ap.add_argument("--delay", type=float, default=45.0, help="seconds the slow call blocks the server")
    args = ap.parse_args()
    if args.pkgs:
        sys.path.insert(0, args.pkgs)
    sys.path.insert(0, REPO)
    from simfoundry.policies.dreamzero import DreamZeroClient

    policy_server = load_by_path("dreamzero_policy_server", os.path.join(args.dreamzero, "eval_utils", "policy_server.py"))
    StubARPolicy = make_stub(*server_conversion(args.dreamzero))
    results = {}

    def check(name, ok, detail=""):
        results[name] = bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    for reply_dict in (False, True):
        policy = StubARPolicy(reply_dict=reply_dict)
        client = DreamZeroClient(host="127.0.0.1", port=serve(policy_server, policy), open_loop_horizon=8)
        shapes = []
        for _ in range(2):
            client.reset()
            for _ in range(3):
                shapes.append(client.infer(OBS, PROMPT)["action"].shape)
        tag = "dict reply" if reply_dict else "array reply"
        check(f"{tag}: one server call per infer", len(policy.calls) == 6, f"({len(policy.calls)})")
        check(f"{tag}: (24, 8) chunks", set(shapes) == {(24, 8)}, str(set(shapes)))
        if reply_dict:
            continue
        sessions = [c["session"] for c in policy.calls]
        check("a new session per episode",
              sessions[0] == sessions[2] and sessions[3] == sessions[5] and sessions[2] != sessions[3])
        check("the server reset once, between the episodes", policy.resets == 1, f"({policy.resets})")
        frames = [c["video"]["video.wrist_image_left"][0] for c in policy.calls]
        check("frames per call", frames == [1, 4, 4, 1, 4, 4], str(frames))
        check("each camera in its model input",
              all(c["video"][k][1:] == (v, v) for c in policy.calls for k, v in EXPECTED_VALUE.items()))
        check("prompt and joint state passed",
              all(c["prompt"] == PROMPT and c["state"] == (1, 7) for c in policy.calls))
        check("the server's image size used", client.image_hw == (180, 320), str(client.image_hw))

    policy = StubARPolicy(slow_call=2, delay_s=args.delay)
    client = DreamZeroClient(host="127.0.0.1", port=serve(policy_server, policy))
    client.reset()
    t0 = time.time()
    try:
        for _ in range(3):
            client.infer(OBS, PROMPT)
        check(f"survives a {args.delay:.0f} s call", len(policy.calls) == 3, f"({time.time() - t0:.0f} s)")
    except Exception as e:  # noqa: BLE001 - reported as the check's failure
        check(f"survives a {args.delay:.0f} s call", False, f"{type(e).__name__} after {time.time() - t0:.0f} s: {e}")

    print("ALL PASS" if all(results.values()) else f"FAILED: {[k for k, v in results.items() if not v]}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
