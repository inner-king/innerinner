# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.7-DROID behind the wire format the eval stage's Gr00tClient speaks (repro/README.md, chapter 30.8).

The eval stage (simfoundry/policies/gr00t.py) talks to a GR00T server over zmq + msgpack, with numpy arrays packed
as npy blobs (simfoundry/policies/gr00t_utils.py, MsgSerializer). For N1.7 it sends each camera frame JPEG-encoded
and reads the actions without a batch dimension, as the server it was written against answered: the authors'
Gr00tG1RealPolicyWrapper, which Isaac-GR00T does not publish. The public N1.7 server (gr00t/eval/run_gr00t_server.py
at the N1.7 release, commit 1a1837f) packs arrays with msgpack_numpy instead and has no such wrapper, so the two do
not understand each other.

This serves the public policy, Gr00tPolicy behind Gr00tSimPolicyWrapper (what run_gr00t_server.py
--use-sim-policy-wrapper builds), in the client's format: JPEG frames are decoded to uint8 arrays of shape
(batch, time, height, width, 3), and the batch dimension is dropped from the returned actions. Nothing else is
touched; the eval stage is unchanged. Endpoints: ping, kill, reset, get_modality_config, get_action.

Run from an Isaac-GR00T checkout at 1a1837f, in its env:
    .venv/bin/python <repo>/repro/policy_eval_ours/gr00t_n17_server.py \\
        --model-path /data/models/gr00t/GR00T-N1.7-DROID --port 5556
"""
import argparse
import dataclasses
import enum
import io

import msgpack
import numpy as np
import zmq
from PIL import Image


def encode(obj):
    """msgpack default hook: numpy arrays as npy blobs, as the client's MsgSerializer packs them."""
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot pack {type(obj)}")


def decode(obj):
    """msgpack object hook: the client's npy blobs back to arrays."""
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def plain(value):
    """A ModalityConfig field as msgpack-native data: dataclasses to dicts, enums to their values."""
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def modality_configs(policy):
    """The policy's modality configs in the client's marker format (ModalityConfig(**as_dict) on its side)."""
    return {name: {"__ModalityConfig_class__": True, "as_dict": plain(cfg)}
            for name, cfg in policy.get_modality_config().items()}


def jpeg_frames(rows):
    """[[jpeg bytes per frame] per batch item] -> uint8 array (batch, time, height, width, 3)."""
    return np.asarray([[np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in row] for row in rows],
                      dtype=np.uint8)


def get_action(policy, observation, options=None):
    obs = dict(observation)
    for key, value in observation.items():
        if key.startswith("video.") and not isinstance(value, np.ndarray):
            obs[key] = jpeg_frames(value)
    action, _info = policy.get_action(obs, options)
    # (batch, time, dim) -> (time, dim): the client takes the batch's only item as the action chunk.
    return [{k: np.asarray(v)[0] for k, v in action.items()}, {}]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--embodiment-tag", default="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5556)
    args = ap.parse_args()

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    policy = Gr00tSimPolicyWrapper(Gr00tPolicy(embodiment_tag=EmbodimentTag.resolve(args.embodiment_tag),
                                               model_path=args.model_path, device=args.device, strict=True))
    handlers = {
        "ping": lambda data: {"status": "ok", "message": "Server is running"},
        "reset": lambda data: policy.reset((data or {}).get("options")) or {},
        "get_modality_config": lambda data: modality_configs(policy),
        "get_action": lambda data: get_action(policy, data["observation"], data.get("options")),
    }
    socket = zmq.Context().socket(zmq.REP)
    socket.bind(f"tcp://{args.host}:{args.port}")
    print(f"GR00T N1.7 server ({args.embodiment_tag}, {args.model_path}) on {args.host}:{args.port}", flush=True)
    while True:
        request = msgpack.unpackb(socket.recv(), object_hook=decode, raw=False)
        endpoint = request.get("endpoint", "get_action")
        if endpoint == "kill":
            socket.send(msgpack.packb({"status": "killed"}, default=encode))
            break
        try:
            if endpoint not in handlers:
                raise ValueError(f"unknown endpoint {endpoint}")
            reply = handlers[endpoint](request.get("data"))
        except Exception as exc:  # noqa: BLE001 - the client raises it as "Server error: ..."
            import traceback
            traceback.print_exc()
            reply = {"error": repr(exc)}
        socket.send(msgpack.packb(reply, default=encode))


if __name__ == "__main__":
    main()
