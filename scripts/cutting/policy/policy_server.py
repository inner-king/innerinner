# SPDX-License-Identifier: Apache-2.0
"""Serves a trained LeRobot policy to tofu_policy_episodes.py --mode policy over a local socket.

Messages (multiprocessing.connection, authkey b"tofu"): {"cmd": "reset"} -> "ok"; {"cmd": "act", "state",
"env_state", "front", "wrist"} -> the next action. Arrays travel as (dtype, shape, bytes), so the two
environments' numpy versions never meet in a pickle. Run in the lerobot env:
    python policy_server.py --ckpt <output_dir>/checkpoints/last/pretrained_model --port 6000
"""
import argparse
import time
from multiprocessing.connection import Listener

import numpy as np
import torch

TASK = "cut the tofu on the board into two pieces"
DICE_TASK = "dice the tofu on the board into a grid of cubes"


def unpack(t):
    return np.frombuffer(t[2], dtype=t[0]).reshape(t[1]).copy()


def pack(a):
    a = np.ascontiguousarray(a)
    return (a.dtype.str, a.shape, a.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=6000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temporal-ensemble", type=float, default=None,
                    help="ACT: predict a chunk every tick and average the overlapping ones with this weight (ACT paper 0.01)")
    ap.add_argument("--dice-variant", choices=["none", "A", "B"], default="none",
                    help="dicing policies (dice_to_lerobot.py): A adds the finished-cut count to the state; B adds the "
                         "hand's state 0.5 s and 1 s ago and the wrist frame 1 s ago (ACT takes one observation step, so "
                         "the history goes in as inputs), kept here per episode")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="ACT: actions executed from each predicted chunk before predicting again (1 = re-plan every tick)")
    args = ap.parse_args()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.device = args.device
    if args.temporal_ensemble is not None:  # an inference-time choice of ACT; the weights are the same
        cfg.n_action_steps = 1
        cfg.temporal_ensemble_coeff = args.temporal_ensemble
    if args.n_action_steps is not None:  # also inference-time only; the chunk the model predicts stays the same
        cfg.n_action_steps = args.n_action_steps
    if getattr(cfg, "use_peft", False):
        # A LoRA checkpoint (π0.5 option B) holds the adapter only: load the base policy it names, then the adapter,
        # as lerobot's make_policy does. PEFT injects the adapter into the base's own modules, so the unwrapped policy
        # is served (its select_action/reset/config are the policy's, with the adapter active).
        from peft import PeftConfig, PeftModel

        pc = PeftConfig.from_pretrained(args.ckpt)
        base = get_policy_class(cfg.type).from_pretrained(pc.base_model_name_or_path, revision=pc.revision, config=cfg)
        policy = PeftModel.from_pretrained(base, args.ckpt, config=pc, is_trainable=False).base_model.model
        print("adapter on", pc.base_model_name_or_path, pc.revision, flush=True)
    else:
        policy = get_policy_class(cfg.type).from_pretrained(args.ckpt, config=cfg)
    print("type", cfg.type, "chunk_size", getattr(cfg, "chunk_size", None), "n_action_steps", cfg.n_action_steps,
          "temporal_ensemble", getattr(cfg, "temporal_ensemble_coeff", None), flush=True)
    policy.to(args.device).eval()
    # The saved pipeline names the device it was trained on; serve on the one asked for.
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt,
                                         preprocessor_overrides={"device_processor": {"device": args.device}})
    feats = set(policy.config.input_features)
    print("inputs:", sorted(feats), flush=True)
    task = DICE_TASK if args.dice_variant != "none" else TASK
    hist = []
    with Listener(("127.0.0.1", args.port), authkey=b"tofu") as listener:
        print("ready", flush=True)
        while True:
            conn = listener.accept()
            n, t_inf = 0, 0.0
            while True:
                try:
                    msg = conn.recv()
                except (EOFError, ConnectionResetError):
                    break
                if msg["cmd"] == "reset":
                    policy.reset()
                    hist = []
                    conn.send("ok")
                    continue
                obs = {"task": task}
                if "observation.state" in feats:
                    st = unpack(msg["state"])
                    n = policy.config.input_features["observation.state"].shape[0]
                    if args.dice_variant == "A":  # the hand (last 9) and the finished cuts
                        st = np.append(st[-9:], float(msg["cuts_done"]))
                    elif args.dice_variant == "B":  # the hand now, 0.5 s and 1 s ago (the episode's first tick before that)
                        hist.append((st[-9:].copy(), unpack(msg["wrist"])))
                        past = [hist[max(0, len(hist) - 1 - k)][0] for k in (0, 5, 10)]
                        st = np.concatenate(past)
                        obs["observation.images.wrist_past"] = torch.from_numpy(
                            hist[max(0, len(hist) - 11)][1]).permute(2, 0, 1).float() / 255.0
                    obs["observation.state"] = torch.from_numpy(st[len(st) - n:]).float()  # the last n: the hand
                if "observation.environment_state" in feats:
                    obs["observation.environment_state"] = torch.from_numpy(unpack(msg["env_state"])).float()
                for cam in ("front", "wrist"):
                    k = f"observation.images.{cam}"
                    if k in feats:
                        obs[k] = torch.from_numpy(unpack(msg[cam])).permute(2, 0, 1).float() / 255.0
                t0 = time.perf_counter()
                with torch.inference_mode():
                    act = post(policy.select_action(pre(obs)))
                t_inf += time.perf_counter() - t0
                n += 1
                conn.send(pack(act.reshape(-1).float().cpu().numpy()))
            print(f"client gone after {n} actions, mean inference {1e3 * t_inf / max(n, 1):.1f} ms", flush=True)


if __name__ == "__main__":
    main()
