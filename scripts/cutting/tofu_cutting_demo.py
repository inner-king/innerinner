# SPDX-License-Identifier: Apache-2.0
"""Cut a block of tofu with a knife in the MPM cutting solver; write a video, a force plot and metrics.

Scenarios:
    single  one press cut through the middle of the block at a fixed speed
    saw     the same cut with a sawing motion along the edge
    dice    three cuts one way and two the other: twelve columns
    score   a 1 cm deep cut (a slit, as for scoring tofu): one piece, the slit stays open
    push    the knife is pushed down with a constant force through a compliant holder, as a
            robot under impedance control would; the tofu's reaction slows it (two-way coupling)
    rock    rocking cuts of 3 cm slices off a 10 x 7 x 5 cm block, as in the Chef'd reference clip
    slab    a press cut 1.2 cm from the end, then a twist that tips the slab over (HDMD clip)
    chop    the same block halved lengthwise by a level blade pressed straight down
    cook    the same cut as the Chef'd cook makes it: tilted, gliding forward while it comes
            down and while it rocks on the belly (slice-to-push about 2.7), then drawn back

The rock and slab scenarios reproduce real clips in Data/tofu_reference with a block of a
typical size. scripts/cutting/tofu_real2sim.py cuts the tofu SimFoundry reconstructed from a
frame of the Chef'd clip instead, and sets the result beside the clip.

Example (run from the repository root; the GPU is shared, see --allow-busy-gpu):
    PYTHONPATH=$PWD ~/miniforge3/envs/simfoundry/bin/python scripts/cutting/tofu_cutting_demo.py \
        --scenario single --device cuda:0 --out results/tofu_cutting/single
"""

import argparse
import csv
import json
import math
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from simfoundry.cutting.export import export_pieces
from simfoundry.cutting.knife import CHEF_KNIFE, KnifeGeometry, KnifePose, KnifeTrajectory
from simfoundry.cutting.materials import PRESETS
from simfoundry.cutting.mpm import MPMCuttingSim, SimConfig
from simfoundry.cutting.render import KITCHEN, STUDIO, Camera, Renderer, tofu_look

# Blade along y cuts the block's length (x); yaws chosen so the handle points away from the camera.
ACROSS = -math.pi / 2.0
ALONG = 0.0
# A US 14 oz firm block, the kind in the Chef'd and Blender Kitchen clips (length : height
# about 1.9 in the Chef'd side view).
REFERENCE_BLOCK = (0.10, 0.07, 0.05)
CLEAVER = KnifeGeometry(half_length=0.09, height=0.09, spine_thickness=3.0e-3, bevel_height=0.03, spine_taper=0.0)
SCENARIO_DEFAULTS = {
    # scenario: (block, knife, descend speed [m/s], view, style)
    "rock": (REFERENCE_BLOCK, CHEF_KNIFE, 0.06, "side", "kitchen"),
    "slab": (REFERENCE_BLOCK, CLEAVER, 0.035, "front", "studio"),
    # The pair compares technique at the same downward speed (the Chef'd cook's 8 cm/s).
    "chop": (REFERENCE_BLOCK, CHEF_KNIFE, 0.08, "side", "kitchen"),
    "cook": (REFERENCE_BLOCK, CHEF_KNIFE, 0.08, "side", "kitchen"),
}
# The HDMD cook drains the tofu under a weight for 45 minutes first.
SCENARIO_MATERIAL = {"slab": "pressed_tofu"}


def gpu_busy_reason():
    """Why the shared GPU should not be used now, or None. Uses nvidia-smi only."""
    try:
        util = []
        for _ in range(3):
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.split()
            util.append(float(out[0]))
            time.sleep(0.5)
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError, IndexError, ValueError) as exc:
        return f"nvidia-smi failed: {exc}"
    if max(util) > 10.0:
        return f"GPU utilisation {max(util):.0f}%"
    # Idle model servers (ComfyUI, Ollama) hold memory but do no work; anything else is a job.
    others = [a for a in apps if a and not any(s in a.lower() for s in ("comfyui", "ollama"))]
    if others:
        return "other GPU processes: " + "; ".join(others)
    return None


class ScriptedKnife:
    """The knife follows a ``KnifeTrajectory`` regardless of the force on it."""

    def __init__(self, trajectory: KnifeTrajectory):
        self.trajectory = trajectory
        self.cut_times = trajectory.cut_times

    @property
    def duration(self):
        return self.trajectory.duration

    def advance(self, sim, t):
        return sim.advance(t, self.trajectory)


class PushedKnife:
    """A knife pressed down with a constant force by a compliant holder: two-way coupling.

    Each control step the knife moves at its current velocity; the solver returns the mean
    force of the tofu on it over the step, which then changes that velocity:
    m dv/dt = -F_push - c v + F_tofu. This is the exchange a robot simulator would make with
    the solver (knife motion in, wrench out), on a robot-like 500 Hz control rate.
    """

    def __init__(self, x, y, yaw, *, top_z, board_z, push_force, mass=0.25, damping=8.0,
                 step=2e-3, start=0.15, hover=0.006, max_push=2.5, lift_speed=0.1, retreat=0.08,
                 settle=0.3, board_gap=0.3e-3):
        self.x, self.y, self.yaw = x, y, yaw
        self.board_z, self.bottom = board_z, board_z + board_gap
        self.push_force, self.mass, self.damping, self.step = push_force, mass, damping, step
        self.start, self.max_push, self.lift_speed, self.settle = start, max_push, lift_speed, settle
        self.z = top_z + hover
        self.top = top_z + hover + retreat
        self.vz = 0.0
        self.phase = "wait"
        self.cut_started = False
        self.cut_times = [start]
        self.push_end = None
        self.lift_end = None
        self.reached_board_at = None
        self.deepest = self.z

    @property
    def duration(self):
        if self.lift_end is not None:
            return self.lift_end + self.settle
        if self.push_end is not None:
            return self.push_end + (self.top - self.deepest) / self.lift_speed + self.settle
        return self.start + self.max_push + (self.top - self.bottom) / self.lift_speed + self.settle

    def advance(self, sim, t_end):
        f_sum = np.zeros(3)
        m_sum = np.zeros(3)
        span = 0.0
        while sim.time < t_end - 1e-12:
            t0 = sim.time
            t1 = min(t0 + self.step, t_end)
            h = t1 - t0
            if self.phase == "wait" and t0 >= self.start - 1e-12:
                self.phase = "push"
            if self.phase == "push":
                z_next = max(self.z + self.vz * h, self.bottom)
            elif self.phase == "lift":
                z_next = min(self.z + self.lift_speed * h, self.top)
            else:
                z_next = self.z
            begins = self.phase == "push" and not self.cut_started
            self.cut_started = self.cut_started or begins
            traj = KnifeTrajectory.linear(
                t0, t1, KnifePose(np.array([self.x, self.y, self.z]), self.yaw),
                KnifePose(np.array([self.x, self.y, z_next]), self.yaw), begins_cut=begins,
            )
            f, m = sim.advance(t1, traj)
            f_sum += f * h
            m_sum += m * h
            span += h
            self.z = z_next
            self.deepest = min(self.deepest, self.z)
            if self.phase == "push":
                self.vz += h * (-self.push_force - self.damping * self.vz + f[2]) / self.mass
                if self.z <= self.bottom + 1e-9:
                    self.reached_board_at = self.reached_board_at or t1
                    self.vz = 0.0
                    self.phase, self.push_end = "lift", t1
                elif t1 >= self.start + self.max_push:
                    self.phase, self.push_end = "lift", t1
            elif self.phase == "lift" and self.z >= self.top - 1e-9:
                self.phase, self.lift_end = "settle", t1
        return f_sum / max(span, 1e-12), m_sum / max(span, 1e-12)


def build_cuts(scenario, sim):
    lo, hi = sim.block_min, sim.block_max
    c = 0.5 * (lo + hi)
    if scenario in ("single", "saw", "score", "push"):
        return [(c[0], c[1], ACROSS)]
    if scenario == "rock":
        # Two 3 cm slices off the +x end; the tip points to +y, away from the camera.
        return [(hi[0] - 0.03, c[1], math.pi / 2.0), (hi[0] - 0.06, c[1], math.pi / 2.0)]
    if scenario == "slab":
        # A 1.2 cm slab off the -x end; e_v = +x, so a positive twist pushes the slab over.
        return [(lo[0] + 0.012, c[1], ACROSS)]
    if scenario in ("chop", "cook"):
        # Lengthwise through the middle, as the Chef'd cook halves the block; tip towards +x.
        return [(c[0], c[1], ALONG)]
    if scenario == "dice":
        lx, ly = hi[0] - lo[0], hi[1] - lo[1]
        xs = [lo[0] + lx * f for f in (0.25, 0.5, 0.75)]
        ys = [lo[1] + ly * f for f in (1.0 / 3.0, 2.0 / 3.0)]
        return [(x, c[1], ACROSS) for x in xs] + [(c[0], y, ALONG) for y in ys]
    raise ValueError(scenario)


def build_driver(args, sim):
    top_z = sim.block_max[2]
    cuts = build_cuts(args.scenario, sim)
    if args.scenario == "push":
        x, y, yaw = cuts[0]
        return PushedKnife(x, y, yaw, top_z=top_z, board_z=sim.board_z, push_force=args.push_force)
    if args.scenario == "rock":
        reach = (sim.block_max[1] - 0.5 * (sim.block_min[1] + sim.block_max[1])) + 0.012
        traj = KnifeTrajectory.rock_strokes(
            cuts, top_z=top_z, board_z=sim.board_z, knife_half_length=sim.knife_geometry.half_length,
            tip_reach=reach, descend_speed=args.descend_speed, geometry=sim.knife_geometry,
        )
        return ScriptedKnife(traj)
    if args.scenario == "cook":
        x, y, yaw = cuts[0]
        reach = 0.5 * (sim.block_max[0] - sim.block_min[0]) - 0.008  # the tip enters at the far corner
        glide, roll_glide = 0.042, 0.024  # tracked in the Chef'd clip
        traj = KnifeTrajectory.rock_strokes(
            cuts, top_z=top_z, board_z=sim.board_z, knife_half_length=sim.knife_geometry.half_length,
            tip_reach=reach, geometry=sim.knife_geometry, glide=glide, roll_glide=roll_glide,
            descend_speed=args.descend_speed, rock_rate=math.radians(67.0),
            slide=reach + glide + roll_glide + 0.5 * (sim.block_max[0] - sim.block_min[0]) + 0.01,
        )
        return ScriptedKnife(traj)
    if args.scenario == "slab":
        traj = KnifeTrajectory.press_twist(
            cuts[0], top_z=top_z, board_z=sim.board_z, descend_speed=args.descend_speed,
            twist=math.radians(args.twist_deg),
        )
        return ScriptedKnife(traj)
    bottom = sim.board_z
    if args.scenario == "score":
        bottom = top_z - 0.010 - 0.3e-3  # stop 10 mm into the tofu
    saw = dict(saw_amplitude=0.012, saw_frequency=2.0) if args.scenario == "saw" else {}
    traj = KnifeTrajectory.strokes(
        cuts, top_z=top_z, board_z=bottom, descend_speed=args.descend_speed, retreat=0.08, **saw
    )
    return ScriptedKnife(traj)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="single",
                    choices=["single", "saw", "dice", "score", "push", "rock", "slab", "chop", "cook"])
    ap.add_argument("--material", default=None, choices=sorted(PRESETS), help="default: firm_tofu (slab: pressed_tofu)")
    ap.add_argument("--fracture-energy", type=float, default=None, help="override the preset's G_c [J/m^2]")
    ap.add_argument("--block", type=float, nargs=3, default=None, help="tofu block size [m] (default per scenario)")
    ap.add_argument("--dx", type=float, default=2.5e-3, help="grid spacing [m]")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-busy-gpu", action="store_true", help="run on a GPU other jobs are using")
    ap.add_argument("--descend-speed", type=float, default=None, help="knife speed while cutting [m/s]")
    ap.add_argument("--push-force", type=float, default=0.6, help="push scenario: downward force [N]")
    ap.add_argument("--twist-deg", type=float, default=15.0, help="slab scenario: tilt of the blade while it pushes [deg]")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--view", choices=["oblique", "side", "front"], default=None, help="camera (default per scenario)")
    ap.add_argument("--style", choices=["studio", "kitchen"], default=None, help="lights and backdrop")
    ap.add_argument("--blade-alpha", type=float, default=1.0, help="blade opacity; below 1 shows the cut through it")
    ap.add_argument("--no-overlay", action="store_true", help="no text on the frames")
    ap.add_argument("--png-every", type=float, default=1.0, help="save a PNG frame every this many seconds")
    ap.add_argument("--out", type=Path, default=Path("results/tofu_cutting/single"))
    args = ap.parse_args()

    if args.device.startswith("cuda") and not args.allow_busy_gpu:
        reason = gpu_busy_reason()
        if reason:
            raise SystemExit(f"Refusing to use the shared GPU ({reason}); use --device cpu or --allow-busy-gpu.")

    args.out.mkdir(parents=True, exist_ok=True)
    args.material = args.material or SCENARIO_MATERIAL.get(args.scenario, "firm_tofu")
    material = PRESETS[args.material]
    if args.fracture_energy is not None:
        material = replace(material, fracture_energy=args.fracture_energy)
    block, knife, speed, view, style = SCENARIO_DEFAULTS.get(
        args.scenario, ((0.08, 0.05, 0.03), KnifeGeometry(), 0.05, "oblique", "studio")
    )
    args.block = list(block) if args.block is None else args.block
    args.descend_speed = speed if args.descend_speed is None else args.descend_speed
    view = args.view or view
    style = {"studio": STUDIO, "kitchen": KITCHEN}[args.style or style]
    sim = MPMCuttingSim(material, args.block, knife=knife, config=SimConfig(dx=args.dx), device=args.device)
    top_z = sim.block_max[2]
    driver = build_driver(args, sim)

    center = 0.5 * (sim.block_min + sim.block_max)
    span = float(np.max(sim.block_max - sim.block_min))
    target = np.array([center[0], center[1], sim.board_z + 0.35 * (top_z - sim.board_z)])
    offset, fov, lift = {
        "oblique": ((0.95, -2.45, 1.55), 32.0, 0.0),
        # Chef'd: a low view of the long side, far enough back to see the knife come down.
        "side": ((1.35, -2.9, 1.0), 34.0, 0.012),
        # HDMD: the long face from the front, a little above.
        "front": ((0.05, -2.5, 0.75), 34.0, 0.01),
    }[view]
    target = target + np.array([0.0, 0.0, lift])
    eye = target + span * np.array(offset)
    cam = Camera(eye, target, fov_deg=fov, width=args.width, height=args.height)
    margin = 0.06
    board_rect = (-margin, -margin, sim.domain_size[0] + margin, sim.domain_size[1] + margin)
    spacing = args.dx / sim.config.particles_per_axis
    renderer = Renderer(
        cam, sim.board_z, board_rect, knife, particle_radius=1.1 * spacing, particle_spacing=spacing,
        blade_alpha=args.blade_alpha, style=style, look=tofu_look(sim.x0, sim.skin),
    )

    writer = None
    if not args.no_video:
        import imageio.v2 as imageio

        writer = imageio.get_writer(str(args.out / "tofu_cutting.mp4"), fps=args.fps, codec="libx264", quality=8)
    frames_dir = args.out / "frames"
    frames_dir.mkdir(exist_ok=True)

    rows = []
    t_wall = time.time()
    sim_wall = 0.0
    fi = 0
    while (fi + 1) / args.fps <= driver.duration + 0.5 / args.fps:
        fi += 1
        t = min(fi / args.fps, driver.duration)
        t0 = time.time()
        force, moment = driver.advance(sim, t)
        sim_wall += time.time() - t0
        pose = sim.knife_pose
        front = float("nan")
        if sim.active_cut >= 0:
            fr = sim.crack_front(sim.active_cut)
            o = sim._cut_frames[sim.active_cut][0]
            if fr.min() < 999.0:
                front = float(fr.min() + o[2] - sim.board_z)
        # Tilt of piece 0 (in the slab scenario, the slab): its thinnest axis from vertical,
        # 90 deg while it stands, 0 once it lies flat.
        x_now = sim.positions()
        sel0 = sim.pieces() == 0
        tilt0 = float("nan")
        if sel0.sum() > 10:
            w0, v0 = np.linalg.eigh(np.cov((x_now[sel0] - x_now[sel0].mean(0)).T))
            tilt0 = math.degrees(math.acos(min(1.0, abs(float(v0[2, 0])))))
        rows.append(dict(
            t=t, edge_height=float(pose.origin[2] - sim.board_z), depth=float(top_z - pose.origin[2]),
            fx=force[0], fy=force[1], fz=force[2], pieces=sim.connected_pieces(), cut=sim.active_cut,
            crack_front_height=front, piece0_thin_axis_from_vertical_deg=tilt0,
        ))
        overlay = [
            f"{args.scenario}: {material.name.replace('_', ' ')}, G_c {material.fracture_energy:g} J/m2   t = {t:5.2f} s",
            f"knife force  Fz = {force[2]:+.2f} N",
            f"pieces: {rows[-1]['pieces']}",
        ]
        save_png = fi == 1 or (fi % max(1, int(round(args.png_every * args.fps))) == 0)
        if writer is not None or save_png:
            x_now = sim.positions()
            img = renderer.render(
                x_now, sim.piece_groups()[sim.pieces()], pose, None if args.no_overlay else overlay,
                sim.crack_faces(x_now),
            )
        if writer is not None:
            writer.append_data(img)
        if save_png:
            import imageio.v2 as imageio

            imageio.imwrite(frames_dir / f"frame_{fi:04d}.png", img)
        if fi % 15 == 0:
            print(f"frame {fi} t={t:.2f}s Fz={force[2]:+.3f}N pieces={rows[-1]['pieces']} "
                  f"wall={time.time() - t_wall:.0f}s", flush=True)
    import imageio.v2 as imageio

    x_now = sim.positions()
    img = renderer.render(
        x_now, sim.piece_groups()[sim.pieces()], sim.knife_pose, None if args.no_overlay else overlay,
        sim.crack_faces(x_now),
    )
    imageio.imwrite(frames_dir / f"frame_{fi:04d}.png", img)
    if writer is not None:
        writer.close()

    with open(args.out / "timeseries.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    x, pc = sim.positions(), sim.pieces()
    pieces = []
    for p in range(sim.n_pieces):
        sel = pc == p
        if sel.any():
            # The thinnest principal axis: horizontal for a slab standing up (90 deg), vertical
            # for one lying flat (0 deg).
            evals, evecs = np.linalg.eigh(np.cov((x[sel] - x[sel].mean(0)).T))
            thin = evecs[:, 0]
            pieces.append(dict(
                piece=p, particles=int(sel.sum()), mass_g=round(1e3 * sim.particle_mass * float(sel.sum()), 2),
                centroid=x[sel].mean(0).round(5).tolist(), extent=(x[sel].max(0) - x[sel].min(0)).round(5).tolist(),
                thin_axis_from_vertical_deg=round(math.degrees(math.acos(min(1.0, abs(float(thin[2]))))), 1),
            ))
    cutting = [r for r in rows if r["depth"] > 0.002 and r["cut"] >= 0 and r["fz"] > 0.0]
    metrics = dict(
        scenario=args.scenario, material=material.__dict__, block=args.block, dx=args.dx, dt=sim.dt,
        device=args.device, particles=sim.n_particles, grid=list(sim.grid_shape),
        duration_s=round(sim.time, 4), substeps=int(round(sim.time / sim.dt)),
        sim_wall_s=round(sim_wall, 1), total_wall_s=round(time.time() - t_wall, 1),
        cuts=sim.n_cuts, cuts_complete=[sim.cut_complete(c) for c in range(sim.n_cuts)],
        # How far below the top the edge was when each cut first cracked (the dent before it cuts).
        crack_onset_depth_mm=[
            None if not np.isfinite(h) else round(1e3 * (top_z - h), 2)
            for h in (sim.crack_onset_height(c) for c in range(sim.n_cuts))
        ],
        connected_pieces=sim.connected_pieces(),
        pieces=pieces, max_knife_force_z_N=max((r["fz"] for r in rows), default=0.0),
        mean_knife_force_z_while_cutting_N=float(np.mean([r["fz"] for r in cutting])) if cutting else 0.0,
        particles_below_board=int((x[:, 2] < sim.board_z - 0.5 * args.dx).sum()),
        nan_positions=bool(np.isnan(x).any()),
    )
    if isinstance(driver, PushedKnife):
        metrics.update(
            push_force_N=args.push_force,
            deepest_edge_depth_mm=round(1e3 * (top_z - driver.deepest), 2),
            reached_board_at_s=driver.reached_board_at,
            push_time_s=None if driver.push_end is None else round(driver.push_end - driver.start, 3),
        )
    # The final particles, and each piece as a closed mesh for a rigid-body scene.
    np.savez_compressed(args.out / "final_particles.npz", x=x, piece=pc, particle_mass=sim.particle_mass)
    groups = sim.piece_groups()[pc]  # fields joined below a slit export as one mesh
    metrics["meshes"] = export_pieces(x, groups, spacing, args.out / "meshes", density=material.density)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ts = np.array([r["t"] for r in rows])
    axes[0].plot(ts, [r["fz"] for r in rows], label="Fz (up)")
    axes[0].plot(ts, [r["fx"] for r in rows], label="Fx", alpha=0.6)
    axes[0].plot(ts, [r["fy"] for r in rows], label="Fy", alpha=0.6)
    for ct in driver.cut_times:
        axes[0].axvline(ct, color="k", lw=0.5, ls=":")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("force of tofu on knife [N]")
    axes[0].legend()
    t_next = driver.cut_times[1] if len(driver.cut_times) > 1 else float("inf")
    down = []  # the downward part of the first stroke
    for r in rows:
        if not driver.cut_times[0] <= r["t"] < t_next:
            continue
        if down and r["depth"] < down[-1]["depth"] - 1e-4:
            break
        down.append(r)
    axes[1].plot([1e3 * r["depth"] for r in down], [r["fz"] for r in down], marker=".")
    axes[1].set_xlabel("knife edge depth below tofu top [mm] (first stroke)")
    axes[1].set_ylabel("Fz [N]")
    axes[1].axvline(1e3 * (top_z - sim.board_z), color="k", lw=0.5, ls=":")
    fig.suptitle(f"{material.name}: {args.scenario}, dx = {1e3 * args.dx:.1f} mm")
    fig.tight_layout()
    fig.savefig(args.out / "knife_force.png", dpi=120)
    keys = ("particles", "cuts_complete", "crack_onset_depth_mm", "sim_wall_s", "total_wall_s",
            "max_knife_force_z_N", "mean_knife_force_z_while_cutting_N")
    print(json.dumps({k: metrics[k] for k in keys}, indent=1))
    if isinstance(driver, PushedKnife):
        print("push:", {k: metrics[k] for k in ("deepest_edge_depth_mm", "reached_board_at_s", "push_time_s")})
    print("pieces", len(pieces), "->", args.out)


if __name__ == "__main__":
    main()
