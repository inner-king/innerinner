# SPDX-License-Identifier: Apache-2.0
"""Cut the tofu that SimFoundry reconstructed from a cooking video, seen from that video's camera.

Pipeline A (scripts/pipeline/A_reconstruction) turns one frame of a clip into a textured mesh
of the tofu, the board under it and the camera. This script fills the tofu with MPM particles
(by default its level box: tofu is a block, and a generated mesh is not quite one), cuts it the
way the cook in the Chef'd clip does next (lengthwise through the middle, the blade's face
towards the camera: tilted, pushed forward as it comes down and as it rocks flat on its belly,
then drawn back, all tracked in the clip), and draws every frame from the
reconstructed camera over the photographed frame, with the real tofu and knife painted out
and the tofu's skin taken from the frame. With --reference the clip itself runs beside it.

The tofu's proportions, its place on the board and its colour come from the reconstruction;
only the material and the knife's motion are chosen here. One correction is made first: a small
object's size and distance rest on the few depth pixels it covers, so the tofu is rescaled and
moved along the board until its outline matches the frame (``real2sim.fit_to_silhouette``), its
bottom on the board's top as the depth map measures it. --no-fit keeps the pipeline's values.

Example (repository root; the scene comes from repro/run_local_pipeline.sh A_reconstruction
--scene-name tofu_chefd ... on a frame of Data/tofu_reference/chefd_side.mp4):
    PYTHONPATH=$PWD ~/miniforge3/envs/simfoundry/bin/python scripts/cutting/tofu_real2sim.py \
        --scene Data/tofu_chefd --photo Data/tofu_reference/chefd_frame_17.9.png \
        --reference Data/tofu_reference/chefd_side.mp4 --reference-start 4.17 \
        --device cuda:0 --out results/tofu_real2sim/chefd
"""

import argparse
import csv
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from simfoundry.cutting.knife import CHEF_KNIFE, KnifeTrajectory
from simfoundry.cutting.materials import PRESETS
from simfoundry.cutting.mpm import MPMCuttingSim, SimConfig
from simfoundry.cutting.real2sim import (
    board_top, fit_to_silhouette, level_box, load_reconstruction, photo_albedo, place_food, stage5_mask,
    surface_height,
)
from simfoundry.cutting.render import KITCHEN, SKIN_RGB, Camera, Renderer, tofu_look

# The cook's stroke, tracked in the clip (the blade's tip, 1 px = 0.49 mm at the tofu): the tip
# enters at the tofu's top far corner, then goes down to the board 4.2 cm further on, and rocks
# flat on the belly while moving another 2.4 cm. Along the tilted blade that is a slice-to-push
# ratio of about 2.7: the cook slices as much as he presses.
CHEFD_GLIDE = 0.042
CHEFD_ROLL_GLIDE = 0.024
# Under the clip's large soft light the blade's flat shows almost white.
CHEFD_STEEL = (0.90, 0.91, 0.93)
# Lights of the Chef'd frame, relative to the camera (azimuth from its viewing direction towards
# its right, elevation above the table). Fitted to the median brightness of the tofu's three
# visible faces in the frame (front 204, top 162, right end 140 in sRGB) under the renderer's
# shading, with the skin albedo of render.SKIN_RGB: a low key light from the front right.
CHEFD_LIGHT = dict(key_az_deg=29.4, key_el_deg=17.3, fill=(-0.5, 0.3, 0.8), ambient=0.15)


def camera_relative(cam_to_frame, right, towards, up):
    """A direction given as (right, towards the camera, up) components, in the camera's frame.

    Right and towards are taken level, so "up" stays the table's normal.
    """
    r = cam_to_frame[:3, 0].copy()
    f = cam_to_frame[:3, 2].copy()
    r[2] = f[2] = 0.0
    r /= np.linalg.norm(r)
    f /= np.linalg.norm(f)
    d = right * r - towards * f + up * np.array([0.0, 0.0, 1.0])
    return d / np.linalg.norm(d)


def paint_out(photo, mask, grow_px=9):
    """The photo with ``mask`` (grown a little) filled in from around it."""
    import cv2

    k = 2 * grow_px + 1
    grown = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((k, k), np.uint8))
    return cv2.inpaint(np.ascontiguousarray(photo[..., :3]), grown, 9, cv2.INPAINT_TELEA)


def auto_crop(cam_full, pts_sim, size, pad=(1.1, 2.4, 1.1, 0.9)):
    """A window of the photo around the food, with room above it for the knife.

    ``pad`` is (left, top, right, bottom) in units of the food's projected width and height.
    """
    px, py, _ = cam_full.project(pts_sim)
    x0, x1, y0, y1 = px.min(), px.max(), py.min(), py.max()
    fw, fh = x1 - x0, y1 - y0
    w, h = size
    box = [x0 - pad[0] * fw, y0 - pad[1] * fh, x1 + pad[2] * fw, y1 + pad[3] * fh]
    box = [int(max(0, box[0])), int(max(0, box[1])), int(min(w, box[2])), int(min(h, box[3]))]
    # Even sizes, for the video encoder.
    box[2] -= (box[2] - box[0]) % 2
    box[3] -= (box[3] - box[1]) % 2
    return tuple(box)


def reference_frames(path, start, fps, crop, scale, count):
    """``count`` frames of the reference clip from ``start`` [s] at ``fps``, cropped and scaled."""
    import imageio_ffmpeg

    x0, y0, x1, y1 = crop
    w, h = int(round((x1 - x0) * scale)), int(round((y1 - y0) * scale))
    gen = imageio_ffmpeg.read_frames(
        str(path), input_params=["-ss", f"{start:.3f}"],
        output_params=["-vf", f"fps={fps},crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale={w}:{h}"],
    )
    meta = next(gen)
    w, h = meta["size"]
    frames = []
    for raw in gen:
        frames.append(np.frombuffer(raw, np.uint8).reshape(h, w, 3))
        if len(frames) >= count:
            break
    gen.close()
    return frames


def label(img, text, light=True):
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, img.shape[0] // 22))
    except OSError:
        font = ImageFont.load_default()
    d.text((12, 8), text, fill=(240, 240, 240) if light else (20, 20, 20), font=font)
    return np.asarray(im)


def silhouette_iou(cam, x, real, knife, radius):
    """IoU of the particles' projection with the real food mask, outside the knife's mask."""
    px, py, z = cam.project(x)
    sim = np.zeros(real.shape, dtype=bool)
    r = max(1, int(math.ceil(cam.focal * radius / float(np.median(z)))))
    ix, iy = np.floor(px).astype(int), np.floor(py).astype(int)
    for oy in range(-r, r + 1):
        for ox in range(-r, r + 1):
            if ox * ox + oy * oy > r * r:
                continue
            jx, jy = ix + ox, iy + oy
            ok = (jx >= 0) & (jx < real.shape[1]) & (jy >= 0) & (jy < real.shape[0])
            sim[jy[ok], jx[ok]] = True
    valid = ~knife
    inter = (sim & real & valid).sum()
    union = ((sim | real) & valid).sum()
    return float(inter / max(union, 1)), sim


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True, help="Pipeline A scene directory")
    ap.add_argument("--img-idx", type=int, default=None, help="input frame of the scene (default: stage 5's)")
    ap.add_argument("--photo", type=Path, required=True, help="the frame the scene was built from")
    ap.add_argument("--reference", type=Path, default=None, help="the clip, for a side-by-side video")
    ap.add_argument("--reference-start", type=float, default=0.0, help="clip time [s] at simulation time 0")
    ap.add_argument("--food", default="tofu", help="category of the food in the scene")
    ap.add_argument("--board", default="board", help="category of the board under it")
    ap.add_argument("--knife", default="knife", help="category of the knife to paint out of the photo")
    ap.add_argument("--shape", choices=["box", "hull", "mesh"], default="box",
                    help="the food's shape: the reconstructed mesh, its convex hull (undoes occlusion), or "
                         "its level box (also undoes the generator's chamfers and lean; tofu is a block)")
    ap.add_argument("--no-fit", action="store_true",
                    help="keep the pipeline's size and place for the food instead of fitting them to its outline")
    ap.add_argument("--material", default="firm_tofu", choices=sorted(PRESETS))
    ap.add_argument("--fracture-energy", type=float, default=None)
    ap.add_argument("--dx", type=float, default=2.5e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--descend-speed", type=float, default=0.08,
                    help="[m/s]; in the clip the tip goes from the tofu's top to the board in 0.7 s")
    ap.add_argument("--tilt-deg", type=float, default=30.0, help="pitch of the blade as it comes down")
    ap.add_argument("--rock-rate-deg", type=float, default=67.0, help="[deg/s]; 30 deg in 0.45 s in the clip")
    ap.add_argument("--no-glide", action="store_true",
                    help="press straight down and rock about the tip, without the cook's forward slide")
    ap.add_argument("--slide-speed", type=float, default=0.3,
                    help="[m/s]; the cook then draws the knife out towards its handle (0: lift it instead)")
    ap.add_argument("--start-time", type=float, default=0.15, help="when the knife starts down [s]")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=float, default=1.0, help="render scale of the cropped photo")
    ap.add_argument("--crop", type=int, nargs=4, default=None, help="x0 y0 x1 y1 in photo pixels (default: around the food)")
    ap.add_argument("--png-every", type=float, default=0.25, help="save a PNG frame every this many seconds")
    ap.add_argument("--out", type=Path, default=Path("results/tofu_real2sim/chefd"))
    args = ap.parse_args()

    import cv2
    import imageio.v2 as imageio

    args.out.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out / "frames"
    frames_dir.mkdir(exist_ok=True)
    t_wall = time.time()

    rec = load_reconstruction(args.scene, args.img_idx)
    food = rec.find(args.food)
    board = rec.find(args.board)
    photo = imageio.imread(args.photo)[..., :3]
    H0, W0 = photo.shape[:2]
    food_mask = stage5_mask(args.scene, food.name, (W0, H0))
    try:
        knife_mask = stage5_mask(args.scene, rec.find(args.knife).name, (W0, H0))
    except (KeyError, FileNotFoundError):
        knife_mask = np.zeros_like(food_mask)
    print(f"{food.category} ({food.name}) as reconstructed: {np.round(100 * food.mesh.extents, 2).tolist()} cm "
          f"(poses from {rec.pose_file})", flush=True)

    # The board's top from the depth map around the food, and the food fitted to its outline on it.
    xs = np.nonzero(food_mask)[1]
    to_depth = rec.calib_size[0] / W0
    board_z_depth = surface_height(rec, board.name, columns=((xs.min() - xs.ptp()) * to_depth, (xs.max() + xs.ptp()) * to_depth))
    board_z_mesh = board_top(board.mesh, near_xy=food.mesh.vertices[:, :2].mean(axis=0))
    shape = level_box(food.mesh.convex_hull) if args.shape == "box" else food.mesh.copy()
    fit = None
    if not args.no_fit:
        cam_world = Camera.from_opencv(rec.cam_to_world, rec.K, W0, H0, calib_size=rec.calib_size)
        T_fit, fit = fit_to_silhouette(shape, cam_world, food_mask, valid=~knife_mask, z_bottom=board_z_depth)
        shape.apply_transform(T_fit)
        print(f"fitted to its outline on the board ({1e3 * board_z_depth:.1f} mm, from the depth map; the board mesh "
              f"says {1e3 * board_z_mesh:.1f} mm): scale {fit['scale']:.3f}, outline IoU "
              f"{fit['iou_before']:.3f} -> {fit['iou_after']:.3f}", flush=True)
    place = place_food(shape, convex=args.shape != "mesh")
    food_bottom = float(shape.vertices[:, 2].min())
    print(f"block {np.round(100 * place.block_size, 2).tolist()} cm, yaw {math.degrees(place.yaw):.1f} deg, "
          f"volume {1e6 * place.local_mesh.volume:.0f} cm3", flush=True)

    material = PRESETS[args.material]
    if args.fracture_energy is not None:
        material = replace(material, fracture_energy=args.fracture_energy)
    sim = MPMCuttingSim(
        material, place.block_size, knife=CHEF_KNIFE, config=SimConfig(dx=args.dx), device=args.device,
        inside=None if args.shape == "box" else place.inside,
    )
    S = place.sim_from_world(sim.block_min)
    cam_sim = S @ rec.cam_to_world

    # The photo's camera, and the window of the photo that is drawn.
    cam_full = Camera.from_opencv(cam_sim, rec.K, W0, H0, calib_size=rec.calib_size)
    crop = tuple(args.crop) if args.crop else auto_crop(cam_full, sim.x0, (W0, H0))
    x0, y0, x1, y1 = crop
    sx = W0 / rec.calib_size[0]
    sy = H0 / rec.calib_size[1]
    K_crop = np.array([[rec.K[0, 0] * sx, 0.0, rec.K[0, 2] * sx - x0], [0.0, rec.K[1, 1] * sy, rec.K[1, 2] * sy - y0], [0.0, 0.0, 1.0]])
    rw, rh = int(round((x1 - x0) * args.scale)), int(round((y1 - y0) * args.scale))
    rw, rh = rw - rw % 2, rh - rh % 2
    cam = Camera.from_opencv(cam_sim, K_crop, rw, rh, calib_size=(x1 - x0, y1 - y0))

    spacing = args.dx / sim.config.particles_per_axis
    iou, sim_mask = silhouette_iou(cam_full, sim.x0, food_mask, knife_mask, 0.6 * spacing)
    print(f"silhouette IoU with the photographed tofu: {iou:.3f}; crop {crop}", flush=True)
    overlay = photo.copy()
    for m, col in ((food_mask, (255, 60, 60)), (sim_mask, (60, 220, 255))):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, cnts, -1, col, 2)
    imageio.imwrite(args.out / "silhouette_check.png", overlay[y0:y1, x0:x1])

    clean = paint_out(photo, food_mask | knife_mask)
    imageio.imwrite(args.out / "painted_out.png", clean[y0:y1, x0:x1])
    background = cv2.resize(clean[y0:y1, x0:x1], (rw, rh), interpolation=cv2.INTER_AREA)

    board_pts = board.mesh.vertices @ S[:3, :3].T + S[:3, 3]
    board_rect = (board_pts[:, 0].min(), board_pts[:, 1].min(), board_pts[:, 0].max(), board_pts[:, 1].max())
    az, el = math.radians(CHEFD_LIGHT["key_az_deg"]), math.radians(CHEFD_LIGHT["key_el_deg"])
    style = replace(
        KITCHEN,
        key_light=tuple(camera_relative(cam_sim, math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el))),
        fill_light=tuple(camera_relative(cam_sim, *CHEFD_LIGHT["fill"])),
        ambient=CHEFD_LIGHT["ambient"],
        steel_rgb=CHEFD_STEEL,
    )

    # Skin as the frame shows it: every particle the camera sees at the start takes the photo's
    # colour less the renderer's light, so the first frame redraws the photo. The rest (the
    # back, the ends, where the knife hid it) takes the median of what was seen on the same
    # face of the block, or of all that was seen, with tofu_look's pores; cut faces keep
    # tofu_look's colour.
    albedo, gloss = tofu_look(sim.x0, sim.skin, cloth=0.0)
    renderer = Renderer(
        cam, sim.board_z, board_rect, CHEF_KNIFE, particle_radius=1.1 * spacing, particle_spacing=spacing,
        style=style, look=(albedo, gloss), photo=background,
    )
    photo_crop = cv2.resize(photo[y0:y1, x0:x1], (rw, rh), interpolation=cv2.INTER_AREA)
    seen_crop = cv2.resize((food_mask & ~knife_mask)[y0:y1, x0:x1].astype(np.uint8), (rw, rh),
                           interpolation=cv2.INTER_NEAREST)
    seen_crop = cv2.erode(seen_crop, np.ones((3, 3), np.uint8)).astype(bool)  # off the outline
    seen_albedo, seen = photo_albedo(renderer, sim.x0, np.zeros(sim.n_particles, dtype=int), photo_crop, seen_crop, gloss)
    rest = sim.skin & ~seen
    pores = albedo / SKIN_RGB[None, :]
    albedo[seen] = seen_albedo[seen]
    albedo[rest] = np.median(seen_albedo[seen], axis=0)[None, :] * pores[rest]
    # The block face each particle is nearest to (-x, -y, -z, +x, +y, +z).
    face = np.argmin(np.concatenate([sim.x0 - sim.block_min, sim.block_max - sim.x0], axis=1), axis=1)
    for f in range(6):
        on, known = rest & (face == f), seen & (face == f)
        if on.any() and known.sum() >= 50:
            albedo[on] = np.median(seen_albedo[known], axis=0)[None, :] * pores[on]
    renderer.look = (albedo, gloss)
    print(f"skin from the photo: {seen.sum()} particles seen, {rest.sum()} more skin from their median "
          f"{np.round(255 * np.median(seen_albedo[seen], axis=0)).astype(int).tolist()}", flush=True)

    # The cook's first stroke: lengthwise through the middle, tip on the right of the picture.
    right = cam_sim[:3, 0]
    knife_yaw = 0.0 if right[0] >= 0.0 else math.pi
    lo, hi = sim.block_min, sim.block_max
    centre = 0.5 * (lo + hi)
    reach = 0.5 * (hi[0] - lo[0]) - 0.008  # the tip enters just inside the far corner
    glide = CHEFD_GLIDE if not args.no_glide else 0.0
    roll_glide = CHEFD_ROLL_GLIDE if not args.no_glide else 0.0
    traj = KnifeTrajectory.rock_strokes(
        [(centre[0], centre[1], knife_yaw)], top_z=hi[2], board_z=sim.board_z,
        knife_half_length=CHEF_KNIFE.half_length, tip_reach=reach, geometry=CHEF_KNIFE,
        glide=glide, roll_glide=roll_glide,
        tilt=math.radians(args.tilt_deg), descend_speed=args.descend_speed,
        rock_rate=math.radians(args.rock_rate_deg), start_time=args.start_time,
        # Drawn back until the tip has cleared the tofu's near end.
        slide=reach + glide + roll_glide + 0.5 * (hi[0] - lo[0]) + 0.01 if args.slide_speed > 0.0 else 0.0,
        slide_speed=max(args.slide_speed, 1e-3),
    )

    n_frames = int(math.ceil(traj.duration * args.fps))
    ref = []
    if args.reference is not None:
        ref = reference_frames(args.reference, args.reference_start, args.fps, crop, args.scale, n_frames + 1)
    writer = imageio.get_writer(str(args.out / "sim.mp4"), fps=args.fps, codec="libx264", quality=8)
    pair_writer = None
    if ref:
        pair_writer = imageio.get_writer(str(args.out / "side_by_side.mp4"), fps=args.fps, codec="libx264", quality=8)

    rows = []
    sim_wall = 0.0
    every = max(1, int(round(args.png_every * args.fps)))
    for fi in range(n_frames + 1):
        t = min(fi / args.fps, traj.duration)
        t0 = time.time()
        force = np.zeros(3)
        if t > sim.time:
            force, _ = sim.advance(t, traj)
        sim_wall += time.time() - t0
        x = sim.positions()
        pose = sim.knife_pose
        rows.append(dict(t=round(t, 4), clip_time=round(args.reference_start + t, 4),
                         edge_height=float(pose.origin[2] - sim.board_z) if pose is not None else float("nan"),
                         fx=force[0], fy=force[1], fz=force[2], pieces=sim.connected_pieces()))
        img = renderer.render(x, sim.piece_groups()[sim.pieces()], pose, None, sim.crack_faces(x))
        writer.append_data(img)
        if fi < len(ref):
            r = ref[fi]
            if r.shape != img.shape:
                r = cv2.resize(r, (img.shape[1], img.shape[0]))
            pair = np.concatenate([label(r, f"video  {args.reference_start + t:5.2f} s"),
                                   label(img, f"real2sim + MPM  {t:5.2f} s   Fz {force[2]:+.2f} N")], axis=1)
            pair_writer.append_data(pair)
            if fi % every == 0:
                imageio.imwrite(frames_dir / f"pair_{fi:04d}.png", pair)
        elif fi % every == 0:
            imageio.imwrite(frames_dir / f"sim_{fi:04d}.png", img)
        if fi % 15 == 0:
            print(f"frame {fi}/{n_frames} t={t:.2f}s Fz={force[2]:+.3f}N pieces={rows[-1]['pieces']} "
                  f"wall={time.time() - t_wall:.0f}s", flush=True)
    writer.close()
    if pair_writer is not None:
        pair_writer.close()

    with open(args.out / "timeseries.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    top_z = sim.block_max[2]
    metrics = dict(
        scene=str(args.scene), pose_file=rec.pose_file, food=food.category, shape=args.shape, material=material.__dict__,
        block_size_cm=np.round(100 * place.block_size, 2).tolist(), yaw_deg=round(math.degrees(place.yaw), 2),
        food_volume_cm3=round(float(place.local_mesh.volume) * 1e6, 1),
        board_top_from_depth_mm=round(1e3 * board_z_depth, 2), board_top_from_mesh_mm=round(1e3 * board_z_mesh, 2),
        food_bottom_mm=round(1e3 * food_bottom, 2), outline_fit=fit,
        silhouette_iou=round(iou, 3), crop=list(crop), dx=args.dx, particles=sim.n_particles,
        mass_g=round(1e3 * sim.particle_mass * sim.n_particles, 1), dt=sim.dt, device=args.device,
        duration_s=round(sim.time, 3), sim_wall_s=round(sim_wall, 1), total_wall_s=round(time.time() - t_wall, 1),
        cut_complete=sim.cut_complete(0) if sim.n_cuts else False, connected_pieces=sim.connected_pieces(),
        crack_onset_depth_mm=None if not sim.n_cuts or not np.isfinite(sim.crack_onset_height(0))
        else round(1e3 * (top_z - sim.crack_onset_height(0)), 2),
        max_knife_force_z_N=max(r["fz"] for r in rows),
        reference=None if args.reference is None else str(args.reference), reference_start_s=args.reference_start,
    )
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("block_size_cm", "silhouette_iou", "particles", "mass_g",
                                              "cut_complete", "connected_pieces", "crack_onset_depth_mm",
                                              "max_knife_force_z_N", "total_wall_s")}, indent=1))


if __name__ == "__main__":
    main()
