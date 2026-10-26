# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write one ARKit LiDAR depth frame as the results.npz that SimFoundry stage 2 (DA3) writes.

Pipeline A's stage 2 runs Depth Anything 3 over the RGB frames and keeps the result in
`s2_da/da/exports/npz/results.npz`. This script writes that file from an ARKit RGB-D capture, so
the later depth stages can use LiDAR depth without the DA3 step. It runs on the CPU and does not
import simfoundry, so importing it has no side effects.

The depth and confidence maps are resampled from the depth grid onto the RGB grid, and the
intrinsics move to the same crop and output size. The arithmetic is written out as specified
because a reference depth array is compared with this output to 1e-4 m: `cv2.remap` quantises
sample positions to 1/32 pixel, so a different formula (for example `warpAffine`) gives different
values.

Usage: python repro/lidar/arkit_depth_to_results_npz.py --capture DIR --frame-id N --rgb PNG
           --out-scene SCENE_DIR [--size WxH] [--crop x0,y0,x1,y1] [--low-conf keep|inpaint] [--check]

Writes `SCENE_DIR/s2_da/da/exports/npz/results.npz` and `SCENE_DIR/s2_da/stage_info.json`, and
nothing else; the capture is only read. With --check, exit status 1 means the written file failed a
shape, dtype or depth check.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

DEFAULT_SIZE = "448x336"
ASPECT_TOLERANCE = 0.01
BACKEND = "arkit_lidar"
PLANE_ITERATIONS = 500
PLANE_TOLERANCE_M = 0.01
PLANE_SEED = 0
INTRINSICS_TOLERANCE = 1e-3


def parse_size(text):
    """Parse an output size written as ``WxH``, width first.

    Args:
        text: For example ``"448x336"``.

    Returns:
        The ``(W, H)`` pair of positive integers.

    Raises:
        argparse.ArgumentTypeError: If the text is not two positive integers joined by ``x``.
    """
    try:
        w, h = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--size must look like 448x336, got {text!r}") from None
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"--size needs positive sizes, got {text!r}")
    return w, h


def parse_crop(text):
    """Parse a crop box written as ``x0,y0,x1,y1``. The range is checked against the frame later.

    Args:
        text: For example ``"256,290,1790,1440"``.

    Returns:
        A tuple of four integers.

    Raises:
        argparse.ArgumentTypeError: If the text does not hold exactly four integers.
    """
    try:
        values = tuple(int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--crop must look like x0,y0,x1,y1, got {text!r}") from None
    if len(values) != 4:
        raise argparse.ArgumentTypeError(f"--crop must have four values, got {text!r}")
    return values


def build_parser():
    parser = argparse.ArgumentParser(description="Write one ARKit LiDAR frame as a stage-2 results.npz.")
    parser.add_argument("--capture", required=True, help="ARKit capture folder with frames.jsonl, depth/ and conf/")
    parser.add_argument("--frame-id", type=int, required=True, help="id of the frame to convert")
    parser.add_argument("--rgb", required=True, help="RGB PNG of the same frame; becomes the image output")
    parser.add_argument("--out-scene", required=True, help="scene folder to write s2_da/ into")
    parser.add_argument("--size", type=parse_size, default=DEFAULT_SIZE,
                        help="output size WxH, width first (default 448x336)")
    parser.add_argument("--crop", type=parse_crop, default=None,
                        help="crop box x0,y0,x1,y1 on the RGB grid; x1 and y1 are excluded")
    parser.add_argument("--low-conf", choices=["keep", "inpaint"], default="keep",
                        help="keep: only zero depth is filled; inpaint: zero depth and conf 0 are filled")
    parser.add_argument("--check", action="store_true", help="print shape, intrinsics, depth and plane checks")
    return parser


def load_frame_record(capture, frame_id):
    """Return the frames.jsonl record whose id is ``frame_id``.

    Raises:
        SystemExit: If frames.jsonl is missing or holds no such frame.
    """
    path = capture / "frames.jsonl"
    if not path.is_file():
        raise SystemExit(f"no frames.jsonl in {capture}")
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["id"] == frame_id:
                return record
    raise SystemExit(f"frame {frame_id} is not in {path}")


def read_depth_and_conf(capture, frame_id, depth_wh):
    """Read the uint16 millimetre depth and the uint8 confidence of one frame.

    Args:
        capture: ARKit capture folder.
        frame_id: Frame id; files are named with six digits.
        depth_wh: The ``(dw, dh)`` depth grid size from frames.jsonl.

    Returns:
        ``(depth_mm, conf)``, both shaped ``(dh, dw)``.

    Raises:
        SystemExit: If a file is missing, has the wrong size, or has the wrong dtype.
    """
    dw, dh = depth_wh
    depth_path = capture / "depth" / f"{frame_id:06d}.png"
    conf_path = capture / "conf" / f"{frame_id:06d}.png"
    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise SystemExit(f"cannot read depth image {depth_path}")
    if conf is None:
        raise SystemExit(f"cannot read confidence image {conf_path}")
    if depth_mm.shape != (dh, dw) or depth_mm.dtype != np.uint16:
        raise SystemExit(f"{depth_path}: expected uint16 {dw}x{dh}, got {depth_mm.dtype} {depth_mm.shape}")
    if conf.shape != (dh, dw) or conf.dtype != np.uint8:
        raise SystemExit(f"{conf_path}: expected uint8 {dw}x{dh}, got {conf.dtype} {conf.shape}")
    return depth_mm, conf


def read_rgb(path):
    """Read an RGB image as uint8 RGB, dropping any alpha channel."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)  # IMREAD_COLOR returns 3 channels, no alpha
    if bgr is None:
        raise SystemExit(f"cannot read RGB image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def image_box(box, rgb_wh, png_wh):
    """Map a crop box from the RGB grid onto the pixel grid of the PNG that was supplied.

    Args:
        box: ``(x0, y0, x1, y1)`` on the RGB grid.
        rgb_wh: ``(rw, rh)`` from frames.jsonl.
        png_wh: ``(pw, ph)`` of the RGB PNG.

    Returns:
        The box on the PNG grid.

    Raises:
        SystemExit: If the PNG's aspect ratio differs from the RGB grid's by more than 1 %.
    """
    rw, rh = rgb_wh
    pw, ph = png_wh
    if (pw, ph) == (rw, rh):
        return box
    # A PNG of a different shape is only a rescaled copy of the same frame, so the box scales
    # with it; a different aspect ratio means the wrong image was given.
    if abs((pw / ph) / (rw / rh) - 1) > ASPECT_TOLERANCE:
        raise SystemExit(f"RGB PNG is {pw}x{ph}, which does not match the frame's aspect {rw}x{rh}")
    sx, sy = pw / rw, ph / rh
    x0, y0, x1, y1 = box
    return round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy)


def sampling_maps(rgb_wh, depth_wh, size, box):
    """Build the remap grids that sample the depth grid at each output pixel.

    Args:
        rgb_wh: ``(rw, rh)`` RGB grid size.
        depth_wh: ``(dw, dh)`` depth grid size.
        size: ``(W, H)`` output size.
        box: ``(x0, y0, x1, y1)`` crop on the RGB grid.

    Returns:
        ``(map_x, map_y)`` float32 arrays shaped ``(H, W)``.
    """
    rw, rh = rgb_wh
    dw, dh = depth_wh
    W, H = size
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # Output pixel centres sit at u + 0.5. The remap indexes the depth grid by pixel index, where
    # index i is centred at i, so the -0.5 converts between the two conventions.
    xr = x0 + (u + 0.5) * cw / W
    yr = y0 + (v + 0.5) * ch / H
    map_x = (xr * dw / rw - 0.5).astype(np.float32)
    map_y = (yr * dh / rh - 0.5).astype(np.float32)
    return map_x, map_y


def crop_intrinsics(K_rgb, size, box):
    """Move a 3x3 RGB intrinsics matrix to the crop and output size."""
    W, H = size
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    K = K_rgb.copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    K[0] *= W / cw
    K[1] *= H / ch
    return K


def fill_invalid_depth(depth_mm, conf, low_conf):
    """Fill invalid depth so that every pixel holds a finite z > 0.

    Zero depth is always invalid. With ``inpaint``, conf 0 is invalid too. Filling happens on the
    depth grid before the remap, so a hole does not blend zeros into the real depth at its edge.

    Args:
        depth_mm: uint16 depth in millimetres, shaped ``(dh, dw)``.
        conf: uint8 confidence, same shape.
        low_conf: ``"keep"`` or ``"inpaint"``.

    Returns:
        A uint16 array with no invalid pixel, or the input itself when there is none.

    Raises:
        SystemExit: If no pixel is valid, so there is nothing to fill from.
    """
    invalid = depth_mm == 0
    if low_conf == "inpaint":
        invalid = invalid | (conf == 0)
    if not invalid.any():
        return depth_mm
    if invalid.all():
        raise SystemExit("no valid depth pixel to fill the holes from")
    # Millimetres as uint16 on purpose. cv2.inpaint (TELEA) on float32 metres fills a large hole with
    # negative and out-of-range values: 46 % of an 80x80 hole in frame 20313 came back <= 0, -0.71..2.21 m.
    filled = cv2.inpaint(depth_mm, invalid.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    still_empty = filled == 0
    if still_empty.any():
        # Telea cannot reach pixels far inside a hole; they take the nearest filled value instead.
        nearest = distance_transform_edt(still_empty, return_distances=False, return_indices=True)
        filled = filled[nearest[0], nearest[1]]
    return filled


def refuse_to_overwrite(stage_dir, results_path):
    """Stop if results.npz exists and was not written by this script.

    A DA3 result at this path must not be replaced silently, so the file is only replaced when the
    stage_info.json beside it says ``backend: arkit_lidar``.

    Raises:
        SystemExit: If results.npz exists without an arkit_lidar stage_info.json.
    """
    if not results_path.exists():
        return
    backend = None
    info_path = stage_dir / "stage_info.json"
    if info_path.is_file():
        try:
            backend = json.loads(info_path.read_text()).get("backend")
        except (OSError, ValueError, AttributeError):
            backend = None
    if backend != BACKEND:
        raise SystemExit(
            f"{results_path} exists and was not written from an ARKit capture "
            f"(backend={backend!r}); refusing to overwrite it"
        )


def fit_plane(depth, K):
    """Fit a plane to the depth map, back-projected through K, with RANSAC.

    Args:
        depth: Depth in metres, shaped ``(H, W)``.
        K: 3x3 intrinsics of the depth map.

    Returns:
        ``(normal, inlier_ratio, rms_m)``: the unit normal oriented toward the camera (``n_z <= 0``),
        the fraction of valid pixels within ``PLANE_TOLERANCE_M`` of the best plane, and the RMS of
        those inliers' distances to a least-squares refit, in metres.
    """
    H, W = depth.shape
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    z = depth.astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    x = (uu - K[0, 2]) / K[0, 0] * z
    y = (vv - K[1, 2]) / K[1, 1] * z
    points = np.stack([x, y, z], axis=-1)[valid]

    rng = np.random.default_rng(PLANE_SEED)
    best = None
    for _ in range(PLANE_ITERATIONS):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length < 1e-12:
            continue
        n /= length
        inlier = np.abs((points - a) @ n) <= PLANE_TOLERANCE_M
        if best is None or inlier.sum() > best.sum():
            best = inlier
    if best is None:
        raise SystemExit("plane fit found no non-degenerate sample")

    inliers = points[best]
    centroid = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vt[-1]
    if normal[2] > 0:
        normal = -normal
    residual = (inliers - centroid) @ normal
    return normal, float(best.sum()) / len(points), float(np.sqrt(np.mean(residual ** 2)))


def run_check(results_path, rec, size, box):
    """Re-read the written file, print the checks, and return the exit status.

    Returns:
        0 when every array has the expected shape and dtype and every depth is finite and positive;
        1 otherwise. An intrinsics mismatch is printed but does not change the status.
    """
    W, H = size
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    expected = {
        "image": ((1, H, W, 3), np.uint8),
        "depth": ((1, H, W), np.float32),
        "conf": ((1, H, W), np.float32),
        "extrinsics": ((1, 3, 4), np.float32),
        "intrinsics": ((1, 3, 3), np.float32),
    }
    ok = True

    with np.load(results_path) as npz:
        keys = list(npz.files)
        arrays = {key: npz[key] for key in keys}
    print("arrays:")
    if keys != list(expected):
        ok = False
        print(f"  key order {keys} is not {list(expected)}")
    for key in keys:
        if key not in expected:
            ok = False
            print(f"  {key:<10} not an expected key  MISMATCH")
            continue
        shape, dtype = expected[key]
        good = arrays[key].shape == shape and arrays[key].dtype == dtype
        ok = ok and good
        print(f"  {key:<10} shape={arrays[key].shape} dtype={arrays[key].dtype} {'ok' if good else 'MISMATCH'}")

    K_rgb = np.array(rec["K"], dtype=np.float64).reshape(3, 3)
    stored = arrays["intrinsics"][0].astype(np.float64)
    pairs = {
        "fx": (stored[0, 0], K_rgb[0, 0] * W / cw),
        "fy": (stored[1, 1], K_rgb[1, 1] * H / ch),
        "cx": (stored[0, 2], (K_rgb[0, 2] - x0) * W / cw),
        "cy": (stored[1, 2], (K_rgb[1, 2] - y0) * H / ch),
    }
    print("intrinsics (written | expected from frames.jsonl):")
    for name, (got, want) in pairs.items():
        diff = got - want
        flag = "" if abs(diff) <= INTRINSICS_TOLERANCE else "  <-- differs by more than 1e-3"
        print(f"  {name}  {got:.4f}  {want:.4f}  diff={diff:+.1e}{flag}")

    depth = arrays["depth"][0]
    finite = np.isfinite(depth)
    nan_count = int(np.isnan(depth).sum())
    inf_count = int(np.isinf(depth).sum())
    nonpositive = int((depth[finite] <= 0).sum())
    if finite.any():
        values = depth[finite]
        print(f"depth (m): min={values.min():.4f} median={np.median(values):.4f} max={values.max():.4f}")
    else:
        ok = False
        print("depth (m): no finite pixel")
    print(f"depth bad pixels: NaN={nan_count} inf={inf_count} <=0={nonpositive}")
    if nan_count or inf_count or nonpositive:
        ok = False

    conf_values, conf_counts = np.unique(arrays["conf"], return_counts=True)
    print("conf counts: " + "  ".join(f"{v:.0f}={c}" for v, c in zip(conf_values, conf_counts)))

    if finite.any():
        normal, inlier_ratio, rms_m = fit_plane(depth, stored)
        print(f"plane (RANSAC {PLANE_ITERATIONS} iters, tol {PLANE_TOLERANCE_M * 100:.0f} cm, seed {PLANE_SEED}):")
        print(f"  normal=({normal[0]:+.4f}, {normal[1]:+.4f}, {normal[2]:+.4f})  "
              f"inliers={inlier_ratio * 100:.1f}%  rms={rms_m * 1000:.2f} mm")
    return 0 if ok else 1


def main(argv=None):
    """Convert one ARKit frame and write it into a scene folder.

    Args:
        argv: Command-line arguments, or None to use ``sys.argv[1:]``.

    Returns:
        0 on success. With ``--check``, 1 when the written file fails a shape, dtype or depth check.

    Raises:
        SystemExit: On bad input, a missing frame, or an existing results.npz that this script did
            not write.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    capture = Path(args.capture)
    stage_dir = Path(args.out_scene) / "s2_da"
    results_path = stage_dir / "da" / "exports" / "npz" / "results.npz"
    refuse_to_overwrite(stage_dir, results_path)

    rec = load_frame_record(capture, args.frame_id)
    rw, rh = rec["rgb_wh"]
    dw, dh = rec["depth_wh"]
    W, H = args.size
    if args.crop is not None:
        x0, y0, x1, y1 = args.crop
        if not (0 <= x0 < x1 <= rw and 0 <= y0 < y1 <= rh):
            parser.error(f"--crop {x0},{y0},{x1},{y1} must satisfy 0 <= x0 < x1 <= {rw} and 0 <= y0 < y1 <= {rh}")
    box = tuple(args.crop) if args.crop is not None else (0, 0, rw, rh)
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    if abs((cw / ch) / (W / H) - 1) > ASPECT_TOLERANCE:
        print(f"warning: crop {cw}x{ch} is not {W}:{H}, so the output has fx != fy", file=sys.stderr)

    depth_mm, conf = read_depth_and_conf(capture, args.frame_id, (dw, dh))
    rgb = read_rgb(args.rgb)
    png_h, png_w = rgb.shape[:2]
    px0, py0, px1, py1 = image_box(box, (rw, rh), (png_w, png_h))
    if not (0 <= px0 < px1 <= png_w and 0 <= py0 < py1 <= png_h):
        raise SystemExit(f"crop {box} maps to an empty box on the {png_w}x{png_h} RGB PNG")

    depth_mm = fill_invalid_depth(depth_mm, conf, args.low_conf)
    depth_m = depth_mm.astype(np.float32) / 1000.0  # z-depth in metres, so no further conversion
    map_x, map_y = sampling_maps((rw, rh), (dw, dh), (W, H), box)
    depth = cv2.remap(depth_m, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    conf_out = cv2.remap(conf, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    image = cv2.resize(rgb[py0:py1, px0:px1], (W, H), interpolation=cv2.INTER_AREA)

    K_rgb = np.array(rec["K"], dtype=np.float64).reshape(3, 3)
    K = crop_intrinsics(K_rgb, (W, H), box)
    # Depth is already z in the camera frame, so the extrinsics are the identity.
    extrinsics = np.hstack([np.eye(3), np.zeros((3, 1))])

    results_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(results_path),
        image=image[None],
        depth=depth[None],
        conf=conf_out[None],
        extrinsics=extrinsics[None].astype(np.float32),
        intrinsics=K[None].astype(np.float32),
    )
    info = {
        "success": True,
        "backend": BACKEND,
        "capture": str(capture.resolve()),
        "frame_id": args.frame_id,
        "size": [W, H],
        "crop": list(box) if args.crop is not None else None,
        "low_conf": args.low_conf,
    }
    with (stage_dir / "stage_info.json").open("w") as f:
        json.dump(info, f, indent=2)
    print(f"wrote {results_path}")

    if args.check:
        return run_check(results_path, rec, (W, H), box)
    return 0


if __name__ == "__main__":
    sys.exit(main())
