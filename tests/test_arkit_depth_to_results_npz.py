# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "repro/lidar/arkit_depth_to_results_npz.py"

RGB_WH = (240, 180)
DEPTH_WH = (32, 24)
K = [200.0, 0.0, 118.5, 0.0, 205.0, 88.0, 0.0, 0.0, 1.0]
W, H = 48, 36
CROP_ARG = "13,7,201,148"
CROP = (13, 7, 201, 148)


def _load_script():
    spec = importlib.util.spec_from_file_location("arkit_depth_to_results_npz", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _step_depth():
    """Depth grid with a vertical step: 1 m for columns below 16, 2 m from column 16 on."""
    depth = np.full((DEPTH_WH[1], DEPTH_WH[0]), 2000, np.uint16)
    depth[:, :16] = 1000
    return depth


def _full_conf():
    return np.full((DEPTH_WH[1], DEPTH_WH[0]), 2, np.uint8)


def _make_inputs(root, depth_mm, conf, marker_col=None, png_wh=RGB_WH):
    """Write a fake capture with a decoy frame line, and an RGB PNG that may carry a marker column."""
    capture = root / "capture"
    (capture / "depth").mkdir(parents=True)
    (capture / "conf").mkdir()
    records = [
        {"id": 6, "K": K, "rgb_wh": list(RGB_WH), "depth_wh": list(DEPTH_WH)},
        {"id": 7, "K": K, "rgb_wh": list(RGB_WH), "depth_wh": list(DEPTH_WH)},
    ]
    (capture / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    cv2.imwrite(str(capture / "depth" / "000007.png"), depth_mm.astype(np.uint16))
    cv2.imwrite(str(capture / "conf" / "000007.png"), conf.astype(np.uint8))

    pw, ph = png_wh
    rgb = np.zeros((ph, pw, 3), np.uint8)
    if marker_col is not None:
        rgb[:, marker_col] = 255
    rgb_path = root / "frame.png"
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return capture, rgb_path


def _args(capture, rgb, scene, *extra, frame_id=7):
    # Extra arguments come last, so argparse's last-wins rule lets a test override --size.
    return [
        "--capture", str(capture), "--frame-id", str(frame_id), "--rgb", str(rgb),
        "--out-scene", str(scene), "--size", f"{W}x{H}", *extra,
    ]


def _results(scene):
    return scene / "s2_da/da/exports/npz/results.npz"


def _crossing(profile, level):
    """Sub-index position where a monotonically rising profile first reaches ``level``."""
    i = int(np.argmax(profile >= level))
    return (i - 1) + (level - profile[i - 1]) / (profile[i] - profile[i - 1])


def _tilted_plane_depth():
    """Depth grid samples of a plane 1 m from the camera, tilted by a few degrees, and its normal."""
    rw, rh = RGB_WH
    dw, dh = DEPTH_WH
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    n = np.array([0.1, -0.05, 1.0])
    n /= np.linalg.norm(n)
    xs = (np.arange(dw) + 0.5) * rw / dw
    ys = (np.arange(dh) + 0.5) * rh / dh
    x, y = np.meshgrid(xs, ys)
    rays = np.stack([(x - cx) / fx, (y - cy) / fy, np.ones_like(x)], axis=-1)
    z = 1.0 / (rays @ n)  # the plane is n . X = 1, and X = z * ray
    return np.round(z * 1000).astype(np.uint16), n


def _reference_maps(size, box):
    """Sampling maps written out from the specification, not taken from the script under test."""
    w, h = size
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    xr = x0 + (u + 0.5) * cw / w
    yr = y0 + (v + 0.5) * ch / h
    map_x = (xr * DEPTH_WH[0] / RGB_WH[0] - 0.5).astype(np.float32)
    map_y = (yr * DEPTH_WH[1] / RGB_WH[1] - 0.5).astype(np.float32)
    return map_x, map_y


@pytest.mark.parametrize("crop_arg", [None, CROP_ARG])
def test_writes_keys_order_shapes_and_dtypes(tmp_path, crop_arg):
    mod = _load_script()
    capture, rgb = _make_inputs(tmp_path, _step_depth(), _full_conf())
    scene = tmp_path / "scene"
    extra = [] if crop_arg is None else ["--crop", crop_arg]

    assert mod.main(_args(capture, rgb, scene, *extra)) == 0

    with np.load(_results(scene)) as npz:
        assert npz.files == ["image", "depth", "conf", "extrinsics", "intrinsics"]
        assert npz["image"].shape == (1, H, W, 3) and npz["image"].dtype == np.uint8
        assert npz["depth"].shape == (1, H, W) and npz["depth"].dtype == np.float32
        assert npz["conf"].shape == (1, H, W) and npz["conf"].dtype == np.float32
        assert npz["extrinsics"].shape == (1, 3, 4) and npz["extrinsics"].dtype == np.float32
        assert npz["intrinsics"].shape == (1, 3, 3) and npz["intrinsics"].dtype == np.float32
        np.testing.assert_array_equal(npz["extrinsics"][0], np.hstack([np.eye(3), np.zeros((3, 1))]))


@pytest.mark.parametrize("crop_arg, box", [(None, (0, 0, 240, 180)), (CROP_ARG, CROP)])
def test_intrinsics_follow_crop_and_size(tmp_path, crop_arg, box):
    mod = _load_script()
    capture, rgb = _make_inputs(tmp_path, _step_depth(), _full_conf())
    scene = tmp_path / "scene"
    extra = [] if crop_arg is None else ["--crop", crop_arg]
    mod.main(_args(capture, rgb, scene, *extra))

    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    fx = K[0] * W / cw
    fy = K[4] * H / ch
    cx = (K[2] - x0) * W / cw
    cy = (K[5] - y0) * H / ch
    with np.load(_results(scene)) as npz:
        np.testing.assert_allclose(npz["intrinsics"][0], [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], rtol=1e-6)


@pytest.mark.parametrize(
    "crop_arg, box, size",
    [
        # No crop: 240x180 to 40x36 scales x by 1/6 and y by 1/5, so the output is anisotropic.
        (None, (0, 0, 240, 180), (40, 36)),
        # A 40x30 output would match the crop's 4:3 shape and be isotropic, so 40x33 is used instead.
        (CROP_ARG, CROP, (40, 33)),
    ],
)
def test_matches_reference_resampling(tmp_path, crop_arg, box, size):
    mod = _load_script()
    rng = np.random.default_rng(7)
    depth = rng.integers(800, 2501, size=(DEPTH_WH[1], DEPTH_WH[0])).astype(np.uint16)  # no zero depth
    conf = rng.integers(0, 3, size=(DEPTH_WH[1], DEPTH_WH[0])).astype(np.uint8)
    capture, rgb = _make_inputs(tmp_path, depth, conf)
    scene = tmp_path / "scene"
    w, h = size
    extra = ["--size", f"{w}x{h}"] + (["--crop", crop_arg] if crop_arg else [])

    assert mod.main(_args(capture, rgb, scene, *extra)) == 0

    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    assert not np.isclose(w / cw, h / ch, rtol=1e-3)  # the two axes really do scale differently
    map_x, map_y = _reference_maps(size, box)
    depth_m = depth.astype(np.float32) / 1000.0
    expected_depth = cv2.remap(depth_m, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    expected_conf = cv2.remap(conf, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    with np.load(_results(scene)) as npz:
        assert np.array_equal(npz["depth"][0], expected_depth)
        assert np.array_equal(npz["conf"][0], expected_conf)
        k = npz["intrinsics"][0].astype(np.float64)
        np.testing.assert_allclose(k[0, 0], K[0] * w / cw, rtol=1e-6)
        np.testing.assert_allclose(k[1, 1], K[4] * h / ch, rtol=1e-6)
        np.testing.assert_allclose(k[0, 2], (K[2] - x0) * w / cw, rtol=1e-6)
        np.testing.assert_allclose(k[1, 2], (K[5] - y0) * h / ch, rtol=1e-6)


@pytest.mark.parametrize("crop_arg, box", [(None, (0, 0, 240, 180)), (CROP_ARG, CROP)])
def test_depth_step_lines_up_with_rgb_marker(tmp_path, crop_arg, box):
    mod = _load_script()
    marker_col = 120  # RGB column of the depth step: depth grid column 16 times 240 / 32
    capture, rgb = _make_inputs(tmp_path, _step_depth(), _full_conf(), marker_col=marker_col)
    scene = tmp_path / "scene"
    extra = [] if crop_arg is None else ["--crop", crop_arg]
    mod.main(_args(capture, rgb, scene, *extra))

    x0, _, x1, _ = box
    expected = (marker_col - x0) * W / (x1 - x0) - 0.5
    with np.load(_results(scene)) as npz:
        step = _crossing(npz["depth"][0, H // 2], 1.5)
        profile = npz["image"][0, :, :, 0].astype(np.float64).mean(axis=0)
        marker = (np.arange(W) * profile).sum() / profile.sum()

    assert abs(step - expected) <= 1.0
    assert abs(marker - expected) <= 1.0
    assert abs(step - marker) <= 1.0


def test_conf_is_nearest_and_inpaint_only_changes_depth_near_conf0(tmp_path):
    mod = _load_script()
    rng = np.random.default_rng(1)
    depth = rng.integers(1000, 1500, size=(DEPTH_WH[1], DEPTH_WH[0])).astype(np.uint16)  # no zero depth
    conf = _full_conf()
    conf[8:16, 10:20] = 0
    conf[0:4] = 1
    capture, rgb = _make_inputs(tmp_path, depth, conf)

    keep = tmp_path / "keep"
    inpaint = tmp_path / "inpaint"
    mod.main(_args(capture, rgb, keep))
    mod.main(_args(capture, rgb, inpaint, "--low-conf", "inpaint"))

    with np.load(_results(keep)) as npz:
        keep_depth = npz["depth"][0]
        keep_conf = npz["conf"]
    with np.load(_results(inpaint)) as npz:
        inpaint_depth = npz["depth"][0]
        inpaint_conf = npz["conf"]

    assert keep_conf.dtype == np.float32
    assert set(np.unique(keep_conf)) <= {0.0, 1.0, 2.0}
    assert set(np.unique(inpaint_conf)) <= {0.0, 1.0, 2.0}

    # keep fills nothing here (no zero depth), so every output pixel is the plain remap of the depth.
    map_x, map_y = _reference_maps((W, H), (0, 0, 240, 180))
    plain = cv2.remap(depth.astype(np.float32) / 1000.0, map_x, map_y,
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    np.testing.assert_allclose(keep_depth, plain, atol=1e-6)
    assert np.abs(inpaint_depth - keep_depth).max() > 1e-4

    # Telea writes only masked pixels, the conf-0 block here. An output pixel whose 2x2 depth footprint
    # misses that block must therefore come out identical in both modes.
    block = np.zeros((DEPTH_WH[1], DEPTH_WH[0]), bool)
    block[8:16, 10:20] = True
    x0 = np.clip(np.floor(map_x), 0, DEPTH_WH[0] - 1).astype(int)
    x1 = np.clip(x0 + 1, 0, DEPTH_WH[0] - 1)
    y0 = np.clip(np.floor(map_y), 0, DEPTH_WH[1] - 1).astype(int)
    y1 = np.clip(y0 + 1, 0, DEPTH_WH[1] - 1)
    touches = block[y0, x0] | block[y0, x1] | block[y1, x0] | block[y1, x1]
    assert (~touches).any()
    assert np.array_equal(inpaint_depth[~touches], keep_depth[~touches])


@pytest.mark.parametrize("low_conf", ["keep", "inpaint"])
def test_large_holes_leave_no_zero_nan_or_inf(tmp_path, low_conf):
    mod = _load_script()
    depth = _step_depth()
    depth[4:14, 6:20] = 0  # zero-depth block: filled in both modes
    conf = _full_conf()
    conf[2:8, 22:30] = 0  # conf-0 block: filled only with inpaint
    capture, rgb = _make_inputs(tmp_path, depth, conf)
    scene = tmp_path / "scene"

    mod.main(_args(capture, rgb, scene, "--low-conf", low_conf))

    with np.load(_results(scene)) as npz:
        out = npz["depth"]
    assert np.isfinite(out).all()
    assert (out > 0).all()
    # Telea can overshoot the boundary values by about a millimetre, so the range is not exact.
    assert out.min() >= 0.99 and out.max() <= 2.01


def test_check_passes_and_reports_a_plane_fit(tmp_path, capsys):
    mod = _load_script()
    depth, true_normal = _tilted_plane_depth()
    capture, rgb = _make_inputs(tmp_path, depth, _full_conf())
    scene = tmp_path / "scene"

    rc = mod.main(_args(capture, rgb, scene, "--check"))
    out = capsys.readouterr().out

    assert rc == 0
    assert "plane" in out and "rms=" in out
    with np.load(_results(scene)) as npz:
        normal, inlier_ratio, rms_m = mod.fit_plane(npz["depth"][0], npz["intrinsics"][0].astype(np.float64))
    assert rms_m < 0.005
    assert inlier_ratio > 0.99
    assert abs(normal @ true_normal) > 0.99


def _write_results(path, depth, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "image": np.zeros((1, H, W, 3), np.uint8),
        "depth": depth,
        "conf": np.full((1, H, W), 2, np.float32),
        "extrinsics": np.hstack([np.eye(3), np.zeros((3, 1))])[None].astype(np.float32),
        "intrinsics": np.array(K, np.float32).reshape(1, 3, 3),
    }
    arrays.update(extra or {})
    np.savez(str(path), **arrays)
    return path


def test_run_check_returns_0_for_a_clean_file(tmp_path):
    mod = _load_script()
    path = _write_results(tmp_path / "clean.npz", np.ones((1, H, W), np.float32))

    assert mod.run_check(path, {"K": K}, (W, H), (0, 0, 240, 180)) == 0


@pytest.mark.parametrize("case", ["zero_depth", "nan_depth", "float64_depth", "unknown_key"])
def test_run_check_returns_1_for_each_failure(tmp_path, case):
    mod = _load_script()
    depth = np.ones((1, H, W), np.float32)
    extra = None
    if case == "zero_depth":
        depth[0, 3, 5] = 0.0
    elif case == "nan_depth":
        depth[0, 3, 5] = np.nan
    elif case == "float64_depth":
        depth = depth.astype(np.float64)
    else:
        extra = {"mystery": np.zeros(3, np.float32)}
    path = _write_results(tmp_path / "bad.npz", depth, extra)

    assert mod.run_check(path, {"K": K}, (W, H), (0, 0, 240, 180)) == 1


def test_nearest_fill_takes_over_where_telea_leaves_zeros(monkeypatch):
    mod = _load_script()

    def telea_that_fills_nothing(src, mask, radius, flags):
        out = src.copy()
        out[mask.astype(bool)] = 0
        return out

    monkeypatch.setattr(mod.cv2, "inpaint", telea_that_fills_nothing)
    depth = np.full((DEPTH_WH[1], DEPTH_WH[0]), 1000, np.uint16)  # 1000 | hole | 2000
    depth[:, 10:20] = 0
    depth[:, 20:] = 2000

    filled = mod.fill_invalid_depth(depth, _full_conf(), "keep")

    assert (filled > 0).all()
    # Each hole pixel takes the nearest valid value: columns 10-14 are nearer to 1000, 15-19 to 2000.
    np.testing.assert_array_equal(filled[:, :15], 1000)
    np.testing.assert_array_equal(filled[:, 15:], 2000)


def test_inpaint_is_given_uint16_depth(monkeypatch):
    mod = _load_script()
    seen = []
    real_inpaint = cv2.inpaint

    def spy(src, mask, radius, flags):
        seen.append(src.dtype)
        return real_inpaint(src, mask, radius, flags)

    def no_nearest_fallback(*args, **kwargs):
        raise AssertionError("TELEA alone should fill this hole")

    monkeypatch.setattr(mod.cv2, "inpaint", spy)
    monkeypatch.setattr(mod, "distance_transform_edt", no_nearest_fallback)
    rng = np.random.default_rng(5)
    depth = rng.integers(800, 2501, size=(DEPTH_WH[1], DEPTH_WH[0])).astype(np.uint16)
    depth[4:20, 6:26] = 0  # 16 x 20 hole

    filled = mod.fill_invalid_depth(depth, _full_conf(), "keep")

    # Metre float32 fed to inpaint comes back negative, so the test pins the millimetre uint16 input.
    assert seen == [np.dtype(np.uint16)]
    assert filled.dtype == np.uint16
    assert (filled > 0).all()


def test_side_effects_are_confined_to_out_scene(tmp_path):
    mod = _load_script()
    capture, rgb = _make_inputs(tmp_path, _step_depth(), _full_conf())
    scene = tmp_path / "scene"

    def snapshot():
        return {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    before = snapshot()
    mod.main(_args(capture, rgb, scene, "--crop", CROP_ARG))
    after = snapshot()

    new_files = set(after) - set(before)
    assert new_files and all(scene in p.parents for p in new_files)
    for path, data in before.items():
        assert after[path] == data

    info = json.loads((scene / "s2_da/stage_info.json").read_text())
    assert info == {
        "success": True,
        "backend": "arkit_lidar",
        "capture": str(capture.resolve()),
        "frame_id": 7,
        "size": [W, H],
        "crop": list(CROP),
        "low_conf": "keep",
    }


def test_refuses_to_replace_a_result_it_did_not_write(tmp_path):
    mod = _load_script()
    capture, rgb = _make_inputs(tmp_path, _step_depth(), _full_conf())
    scene = tmp_path / "scene"
    existing = _results(scene)
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"da3 result")

    with pytest.raises(SystemExit):
        mod.main(_args(capture, rgb, scene))
    assert existing.read_bytes() == b"da3 result"
    assert not (scene / "s2_da/stage_info.json").exists()

    (scene / "s2_da/stage_info.json").write_text(json.dumps({"backend": "da3"}))
    with pytest.raises(SystemExit):
        mod.main(_args(capture, rgb, scene))
    assert existing.read_bytes() == b"da3 result"


def test_replaces_a_result_this_script_wrote(tmp_path):
    mod = _load_script()
    capture, rgb = _make_inputs(tmp_path, _step_depth(), _full_conf())
    scene = tmp_path / "scene"

    assert mod.main(_args(capture, rgb, scene)) == 0
    assert mod.main(_args(capture, rgb, scene, "--crop", CROP_ARG)) == 0

    with np.load(_results(scene)) as npz:
        assert npz["intrinsics"][0, 0, 0] == pytest.approx(K[0] * W / 188, rel=1e-6)


@pytest.mark.parametrize(
    "extra, frame_id",
    [
        (["--crop", "0,0,241,180"], 7),  # x1 past the RGB width
        (["--crop", "10,10,10,50"], 7),  # empty box
        (["--crop", "1,2,3"], 7),  # wrong number of values
        (["--size", "448"], 7),
        (["--size", "0x36"], 7),
        (["--size", "axb"], 7),
        ([], 99),  # no such frame
    ],
)
def test_bad_input_is_refused(tmp_path, extra, frame_id):
    mod = _load_script()
    capture, rgb = _make_inputs(tmp_path, _step_depth(), _full_conf())

    with pytest.raises(SystemExit):
        mod.main(_args(capture, rgb, tmp_path / "scene", *extra, frame_id=frame_id))


def test_rgb_png_with_another_aspect_ratio_is_refused(tmp_path):
    mod = _load_script()
    capture, rgb = _make_inputs(tmp_path, _step_depth(), _full_conf(), png_wh=(240, 120))

    with pytest.raises(SystemExit):
        mod.main(_args(capture, rgb, tmp_path / "scene"))
