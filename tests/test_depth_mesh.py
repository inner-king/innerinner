# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The optional depth-mesh background stage: geometry, texture mapping, glTF axes and the stage itself.

All tests use a synthetic scene (a camera 0.7 m above a flat support plane, a box standing on it), so
the expected positions are known in closed form. None needs a GPU. The geometry and file tests need only
numpy, PIL, scipy and trimesh, so they also run in the light editor's environment, where the test of the
editor's real mesh import (it needs OpenUSD) runs too; the tests of the stage, the plan and the config
need the pipeline environment and skip elsewhere.
"""

import ast
import json
import os
import pathlib
import struct
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from simfoundry.pipeline import depth_mesh
from simfoundry.pipeline.depth_mesh import (
    DEFAULT_PARAMS,
    WORLD_TO_GLTF,
    build_background_mesh,
    export_glb,
    find_stage5_inputs,
    frame_mask_to_grid,
    grid_triangles,
    padded_geometry,
    plane_intersections,
    snap_to_plane,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
STAGE = REPO_ROOT / "scripts/pipeline/A_reconstruction/stages/5c_depth_mesh_background.py"

H, W, FOCAL, CAM_HEIGHT = 48, 64, 80.0, 0.7
BOX_ROWS, BOX_COLS, BOX_HEIGHT = (18, 30), (26, 44), 0.1
K = np.array([[FOCAL, 0.0, 32.0], [0.0, FOCAL, 24.0], [0.0, 0.0, 1.0]])
# A camera looking straight down: image right = world +x, image down = world -y, optical axis = world -z.
CAM2WORLD = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, CAM_HEIGHT], [0.0, 0.0, 0.0, 1.0]])


def require_pipeline_env():
    pytest.importorskip("omegaconf")
    pytest.importorskip("hydra")


def scene_depth(with_box=True):
    depth = np.full((H, W), CAM_HEIGHT)
    if with_box:
        depth[BOX_ROWS[0]:BOX_ROWS[1], BOX_COLS[0]:BOX_COLS[1]] = CAM_HEIGHT - BOX_HEIGHT
    return depth


def plane_xy(rows, cols):
    """World (x, y) of the support-plane point under pixel (row, col) for this camera."""
    return (cols - K[0, 2]) * CAM_HEIGHT / FOCAL, -(rows - K[1, 2]) * CAM_HEIGHT / FOCAL


def box_mask_frame(scale=1):
    mask = np.zeros((H * scale, W * scale), dtype=bool)
    mask[BOX_ROWS[0] * scale:BOX_ROWS[1] * scale, BOX_COLS[0] * scale:BOX_COLS[1] * scale] = True
    return mask


def pattern_plate(scale=1):
    """A plate whose colour names its pixel, so a texture lookup says which pixel it came from."""
    rows, cols = np.meshgrid(np.arange(H * scale), np.arange(W * scale), indexing="ij")
    return np.stack([rows * 5 % 256, cols * 4 % 256, np.full_like(rows, 7)], axis=-1).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Frame mapping
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("grid_hw, frame_hw", [
    ((336, 448), (864, 1184)),   # stage 5's frame for a 4:3 depth grid: a little wider, so columns are padded
    ((336, 448), (864, 1152)),   # exactly 4:3, nothing padded
    ((336, 448), (1184, 864)),   # a taller frame, so rows are padded
    ((375, 500), (864, 1184)),
])
def test_padded_geometry_matches_pad_image_to_ratio(grid_hw, frame_hw):
    pytest.importorskip("cv2")
    from simfoundry.utils.processing_utils import pad_image_to_ratio

    padded, (delta_w, delta_h) = pad_image_to_ratio(
        np.zeros((*grid_hw, 3), dtype=np.uint8), target_ratio=frame_hw[1] / frame_hw[0], return_padding_size=True,
    )
    geometry = padded_geometry(grid_hw, frame_hw)

    assert geometry["padded_hw"] == padded.shape[:2]
    assert geometry["pad"] == (delta_w, delta_h)


def test_frame_mask_lands_where_stage_8_unpads_it():
    cv2 = pytest.importorskip("cv2")

    from simfoundry.utils.processing_utils import pad_image_to_ratio, unpad_image

    grid_hw, frame_hw = (336, 448), (864, 1184)
    frame = np.zeros(frame_hw, dtype=np.uint8)
    frame[300:520, 400:700] = 255
    padded, (delta_w, delta_h) = pad_image_to_ratio(
        np.zeros((*grid_hw, 3), dtype=np.uint8), target_ratio=frame_hw[1] / frame_hw[0], return_padding_size=True,
    )
    # Stage 8 resizes the frame to the padded grid and unpads it.
    reference = unpad_image(cv2.resize(frame, (padded.shape[1], padded.shape[0])), delta_w, delta_h) > 127

    ours = frame_mask_to_grid(frame, grid_hw)

    assert ours.shape == reference.shape == grid_hw
    iou = (ours & reference).sum() / (ours | reference).sum()
    assert iou > 0.95


def test_the_texture_drops_the_padding_columns():
    # A depth grid of the real size: 336 x 448 is padded by 6 columns a side to stage 5's 1184 x 864 frame.
    grid_k = np.array([[318.9, 0.0, 224.0], [0.0, 318.9, 168.0], [0.0, 0.0, 1.0]])
    frame = np.zeros((864, 1184, 3), dtype=np.uint8)
    frame[:, :, 1] = 120
    frame[:, :12] = (255, 0, 0)        # the model's padding, marked
    frame[:, -12:] = (255, 0, 0)
    _, texture, _ = build_background_mesh(np.full((336, 448), CAM_HEIGHT), grid_k, CAM2WORLD, frame)

    pixels = np.asarray(texture)
    assert not (pixels[..., 0] == 255).any()
    assert abs(texture.width / texture.height - 448 / 336) < 0.01


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------

def test_plane_pixels_become_vertices_on_the_support_plane():
    mesh, _, stats = build_background_mesh(scene_depth(False), K, CAM2WORLD, pattern_plate())

    assert stats["vertices"] == H * W
    assert np.allclose(mesh.vertices[:, 2], 0.0, atol=1e-9)
    # The pixel at the principal point is straight below the camera.
    centre = mesh.vertices[np.argmin(np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]))]
    assert np.allclose(centre[:2], 0.0, atol=1e-9)
    # Pixel (row 0, col 0) is up-left in the image: world -x, +y.
    corner_x, corner_y = plane_xy(0, 0)
    assert np.any(np.isclose(mesh.vertices[:, 0], corner_x) & np.isclose(mesh.vertices[:, 1], corner_y))


def test_the_plane_faces_up():
    mesh, _, _ = build_background_mesh(scene_depth(False), K, CAM2WORLD, pattern_plate())

    assert (mesh.face_normals[:, 2] > 0.999).all()


def test_box_top_sits_at_its_height_and_nothing_joins_it_to_the_plane():
    mesh, _, stats = build_background_mesh(scene_depth(), K, CAM2WORLD, pattern_plate())

    z = mesh.vertices[:, 2]
    assert np.isclose(z.max(), BOX_HEIGHT, atol=1e-9)
    on_box, on_plane = np.isclose(z, BOX_HEIGHT), np.isclose(z, 0.0, atol=1e-9)
    assert on_box.sum() == (BOX_ROWS[1] - BOX_ROWS[0]) * (BOX_COLS[1] - BOX_COLS[0])
    assert (on_box | on_plane).all()
    # A rubber sheet would put one face across both levels.
    spans = np.ptp(z[mesh.faces], axis=1)
    assert spans.max() < 1e-6
    assert stats["triangles_dropped_stretched"] > 0


def test_a_gentle_slope_is_kept_and_a_cliff_is_cut():
    rows = np.arange(H)[:, None] * np.ones((1, W))
    gentle = CAM_HEIGHT - 0.0003 * rows          # 0.3 mm of depth per pixel
    cliff = np.where(rows < H / 2, CAM_HEIGHT, CAM_HEIGHT - 0.3)
    _, _, kept = build_background_mesh(gentle, K, CAM2WORLD, pattern_plate())
    _, _, cut = build_background_mesh(cliff, K, CAM2WORLD, pattern_plate())

    assert kept["triangles_dropped_stretched"] == 0
    assert cut["triangles_dropped_stretched"] > 0


def test_grid_triangles_edge_ratio_is_one_for_a_face_on_surface_and_grows_with_slant():
    h, w = 20, 20
    points = np.zeros((h, w, 3))
    points[..., 0], points[..., 1] = np.meshgrid(np.arange(w) * 0.01, np.arange(h) * 0.01)
    points[..., 2] = 1.0
    z = points[..., 2]
    valid = np.ones((h, w), dtype=bool)
    # One pixel at depth 1 m and focal 100 px spans 0.01 m: neighbouring points here are exactly one footprint apart.
    # The limit is tight on purpose: a diagonal edge is sqrt(2) pixels long and must still count as one footprint.
    kept, counts = grid_triangles(valid, points, z, 100.0, max_edge_ratio=1.05)
    assert counts["triangles_dropped_stretched"] == 0 and len(kept) == 2 * (h - 1) * (w - 1)
    # Halving the footprint makes every edge twice as long as it may be.
    _, counts = grid_triangles(valid, points, z, 200.0, max_edge_ratio=1.05)
    assert counts["triangles_dropped_stretched"] == 2 * (h - 1) * (w - 1)


def test_zero_depth_leaves_no_vertex_and_is_dropped_as_invalid():
    depth = scene_depth(False)
    depth[10:14, 10:14] = 0.0
    mesh, _, stats = build_background_mesh(depth, K, CAM2WORLD, pattern_plate())

    assert stats["vertices"] == H * W - 16
    assert np.isfinite(mesh.vertices).all()
    # They are dropped for having no depth, not because the triangle happened to be stretched.
    assert stats["triangles_dropped_invalid"] > 0
    assert stats["triangles_dropped_stretched"] == 0


def test_a_nan_depth_leaves_no_vertex():
    depth = scene_depth(False)
    depth[20, 20] = np.nan
    mesh, _, stats = build_background_mesh(depth, K, CAM2WORLD, pattern_plate())

    assert stats["vertices"] == H * W - 1
    assert np.isfinite(mesh.vertices).all()


def test_far_pixels_are_dropped_by_z_far():
    depth = scene_depth(False)
    depth[:, :10] = 6.0
    _, _, stats = build_background_mesh(depth, K, CAM2WORLD, pattern_plate(), params={"z_far": 5.0})

    assert stats["vertices"] <= H * (W - 10)


def test_snapping_moves_only_vertices_near_the_plane():
    points = np.array([[0.0, 0.0, 0.004], [0.0, 0.0, -0.009], [0.0, 0.0, 0.02], [1.0, 2.0, 0.5]])
    snapped, mask = snap_to_plane(points, 0.01)

    assert snapped[:, 2].tolist() == [0.0, 0.0, 0.02, 0.5]
    assert mask.tolist() == [True, True, False, False]
    assert np.array_equal(snapped[:, :2], points[:, :2])


def test_a_counter_a_few_millimetres_off_the_plane_ends_up_on_it():
    depth = scene_depth(False) - 0.006        # the whole surface 6 mm above the plane
    snapped, _, _ = build_background_mesh(depth, K, CAM2WORLD, pattern_plate())
    kept, _, _ = build_background_mesh(depth, K, CAM2WORLD, pattern_plate(), params={"plane_snap_tol_m": 0.0})

    assert np.allclose(snapped.vertices[:, 2], 0.0)
    assert np.allclose(kept.vertices[:, 2], 0.006)


# --------------------------------------------------------------------------------------
# Object pixels
# --------------------------------------------------------------------------------------

def test_masked_pixels_are_replaced_by_where_their_ray_meets_the_plane():
    plain, _, _ = build_background_mesh(scene_depth(False), K, CAM2WORLD, pattern_plate())
    mesh, _, stats = build_background_mesh(
        scene_depth(), K, CAM2WORLD, pattern_plate(), object_mask=box_mask_frame(), params={"mask_dilate_px": 0},
    )

    # With the box masked, the surface is the plane again, vertex for vertex.
    assert stats["object_pixels_replaced_by_plane"] == (BOX_ROWS[1] - BOX_ROWS[0]) * (BOX_COLS[1] - BOX_COLS[0])
    assert np.allclose(mesh.vertices, plain.vertices)
    assert mesh.vertices[:, 2].max() < 1e-9
    assert stats["triangles_dropped_stretched"] == 0


def test_the_mask_grows_by_the_dilation_setting():
    _, _, none = build_background_mesh(
        scene_depth(), K, CAM2WORLD, pattern_plate(), object_mask=box_mask_frame(), params={"mask_dilate_px": 0})
    _, _, grown = build_background_mesh(
        scene_depth(), K, CAM2WORLD, pattern_plate(), object_mask=box_mask_frame(), params={"mask_dilate_px": 2})

    box = (BOX_ROWS[1] - BOX_ROWS[0]) * (BOX_COLS[1] - BOX_COLS[0])
    assert none["object_pixels_replaced_by_plane"] == box
    assert grown["object_pixels_replaced_by_plane"] == (BOX_ROWS[1] - BOX_ROWS[0] + 4) * (BOX_COLS[1] - BOX_COLS[0] + 4)


def test_a_mask_in_a_larger_frame_selects_the_same_pixels():
    plate = pattern_plate(scale=3)
    _, _, stats = build_background_mesh(
        scene_depth(), K, CAM2WORLD, plate, object_mask=box_mask_frame(scale=3), params={"mask_dilate_px": 0})

    assert stats["object_pixels_replaced_by_plane"] == (BOX_ROWS[1] - BOX_ROWS[0]) * (BOX_COLS[1] - BOX_COLS[0])


def test_a_ray_that_never_reaches_the_plane_leaves_a_hole_not_a_wild_point():
    # Looking up from below the plane: nothing a ray meets is on the plane in front of the camera.
    cam_below = CAM2WORLD.copy()
    cam_below[2, 3] = -0.7
    mask = np.ones((H, W), dtype=bool)
    mesh, _, stats = build_background_mesh(scene_depth(False), K, cam_below, pattern_plate(), mask, {"mask_dilate_px": 0})

    assert stats["object_pixels_without_plane_hit"] == H * W
    assert stats["vertices"] == 0 and len(mesh.faces) == 0


def test_plane_intersections_of_a_slanted_ray():
    point, valid = plane_intersections(np.array([[1.0, 0.0, -1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]), np.array([0.0, 0.0, 2.0]))

    assert valid.tolist() == [True, False, False]       # down; up; parallel
    assert np.allclose(point[0], [2.0, 0.0, 0.0])


# --------------------------------------------------------------------------------------
# Texture and file
# --------------------------------------------------------------------------------------

def read_glb(path):
    """The glTF JSON and the binary chunk of a .glb, parsed by hand so the checks do not lean on a loader."""
    data = pathlib.Path(path).read_bytes()
    magic, _, length = struct.unpack("<4sII", data[:12])
    assert magic == b"glTF"
    offset, document, binary = 12, None, None
    while offset < length:
        size, kind = struct.unpack("<I4s", data[offset:offset + 8])
        chunk = data[offset + 8:offset + 8 + size]
        offset += 8 + size
        if kind == b"JSON":
            document = json.loads(chunk)
        elif kind == b"BIN\x00":
            binary = chunk
    return document, binary


def accessor(document, binary, index):
    spec = document["accessors"][index]
    view = document["bufferViews"][spec["bufferView"]]
    assert "byteStride" not in view
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[spec["type"]]
    dtype = {5126: "<f4", 5125: "<u4", 5123: "<u2"}[spec["componentType"]]
    values = np.frombuffer(binary, dtype=dtype, count=spec["count"] * width,
                           offset=view.get("byteOffset", 0) + spec.get("byteOffset", 0))
    return values.reshape(-1, width) if width > 1 else values


def read_primitive(path):
    document, binary = read_glb(path)
    primitive = document["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]
    return {
        "document": document,
        "position": accessor(document, binary, attributes["POSITION"]).astype(np.float64),
        "uv": accessor(document, binary, attributes["TEXCOORD_0"]).astype(np.float64),
        "faces": accessor(document, binary, primitive["indices"]).reshape(-1, 3),
        "binary": binary,
    }


def gltf_to_world(position):
    """What the light editor's import does to a glTF: rotate +90 degrees about X (asset_import.prepare_mesh)."""
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    return position @ rotation.T


def test_the_exported_file_is_y_up_and_the_editor_import_restores_the_world(tmp_path):
    mesh, _, _ = build_background_mesh(scene_depth(), K, CAM2WORLD, pattern_plate())
    export_glb(mesh, tmp_path / "bg.glb")
    primitive = read_primitive(tmp_path / "bg.glb")

    file_position = primitive["position"]
    # The support plane (world z = 0) is the file's y = 0, and the camera's side of the scene (world -y) is file +z.
    assert np.allclose(file_position[:, 1].min(), 0.0, atol=1e-6)
    assert np.isclose(file_position[:, 1].max(), BOX_HEIGHT, atol=1e-6)
    world = gltf_to_world(file_position)
    assert np.allclose(world, mesh.vertices, atol=1e-6)
    assert np.allclose(world.max(axis=0) - world.min(axis=0), mesh.extents, atol=1e-6)
    # And the matrix this module writes with is the inverse of that rotation.
    assert np.allclose(WORLD_TO_GLTF[:3, :3] @ gltf_to_world(np.eye(3)).T, np.eye(3))


def test_the_real_editor_import_gives_back_the_world(tmp_path):
    pytest.importorskip("pxr")        # asset_import needs OpenUSD: run this in the simfoundry-editor env
    sys.path.insert(0, str(REPO_ROOT / "scripts/interactive/light_editor"))
    try:
        import asset_import
    finally:
        sys.path.pop(0)
    mesh, _, _ = build_background_mesh(scene_depth(), K, CAM2WORLD, pattern_plate())
    export_glb(mesh, tmp_path / "bg.glb")

    meshes, rotated = asset_import.prepare_mesh(tmp_path / "bg.glb")

    assert rotated
    assert len(meshes) == 1
    imported = meshes[0][2]
    assert np.allclose(imported.vertices, mesh.vertices, atol=1e-5)
    assert np.allclose(imported.bounds, mesh.bounds, atol=1e-5)


def test_the_texture_coordinates_name_the_pixel_each_vertex_came_from(tmp_path):
    plate = pattern_plate()
    mesh, _, _ = build_background_mesh(scene_depth(False), K, CAM2WORLD, plate)
    export_glb(mesh, tmp_path / "bg.glb")
    primitive = read_primitive(tmp_path / "bg.glb")

    image_view = primitive["document"]["bufferViews"][primitive["document"]["images"][0]["bufferView"]]
    start = image_view.get("byteOffset", 0)
    import io
    texture = np.asarray(Image.open(io.BytesIO(primitive["binary"][start:start + image_view["byteLength"]])).convert("RGB"))
    assert texture.shape == plate.shape

    world = gltf_to_world(primitive["position"])
    # Invert the camera model: which pixel is below this world point?
    cols = np.rint(world[:, 0] * FOCAL / CAM_HEIGHT + K[0, 2]).astype(int)
    rows = np.rint(-world[:, 1] * FOCAL / CAM_HEIGHT + K[1, 2]).astype(int)
    # glTF's texture origin is the top-left corner, with v running down.
    tex_cols = np.floor(primitive["uv"][:, 0] * texture.shape[1]).astype(int)
    tex_rows = np.floor(primitive["uv"][:, 1] * texture.shape[0]).astype(int)

    assert np.array_equal(tex_cols, cols) and np.array_equal(tex_rows, rows)
    # A pixel's centre, not its corner: the coordinate is (index + 0.5) / size, with v running down in glTF.
    assert np.allclose(primitive["uv"][:, 0], (cols + 0.5) / texture.shape[1], atol=1e-6)
    assert np.allclose(primitive["uv"][:, 1], (rows + 0.5) / texture.shape[0], atol=1e-6)
    assert np.array_equal(texture[tex_rows, tex_cols], plate[rows, cols])


def test_the_material_is_matte_and_untouched_by_lighting_defaults(tmp_path):
    mesh, _, _ = build_background_mesh(scene_depth(False), K, CAM2WORLD, pattern_plate())
    export_glb(mesh, tmp_path / "bg.glb")
    material = read_glb(tmp_path / "bg.glb")[0]["materials"][0]["pbrMetallicRoughness"]

    assert material["metallicFactor"] == 0.0
    assert material["roughnessFactor"] == 1.0


# --------------------------------------------------------------------------------------
# Stage 5 inputs
# --------------------------------------------------------------------------------------

def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_the_clean_plate_is_the_last_repainted_frame_and_masks_stop_there(tmp_path):
    for i in (0, 1, 3, 7):                                   # iteration numbers have gaps
        touch(tmp_path / "post_object_removal" / f"iter_{i}.png")
    for i in (0, 1, 3, 7, 9):                                # a mask beyond the last repainted frame
        touch(tmp_path / "removal_mask" / f"iter_{i}.png")
    touch(tmp_path / "removal_mask" / "iter_5.txt")

    found = find_stage5_inputs(tmp_path)

    assert found["iteration"] == 7
    assert found["plate"].endswith("post_object_removal/iter_7.png")
    assert [os.path.basename(p) for p in found["masks"]] == ["iter_0.png", "iter_1.png", "iter_3.png", "iter_7.png"]


def test_iterations_are_compared_as_numbers(tmp_path):
    for i in (2, 10):
        touch(tmp_path / "post_object_removal" / f"iter_{i}.png")

    assert find_stage5_inputs(tmp_path)["iteration"] == 10


def test_without_a_repainted_frame_the_source_frame_is_the_plate(tmp_path):
    touch(tmp_path / "source_padded_resized.png")

    found = find_stage5_inputs(tmp_path)

    assert found == {"plate": str(tmp_path / "source_padded_resized.png"), "masks": [], "iteration": None}


def test_stage_5_inputs_missing_is_an_error_that_names_the_cause(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run stage 5 first"):
        find_stage5_inputs(tmp_path)


# --------------------------------------------------------------------------------------
# The stage, end to end on a synthetic scene
# --------------------------------------------------------------------------------------

def write_scene(root, scene="synthetic"):
    scene_dir = root / scene
    (scene_dir / "s2_da/da/exports/npz").mkdir(parents=True)
    np.savez(
        scene_dir / "s2_da/da/exports/npz/results.npz",
        image=np.zeros((1, H, W, 3), dtype=np.uint8), depth=scene_depth()[None].astype(np.float32),
        intrinsics=K[None].astype(np.float32),
    )
    (scene_dir / "s4_frame").mkdir()
    np.save(scene_dir / "s4_frame/image_0_cam2world.npy", CAM2WORLD)
    (scene_dir / "s5_scene/post_object_removal").mkdir(parents=True)
    (scene_dir / "s5_scene/removal_mask").mkdir()
    Image.fromarray(pattern_plate(scale=2)).save(scene_dir / "s5_scene/post_object_removal/iter_0.png")
    Image.fromarray(box_mask_frame(scale=2).astype(np.uint8) * 255).save(scene_dir / "s5_scene/removal_mask/iter_0.png")
    return scene_dir


def run_stage(root, *overrides):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "CUDA_VISIBLE_DEVICES": ""}   # the worktree's simfoundry, not an installed one
    return subprocess.run(
        [sys.executable, str(STAGE), "scene_name=synthetic", f"root_dir={root}", "s3_ground.img_idx=0", *overrides],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
    )


def test_the_stage_writes_the_mesh_the_texture_and_its_info(tmp_path):
    require_pipeline_env()
    scene_dir = write_scene(tmp_path)

    proc = run_stage(tmp_path)

    assert proc.returncode == 0, proc.stderr[-2000:]
    out = scene_dir / "s_bg_mesh"
    info = json.loads((out / "stage_info.json").read_text())
    assert info["success"] is True
    assert info["clean_plate_iteration"] == 0 and info["removal_masks_used"] == 1
    assert info["file_up_axis"] == "Y"
    # The box was masked, so the whole mesh is the support plane.
    assert info["object_pixels_replaced_by_plane"] > 0
    primitive = read_primitive(out / "background_mesh.glb")
    assert np.allclose(primitive["position"][:, 1], 0.0, atol=1e-6)
    assert len(primitive["position"]) == info["vertices"]
    assert (out / "background_texture.png").is_file()


def test_the_stage_honours_its_settings(tmp_path):
    require_pipeline_env()
    write_scene(tmp_path)

    proc = run_stage(tmp_path, "s_bg_mesh.mask_dilate_px=0", "s_bg_mesh.z_far=0.65")
    info = json.loads((tmp_path / "synthetic/s_bg_mesh/stage_info.json").read_text())

    assert proc.returncode == 0, proc.stderr[-2000:]
    assert info["params"]["mask_dilate_px"] == 0
    assert info["params"]["z_far"] == 0.65
    # 0.65 m is nearer than the support plane (0.7 m), so only the masked box pixels survive.
    assert info["vertices"] < H * W


# --------------------------------------------------------------------------------------
# Off by default, and the settings agree
# --------------------------------------------------------------------------------------

def test_the_stage_is_not_in_the_default_plan_and_follows_stage_5_when_asked_for():
    require_pipeline_env()
    from simfoundry.pipeline.orchestrator import get_stage_plan

    default = [spec.stage_id for spec in get_stage_plan("video", pipeline_name="reconstruction")]
    with_mesh = get_stage_plan("video", pipeline_name="reconstruction", bg_mesh=True)
    ids = [spec.stage_id for spec in with_mesh]

    assert "5c" not in default
    assert ids == default[:default.index("5") + 1] + ["5c"] + default[default.index("5") + 1:]
    spec = with_mesh[ids.index("5c")]
    assert spec.script == "scripts/pipeline/A_reconstruction/stages/5c_depth_mesh_background.py"
    assert spec.env == "simfoundry" and spec.cfg_key == "s_bg_mesh"


def test_the_command_line_flag_reaches_the_plan():
    require_pipeline_env()
    cmd = [sys.executable, str(REPO_ROOT / "scripts/pipeline/A_reconstruction/run_reconstruction.py"),
           "--dry-run", "--exec-mode", "direct", "--bg-mesh", "--include", "5c", "scene_name=x"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT),
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})

    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "5c_depth_mesh_background.py" in proc.stdout
    without = subprocess.run(cmd[:cmd.index("--bg-mesh")] + cmd[cmd.index("--bg-mesh") + 1:], capture_output=True, text=True,
                             cwd=str(REPO_ROOT), env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    assert "5c_depth_mesh_background.py" not in without.stdout


def test_the_shell_wrapper_forwards_the_flag():
    text = (REPO_ROOT / "scripts/pipeline/A_reconstruction/run.sh").read_text()

    assert "--bg-mesh)" in text and 'CMD+=("--bg-mesh")' in text


def test_the_config_section_and_the_module_defaults_agree():
    require_pipeline_env()
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(OmegaConf.load(REPO_ROOT / "scripts/cfg/real2sim_cfg.yaml").s_bg_mesh)

    assert {key: config[key] for key in DEFAULT_PARAMS} == DEFAULT_PARAMS
    assert config["out_dirname"] == "s_bg_mesh"


def test_the_stage_script_reads_every_setting_and_writes_through_the_module():
    tree = ast.parse(STAGE.read_text())
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                and node.module == "simfoundry.pipeline.depth_mesh" for alias in node.names}
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}

    assert {"DEFAULT_PARAMS", "build_background_mesh", "export_glb", "find_stage5_inputs"} <= imported
    assert {"build_background_mesh", "export_glb", "find_stage5_inputs", "finalize_stage"} <= calls
