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
    drop_small_components,
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
    # The border trim has its own test; here every pixel must stay, so the plane's geometry is what is checked.
    mesh, _, stats = build_background_mesh(
        scene_depth(False), K, CAM2WORLD, pattern_plate(), params={"border_trim_px": 0})

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
    mesh, _, stats = build_background_mesh(depth, K, CAM2WORLD, pattern_plate(), params={"border_trim_px": 0})

    assert stats["vertices"] == H * W - 16
    assert np.isfinite(mesh.vertices).all()
    # They are dropped for having no depth, not because the triangle happened to be stretched.
    assert stats["triangles_dropped_invalid"] > 0
    assert stats["triangles_dropped_stretched"] == 0


def test_a_nan_depth_leaves_no_vertex():
    depth = scene_depth(False)
    depth[20, 20] = np.nan
    mesh, _, stats = build_background_mesh(depth, K, CAM2WORLD, pattern_plate(), params={"border_trim_px": 0})

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


# --------------------------------------------------------------------------------------
# Surfaces under objects, the two layers, the border and the crumbs
# --------------------------------------------------------------------------------------

GRID = 40
RING_BOX = (9, 31)  # a 22 x 22 mask at rows and columns 9..30 of a 40 x 40 grid


def tilted_camera(theta_deg):
    """A camera theta degrees from straight down, a grazing view of the support plane."""
    t = np.radians(theta_deg)
    cam = np.eye(4)
    cam[:3, :3] = [[1.0, 0.0, 0.0], [0.0, -np.cos(t), np.sin(t)], [0.0, -np.sin(t), -np.cos(t)]]
    cam[:3, 3] = [0.0, 0.0, CAM_HEIGHT]
    return cam


def depth_of_surface(cam2world, z_level):
    """The depth map of a horizontal surface at world height z_level, as seen from cam2world."""
    rays = depth_mesh.pixel_rays((H, W), K)
    rays_world = rays @ cam2world[:3, :3].T
    return (z_level - cam2world[2, 3]) / rays_world[..., 2]


def ring_case(ring_heights_of, ring_px=1, box=RING_BOX):
    """A square mask on a flat grid at 12.5 mm, and the heights of its ring as ring_heights_of(ring_index, row, col)."""
    from scipy.ndimage import binary_dilation

    mask = np.zeros((GRID, GRID), dtype=bool)
    mask[box[0]:box[1], box[0]:box[1]] = True
    ring = binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=ring_px) & ~mask
    rows, cols = np.nonzero(ring)
    z = np.full((GRID, GRID), 0.0125)
    z[rows, cols] = [ring_heights_of(i, r, c) for i, (r, c) in enumerate(zip(rows, cols))]
    grid_rows, grid_cols = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    points = np.stack([grid_cols.astype(float), grid_rows.astype(float), z], axis=-1)
    return mask, points, ~mask


def test_a_ring_with_two_surfaces_is_filled_at_the_wider_one_not_at_the_median():
    # Interleaved, as a real ring is: laid out in order, each surface would be a band a tilted plane fits.
    heights = np.array([-0.0325] * 25 + [-0.0075] * 25 + [0.0125] * 42)
    heights = heights[np.random.default_rng(0).permutation(len(heights))]
    mask, points, usable = ring_case(lambda i, r, c: heights[i])
    _, _, c, kind = depth_mesh.local_support_planes(mask, points, usable, {**DEFAULT_PARAMS, "ring_px": 1})

    assert set(np.unique(kind[mask]).tolist()) == {2}
    assert np.allclose(c[mask], 0.0125)
    # The median of this ring is -7.5 mm, a height neither surface has, so a median rule would fail here.
    assert np.median(heights) == -0.0075


def test_a_tilted_ring_is_filled_horizontally_at_its_dominant_height():
    mask, points, usable = ring_case(lambda i, r, c: 0.1 * c)
    a, b, _, kind = depth_mesh.local_support_planes(mask, points, usable, {**DEFAULT_PARAMS, "ring_px": 1})

    assert set(np.unique(kind[mask]).tolist()) == {2}
    assert np.allclose(a[mask], 0.0) and np.allclose(b[mask], 0.0)


def test_a_flat_ring_gives_the_fitted_local_plane():
    rng = np.random.default_rng(3)
    mask, points, usable = ring_case(lambda i, r, c: rng.uniform(-0.001, 0.001))
    a, b, c, kind = depth_mesh.local_support_planes(mask, points, usable, {**DEFAULT_PARAMS, "ring_px": 1})

    assert set(np.unique(kind[mask]).tolist()) == {1}
    assert np.allclose(c[mask], 0.0, atol=2e-3)


def test_a_ring_too_small_to_read_stays_on_the_support_plane():
    mask, points, usable = ring_case(lambda i, r, c: 0.02, ring_px=1, box=(19, 21))  # a 2 x 2 mask: 12 ring pixels
    _, _, c, kind = depth_mesh.local_support_planes(mask, points, usable, {**DEFAULT_PARAMS, "ring_px": 1})

    assert set(np.unique(kind[mask]).tolist()) == {0}
    assert np.allclose(c[mask], 0.0)


def test_mask_surface_plane_keeps_every_object_on_the_support_plane():
    heights = [-0.0325] * 25 + [-0.0075] * 25 + [0.0125] * 42
    mask, points, usable = ring_case(lambda i, r, c: heights[i])
    _, _, c, kind = depth_mesh.local_support_planes(mask, points, usable, {**DEFAULT_PARAMS, "mask_surface": "plane"})

    assert set(np.unique(kind).tolist()) == {0}
    assert np.allclose(c, 0.0)


def test_mask_surface_other_than_local_or_plane_is_an_error():
    with pytest.raises(ValueError):
        build_background_mesh(scene_depth(False), K, CAM2WORLD, pattern_plate(), params={"mask_surface": "fitted"})


def test_the_ring_fit_ignores_points_that_belong_to_a_neighbour():
    x, y = np.meshgrid(np.arange(20.0), np.arange(20.0))
    z = 0.001 * x + 0.002 * y + 0.01
    points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    points[:15, 2] += 0.2  # a neighbouring object's pixels in the ring

    a, b, c, _ = depth_mesh.fit_plane_robust(points)

    assert abs(a - 0.001) < 1e-6 and abs(b - 0.002) < 1e-6 and abs(c - 0.01) < 1e-6
    assert depth_mesh.fit_plane_robust(points[:2]) is None


def test_a_ring_point_of_a_neighbouring_object_does_not_change_the_ring():
    # Two objects one pixel apart. The object at the top is 10 cm high; the other's ring must not include it.
    mask = np.zeros((GRID, GRID), dtype=bool)
    mask[5:15, 5:15] = True
    mask[17:27, 5:15] = True
    tops = np.zeros((GRID, GRID))
    tops[mask] = 0.1
    grid_rows, grid_cols = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    points = np.stack([grid_cols.astype(float), grid_rows.astype(float), np.where(mask, tops, 0.0)], axis=-1)
    _, _, c, kind = depth_mesh.local_support_planes(mask, points, ~mask, {**DEFAULT_PARAMS, "ring_px": 6})

    assert set(np.unique(kind[mask]).tolist()) == {1}
    assert np.allclose(c[mask], 0.0, atol=1e-9)


def test_each_ring_gets_its_own_plane():
    mask = np.zeros((GRID, GRID), dtype=bool)
    mask[4:12, 4:12] = True
    mask[4:12, 26:34] = True
    z = np.zeros((GRID, GRID))
    z[:, :20] = 0.01
    z[:, 20:] = -0.02
    grid_rows, grid_cols = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    points = np.stack([grid_cols.astype(float), grid_rows.astype(float), z], axis=-1)
    _, _, c, kind = depth_mesh.local_support_planes(mask, points, ~mask, {**DEFAULT_PARAMS, "ring_px": 3})

    assert np.allclose(c[4:12, 4:12], 0.01)
    assert np.allclose(c[4:12, 26:34], -0.02)
    assert set(np.unique(kind[mask]).tolist()) == {1}


def test_the_three_surface_counts_add_up_to_the_object_pixels():
    depth = scene_depth(False)
    depth[:BOX_ROWS[0] - 2] -= 0.02  # a step in the surround, so the ring can hold two heights
    _, _, stats = build_background_mesh(depth, K, CAM2WORLD, pattern_plate(), box_mask_frame(), params={"mask_dilate_px": 0})

    assert stats["object_pixels_ring_level"] > 0  # the scene must exercise the ring-level term
    split = (stats["object_pixels_local_surface"] + stats["object_pixels_ring_level"]
             + stats["object_pixels_support_plane"])
    assert split == stats["object_pixels_replaced_by_plane"]
    assert stats["object_pixels_replaced_by_plane"] + stats["object_pixels_without_plane_hit"] == (
        (BOX_ROWS[1] - BOX_ROWS[0]) * (BOX_COLS[1] - BOX_COLS[0]))
    assert stats["triangles_plane_layer"] + stats["triangles_relief"] == stats["triangles"]


def test_the_plane_layer_is_not_judged_by_the_relief_limit():
    # Edges about six pixel footprints long: over the relief limit of 4, under the plane limit of 8.
    h, w = 4, 6
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    points = np.stack([cols * 0.06, rows * 0.01, np.ones((h, w))], axis=-1)
    z_cam = np.ones((h, w))
    valid = np.ones((h, w), dtype=bool)

    _, relief = grid_triangles(valid, points, z_cam, 100.0, max_edge_ratio=4.0)
    _, plane = grid_triangles(valid, points, z_cam, 100.0, max_edge_ratio=4.0,
                              plane_mask=np.ones((h, w), dtype=bool), plane_max_edge_ratio=8.0)

    assert relief["triangles_dropped_stretched"] == relief["triangles_total"]
    assert plane["triangles_dropped_stretched"] == 0


def test_a_relief_triangle_is_judged_by_the_relief_limit_even_when_a_plane_limit_is_given():
    h, w = 4, 6
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    points = np.stack([cols * 0.06, rows * 0.01, np.ones((h, w))], axis=-1)
    _, counts = grid_triangles(np.ones((h, w), dtype=bool), points, np.ones((h, w)), 100.0, max_edge_ratio=4.0,
                               plane_mask=np.zeros((h, w), dtype=bool), plane_max_edge_ratio=8.0)

    assert counts["triangles_dropped_stretched"] == counts["triangles_total"]


def test_a_mask_edge_on_the_counter_does_not_crack_the_plane_layer():
    # A grazing view (60 degrees) of the counter, 8 mm above the support plane, with an object patch moved onto
    # z = 0 at the mask edge. Both sides are plane pixels, but the 894363c limit of 4 cuts their edges: a crack.
    depth = depth_of_surface(tilted_camera(60), 0.008)
    depth[BOX_ROWS[0]:BOX_ROWS[1], BOX_COLS[0]:BOX_COLS[1]] -= 0.1 / np.cos(np.radians(60))
    shared = {"mask_dilate_px": 0, "mask_surface": "plane", "border_trim_px": 0, "min_component_triangles": 0,
              "relief_max_edge_ratio": 4.0}
    _, _, as_894363c = build_background_mesh(depth, K, tilted_camera(60), pattern_plate(), box_mask_frame(),
                                             {**shared, "plane_max_edge_ratio": 4.0})
    _, _, now = build_background_mesh(depth, K, tilted_camera(60), pattern_plate(), box_mask_frame(),
                                      {**shared, "plane_max_edge_ratio": 8.0})

    assert as_894363c["triangles_dropped_stretched_plane"] > 0
    assert now["triangles_dropped_stretched_plane"] == 0


def test_a_surface_below_the_support_plane_is_not_snapped_onto_it():
    depth = depth_of_surface(CAM2WORLD, -0.2)
    mesh, _, stats = build_background_mesh(depth, K, CAM2WORLD, pattern_plate(), params={"border_trim_px": 0})

    assert np.allclose(mesh.vertices[:, 2], -0.2)
    assert stats["vertices_snapped_to_plane"] == 0


def test_the_border_pixels_are_trimmed_with_their_triangles():
    _, _, stats = build_background_mesh(scene_depth(False), K, CAM2WORLD, pattern_plate(), params={"border_trim_px": 4})

    assert stats["vertices"] == (H - 8) * (W - 8)
    assert stats["vertices_trimmed_border"] == H * W - (H - 8) * (W - 8)


def test_small_pieces_of_the_triangulation_are_dropped_and_large_ones_kept():
    strip = np.array([[i, i + 1, i + 2] for i in range(5)])
    big = np.array([[100 + i, 101 + i, 102 + i] for i in range(300)])
    triangles = np.concatenate([strip, big])

    kept, dropped = drop_small_components(triangles, 500, 200)

    assert dropped == 5
    assert len(kept) == 300 and kept.min() >= 100
    assert len(drop_small_components(triangles, 500, 0)[0]) == len(triangles)


def test_no_triangle_is_emitted_twice():
    mesh, _, _ = build_background_mesh(scene_depth(), K, CAM2WORLD, pattern_plate(), box_mask_frame())
    keys = np.sort(mesh.faces, axis=1)

    assert len({tuple(k) for k in keys.tolist()}) == len(keys)


def test_the_synthetic_scenes_reproduce_894363c_with_the_regression_settings():
    # Recorded by running 894363c's depth_mesh on the same scenes: counts only, the settings that match it.
    regression = {"mask_surface": "plane", "border_trim_px": 0, "min_component_triangles": 0,
                  "relief_max_edge_ratio": 4.0, "plane_max_edge_ratio": 4.0}
    _, _, masked = build_background_mesh(scene_depth(), K, CAM2WORLD, pattern_plate(), box_mask_frame(), regression)
    _, _, measured = build_background_mesh(scene_depth(), K, CAM2WORLD, pattern_plate(), None, regression)

    assert (masked["vertices"], masked["triangles"]) == (3072, 5922)
    assert (measured["vertices"], measured["triangles"]) == (3072, 5804)


def test_missing_depth_next_to_an_object_stays_out_of_its_ring():
    # Depth 0 and NaN just outside the mask: a zero depth maps to the camera's own position and a NaN to
    # nothing. Either must not pull the surface an object is filled with off the counter's plane.
    for missing in (0.0, np.nan):
        depth = scene_depth(False)
        depth[BOX_ROWS[0] - 6:BOX_ROWS[0], BOX_COLS[0] - 6:BOX_COLS[1] + 6] = missing
        depth[BOX_ROWS[1]:BOX_ROWS[1] + 6, BOX_COLS[0] - 6:BOX_COLS[1] + 6] = missing
        mesh, _, stats = build_background_mesh(depth, K, CAM2WORLD, pattern_plate(), box_mask_frame(),
                                               params={"mask_dilate_px": 0, "border_trim_px": 0})

        assert np.isfinite(mesh.vertices).all()
        assert stats["object_pixels_local_surface"] + stats["object_pixels_ring_level"] + stats["object_pixels_support_plane"] \
            == stats["object_pixels_replaced_by_plane"]
        # The object is on the counter, so its pixels land at z = 0 whichever surface the ring gives.
        assert np.allclose(mesh.vertices[:, 2], 0.0, atol=1e-6)


def test_a_noise_free_plane_ring_gives_the_local_plane():
    mask, points, usable = ring_case(lambda i, r, c: 0.02 * c)
    a, _, c, kind = depth_mesh.local_support_planes(mask, points, usable, {**DEFAULT_PARAMS, "ring_px": 1})

    assert set(np.unique(kind[mask]).tolist()) == {1}
    assert np.allclose(a[mask], 0.02)


def test_one_ring_point_three_millimetres_off_still_gives_the_local_plane():
    def heights(i, r, c):
        return 0.02 * c + (0.003 if i == 0 else 0.0)

    mask, points, usable = ring_case(heights)
    a, _, _, kind = depth_mesh.local_support_planes(mask, points, usable, {**DEFAULT_PARAMS, "ring_px": 1})

    assert set(np.unique(kind[mask]).tolist()) == {1}
    assert np.allclose(a[mask], 0.02, atol=1e-3)


def test_a_mask_is_classed_as_plane_layer_by_where_it_lands_not_by_where_it_was_measured():
    # The box top is measured 10 cm up; the mask moves it onto the plane, so its triangles are plane layer.
    _, _, masked = build_background_mesh(scene_depth(), K, CAM2WORLD, pattern_plate(), box_mask_frame(),
                                         params={"mask_dilate_px": 0, "border_trim_px": 0})
    _, _, unmasked = build_background_mesh(scene_depth(), K, CAM2WORLD, pattern_plate(), None,
                                           params={"border_trim_px": 0})

    assert masked["triangles_relief"] == 0
    assert masked["triangles_plane_layer"] == masked["triangles"]
    assert unmasked["triangles_relief"] > 0


def test_plane_intersections_at_a_height_above_zero_hit_that_height():
    point, valid = plane_intersections(np.array([[1.0, 0.0, -1.0]]), np.array([0.0, 0.0, 2.0]), height=0.5)

    assert valid.tolist() == [True]
    assert np.allclose(point[0], [1.5, 0.0, 0.5])


def test_a_tie_between_two_height_bands_goes_to_the_lower_one():
    z = np.array([0.0125] * 10 + [0.0275] * 10)

    assert depth_mesh.dominant_height(z) == 0.0125


def test_vertices_snapped_counts_only_the_vertices_the_mesh_uses():
    _, _, stats = build_background_mesh(scene_depth(False), K, CAM2WORLD, pattern_plate(), params={"border_trim_px": 4})

    assert stats["vertices_snapped_to_plane"] == stats["vertices"]
    assert stats["vertices_snapped_to_plane"] < H * W


def test_the_two_millimetre_floor_keeps_a_point_that_is_nearly_on_a_flat_ring():
    # A flat ring with one point 1.9 mm off: within the 2 mm floor, so the fit keeps it. Without the floor the
    # spread of this noise-free ring is zero and the point is dropped.
    x, y = np.meshgrid(np.arange(8.0), np.arange(8.0))
    x, y = x.ravel(), y.ravel()
    z = 0.02 * x
    z[5] += 0.0019
    design = np.column_stack([x, y, np.ones_like(x)])
    c_all = np.linalg.lstsq(design, z, rcond=None)[0][2]
    keep = np.ones(len(z), dtype=bool)
    keep[5] = False
    c_clean = np.linalg.lstsq(design[keep], z[keep], rcond=None)[0][2]

    _, _, c, _ = depth_mesh.fit_plane_robust(np.stack([x, y, z], axis=1))

    assert np.isclose(c, c_all, atol=1e-9) and not np.isclose(c_all, c_clean, atol=1e-9)


def test_the_snapped_count_is_the_plane_vertices_the_mesh_uses():
    # A 3 x 3 island of counter cut off by a 50 cm trench: its pixels are plane pixels, but no triangle uses them.
    depth = scene_depth(False)
    trench = np.zeros(depth.shape, dtype=bool)
    trench[18:25, 28:35] = True
    island = np.zeros(depth.shape, dtype=bool)
    island[20:23, 30:33] = True
    depth[trench & ~island] = CAM_HEIGHT + 0.5
    mesh, _, stats = build_background_mesh(depth, K, CAM2WORLD, pattern_plate(), params={"border_trim_px": 0})

    assert stats["vertices_snapped_to_plane"] == int((np.abs(mesh.vertices[:, 2]) < 0.01).sum())
    assert stats["vertices_snapped_to_plane"] < int((np.abs(depth - CAM_HEIGHT) < 0.01).sum())
