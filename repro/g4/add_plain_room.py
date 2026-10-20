# SPDX-License-Identifier: Apache-2.0
"""Put plain walls, a floor and a ceiling around an auto_bg scene whose splat keeps only the desk.

desk_room_splat.py keeps the part of the background splat the video observed well, the desk top, and measures where
the room's surfaces are (<ply>.room.json). This writes those surfaces as one new visual-only object, `plain_room`,
into a step-7 scene state (7_build_og_scene_assets.py, check_bg_variant.sh): a box of six quads, the floor in the
splat's own carpet colour and the rest in its wall colour, so the robot's cameras see a room wherever the splat has
nothing, and never the sky.

  - The box is a new asset in the scene's own objects/ folder; no existing USD is touched (AGENTS.md, rule 1). The
    entry is additive: a consumer that does not know it loads one more fixed, visual-only object.
  - Every quad sets primvars:doNotCastShadows. The objects are lit by the simulator's dome light (the skybox), and a
    closed box would shadow them from it.
  - A camera outside the box would see the back of a wall, so the box grows until it holds every external camera
    of the teleop and DROID layouts (sim_franka_teleop, nv_franka_droid_high_res) and the robot's base: with
    --pick/--place at the pose scripted_pick_place.py and render_bg_views.py move the robot to, otherwise at the
    scene's own robot pose.

Usage (any env with numpy and pyyaml):
  python add_plain_room.py <scene_state.json> <room.json> [--pick iter_N --place iter_M] [--standoff 0.55]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
LAYOUTS = ("sim_franka_teleop", "nv_franka_droid_high_res")
CAMERA_CLEARANCE = 0.02  # m inside the walls
BASE_CLEARANCE = 0.30    # m, the arm's footprint around its base
OUTER = 1.0              # m between the box and its outer shell, beyond the longest near clipping distance


def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def quat_yaw(yaw):
    return [0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))]


def rotate(q, p):
    """Rotate p by the XYZW quaternion q."""
    x, y, z, w = q
    u = np.array([x, y, z])
    return p + 2 * np.cross(u, np.cross(u, p) + w * p)


def cameras(base_pos, base_ori):
    """Position and near clipping distance of every external camera of the layouts above, on a robot at this pose."""
    out = {}
    for layout in LAYOUTS:
        cfg = yaml.safe_load(open(REPO / "scripts" / "cfg" / "external_sensors" / f"{layout}.yaml"))
        for c in cfg["external_sensors"]:
            near = c.get("sensor_kwargs", {}).get("clipping_range", [0.001])[0]  # OmniGibson's VisionSensor default
            out[f"{layout}/{c['name']}"] = (np.asarray(base_pos) + rotate(base_ori, np.asarray(c["position"], float)),
                                            float(near))
    return out


def quad(name, corners, normal, material):
    pts = ", ".join(f"({x:.4f}, {y:.4f}, {z:.4f})" for x, y, z in corners)
    n = "({:.0f}, {:.0f}, {:.0f})".format(*normal)
    lo, hi = np.min(corners, 0), np.max(corners, 0)
    return f"""
        def Mesh "{name}" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            uniform bool doubleSided = 1
            float3[] extent = [({lo[0]:.4f}, {lo[1]:.4f}, {lo[2]:.4f}), ({hi[0]:.4f}, {hi[1]:.4f}, {hi[2]:.4f})]
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            rel material:binding = </plain_room/Looks/{material}>
            normal3f[] normals = [{n}, {n}, {n}, {n}] (
                interpolation = "vertex"
            )
            point3f[] points = [{pts}]
            bool primvars:doNotCastShadows = 1
            uniform token subdivisionScheme = "none"
        }}
"""


def material(name, rgb):
    return f"""
        def Material "{name}"
        {{
            token outputs:surface.connect = </plain_room/Looks/{name}/Shader.outputs:surface>

            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f})
                float inputs:metallic = 0
                float inputs:roughness = 0.9
                token outputs:surface
            }}
        }}
"""


def box(prefix, hu, hv, z0, z1):
    """Six inward-facing quads of a box centred on the origin in x, y, from z0 to z1."""

    def c(xs, ys, zs):
        return list(zip(xs, ys, zs, strict=True))

    return [
        quad(f"{prefix}floor", c([-hu, hu, hu, -hu], [-hv, -hv, hv, hv], [z0] * 4), (0, 0, 1), "floor"),
        quad(f"{prefix}ceiling", c([-hu, -hu, hu, hu], [-hv, hv, hv, -hv], [z1] * 4), (0, 0, -1), "wall"),
        quad(f"{prefix}wall_v_max", c([-hu, hu, hu, -hu], [hv] * 4, [z0, z0, z1, z1]), (0, -1, 0), "wall"),
        quad(f"{prefix}wall_v_min", c([hu, -hu, -hu, hu], [-hv] * 4, [z0, z0, z1, z1]), (0, 1, 0), "wall"),
        quad(f"{prefix}wall_u_max", c([hu] * 4, [hv, -hv, -hv, hv], [z0, z0, z1, z1]), (-1, 0, 0), "wall"),
        quad(f"{prefix}wall_u_min", c([-hu] * 4, [-hv, hv, hv, -hv], [z0, z0, z1, z1]), (1, 0, 0), "wall"),
    ]


def room_usda(hu, hv, z0, z1, wall_rgb, floor_rgb, comment):
    # A camera's near clipping plane can cut through a wall it stands close to, and the sky then shows
    # through: the teleop layout's external_cam0_opposite clips at 0.6 m and stands 16 cm from the wall
    # behind the desk in Cup in Bowl. A second box, OUTER beyond the first on every side, catches those rays.
    faces = box("", hu, hv, z0, z1) + box("outer_", hu + OUTER, hv + OUTER, z0 - OUTER, z1 + OUTER)
    return f"""#usda 1.0
(
    customLayerData = {{
        string comment = "{comment}"
    }}
    defaultPrim = "plain_room"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "plain_room"
{{
    def Xform "base_link"
    {{{''.join(faces)}    }}

    def Scope "Looks"
    {{{material("wall", wall_rgb)}{material("floor", floor_rgb)}    }}
}}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_state")
    ap.add_argument("room")
    ap.add_argument("--pick", default=None)
    ap.add_argument("--place", default=None)
    ap.add_argument("--standoff", type=float, default=0.55)
    ap.add_argument("--height", type=float, default=2.7, help="floor to ceiling (m); the video never saw a ceiling")
    args = ap.parse_args()
    state_path = Path(args.scene_state).resolve()
    state = json.loads(state_path.read_text())
    room = json.loads(Path(args.room).read_text())
    init = state["objects_info"]["init_info"]
    reg = state["state"]["registry"]["object_registry"]

    yaw = room["yaw"]
    to_uv = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    lim = {k: w["at"] for k, w in room["walls"].items()}
    z0 = room["floor"]["z"]
    z1 = z0 + args.height

    # What has to be inside: the robot (with its arm's footprint) and every camera that may render the room, where
    # the robot will stand. The pipeline's own robot pose is a placeholder out of reach of the objects
    # (scripted_pick_place.py); the demo and the view check move the robot in front of them first.
    base = reg["robot0"]["root_link"]
    poses = [("scene", np.asarray(base["pos"], float), base["ori"])]
    if args.pick and args.place:
        mid = (np.asarray(reg[args.pick]["root_link"]["pos"]) + np.asarray(reg[args.place]["root_link"]["pos"])) / 2
        poses = [("demo", np.array([mid[0] - args.standoff, mid[1], base["pos"][2]]), [0.0, 0.0, 0.0, 1.0])]
    need = []
    for tag, pos, ori in poses:
        need.append((f"{tag} robot base", pos, BASE_CLEARANCE))
        # A camera may stand just behind a wall when its near clipping plane hides that wall: at a tenth of the
        # clipping distance behind it, every ray within 84 degrees of the wall's normal is clipped first. The
        # teleop layout's external_cam0_opposite (0.6 m) stands 5 cm behind the wall in Marker in Cup; pushing
        # the wall out for it would open a gap between the desk and the wall.
        need += [(f"{tag} {name}", p, CAMERA_CLEARANCE if near / 10 < CAMERA_CLEARANCE else -near / 10)
                 for name, (p, near) in cameras(pos, ori).items()]
    moved = []
    for what, p, margin in need:
        u, v = to_uv @ p[:2]
        for key, val, sign in (("u_min", u, -1), ("u_max", u, 1), ("v_min", v, -1), ("v_max", v, 1)):
            want = val + sign * margin
            if sign * (want - lim[key]) > 0:
                moved.append(f"{key} {lim[key]:+.3f} -> {want:+.3f} for {what}")
                lim[key] = want
        if not z0 + margin < p[2] < z1 - margin:
            raise SystemExit(f"{what} at z {p[2]:.2f} is outside the floor-to-ceiling range {z0:.2f}..{z1:.2f}")

    cu, cv = (lim["u_min"] + lim["u_max"]) / 2, (lim["v_min"] + lim["v_max"]) / 2
    hu, hv = (lim["u_max"] - lim["u_min"]) / 2, (lim["v_max"] - lim["v_min"]) / 2
    centre = to_uv.T @ np.array([cu, cv])
    out_dir = state_path.parent / "objects" / "plain_room"
    out_dir.mkdir(parents=True, exist_ok=True)
    usd = out_dir / "plain_room.usda"
    usd.write_text(room_usda(hu, hv, z0, z1, srgb_to_linear(room["colors"]["wall"]),
                             srgb_to_linear(room["colors"]["floor"]),
                             f"Plain room for {state_path.name}, from {Path(args.room).name} "
                             f"(repro/g4/add_plain_room.py)"))
    init["plain_room"] = {
        "class_module": "omnigibson.objects.usd_object",
        "class_name": "USDObject",
        "args": {"name": "plain_room", "usd_path": str(usd), "category": "plain_room", "fixed_base": True,
                 "visual_only": True, "expected_file_hash": hashlib.md5(usd.read_bytes()).hexdigest()},
    }
    reg["plain_room"] = {"is_asleep": True, "root_link": {"pos": [float(centre[0]), float(centre[1]), 0.0],
                                                          "ori": quat_yaw(yaw), "lin_vel": [0.0, 0.0, 0.0],
                                                          "ang_vel": [0.0, 0.0, 0.0]}}
    state_path.write_text(json.dumps(state, indent=2))
    for m in moved:
        print(f"  moved {m}")
    print(f"plain room {2 * hu:.2f} x {2 * hv:.2f} m, floor z {z0:+.3f}, ceiling z {z1:+.3f}, yaw "
          f"{np.degrees(yaw):.1f} deg -> {usd}; added to {state_path.name}")


if __name__ == "__main__":
    main()
