"""MPM 식재료 ↔ PhysX 강체 넘겨주기 (2026-11-24, reports/67, 계획 docs/plan_2026-11-24_force_pipeline_grasp.md 5절).

왜: 식재료는 Warp MPM 솔버 안에만 있어 로봇 집게(PhysX)와 부딪히지 않는다(팀원 README 'PhysX 쪽의 두부'). 한 팔 로봇은 칼을 쥔 동안 식재료를
잡을 수 없으므로, 과제가 바뀌는 때에만 넘겨준다.
- pieces_to_rigid(): 썰기가 끝난 뒤 MPM 조각(아직 붙은 조각은 한 묶음, sim.piece_groups)마다 PhysX 강체(USDObject)를 만든다.
  모양 = 그 조각 입자의 볼록 껍질(시각·충돌 같음. 입자 가운데를 지나므로 실제 겉면보다 입자 반 간격 안쪽 → 칼질 평면으로 갈린 이웃과 안 겹침).
  질량 = 입자 질량 × 입자 수, 무게중심·관성 = 입자로. 색 = 조각 입자 색 평균(속살·껍질).
  팀원 rule.py(강체 썰기)와 같은 순서: USD 쓰기 → og.sim.batch_add_objects → 바로 초기화(rule._initialize_now 와 같은 식) → 자세·속도.
- food_rigid_usd(): 썰기 전 통 식재료를 강체로(놓기 과제). 모양 = --shape 의 입자 틀 볼록 껍질(우리 좌표 = 모양 좌표).
한계: 볼록 껍질이라 양파 겹처럼 오목한 모양은 메워진다. 강체가 된 조각은 휘지 않는다.
"""
import math
from pathlib import Path

import numpy as np


def _hull(points):
    """점들의 볼록 껍질 (V, F) — 바깥쪽 법선(팀원 write_knife_usd 와 같은 식)."""
    from scipy.spatial import ConvexHull

    try:
        h = ConvexHull(points)
    except Exception:  # 11-26 reports/80: 납작·겹친 점에서 qhull 오류(QH6421) — 흔들기(QJ)로 한 번 더, 그래도 안 되면 부른 쪽이 건너뜀
        h = ConvexHull(points, qhull_options="QJ")
    V = points[h.vertices]
    remap = np.full(len(points), -1)
    remap[h.vertices] = np.arange(len(h.vertices))
    F = remap[h.simplices]
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.einsum("ij,ij->i", n, tri[:, 0] - V.mean(0)) >= 0
    F[~out] = F[~out][:, ::-1]
    return V, F


def _hull_vertex_ids(points):
    from scipy.spatial import ConvexHull

    return ConvexHull(points).vertices


def _mass_props(local, m_particle):
    """입자(무게중심 기준 자리)로 질량·관성(주축 대각·주축 회전 xyzw)."""
    from scipy.spatial.transform import Rotation

    m = m_particle * len(local)
    r2 = np.einsum("ij,ij->i", local, local)
    I = m_particle * (np.eye(3) * r2.sum() - local.T @ local)
    w, axes = np.linalg.eigh(I)
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    w = np.maximum(w, 1e-12)
    return float(m), w, Rotation.from_matrix(axes).as_quat()


def write_rigid_usd(path, name, category, V, F, rgb, mass, inertia_diag, axes_xyzw, roughness=0.55):
    """한 링크 강체 USD(팀원 og_cosim.write_knife_usd 와 같은 짜임: /<name>/base_link/{visuals,collisions})."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from simfoundry.cutting.og_cosim import _identity_ops, _mesh, _omni_pbr, srgb_to_linear

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    _identity_ops(root)
    root.GetPrim().CreateAttribute("ig:category", Sdf.ValueTypeNames.String).Set(category)
    UsdGeom.Scope.Define(stage, f"/{name}/Looks")
    look = _omni_pbr(stage, f"/{name}/Looks/food", srgb_to_linear(np.clip(np.asarray(rgb, float), 0, 1)), roughness=roughness)
    link = UsdGeom.Xform.Define(stage, f"/{name}/base_link")
    _identity_ops(link)
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    ma = UsdPhysics.MassAPI.Apply(link.GetPrim())
    ma.CreateMassAttr(float(mass))
    ma.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia_diag)))
    qx, qy, qz, qw = map(float, axes_xyzw)
    ma.CreatePrincipalAxesAttr(Gf.Quatf(qw, qx, qy, qz))
    UsdGeom.Xform.Define(stage, f"/{name}/base_link/visuals")
    UsdGeom.Xform.Define(stage, f"/{name}/base_link/collisions")
    Vu = V[F.reshape(-1)]  # 꼭짓점을 안 나누면 매끈하게, 나누면 면마다 평평하게 — 칼로 썬 면이 보이게 나눔
    vis = _mesh(stage, f"/{name}/base_link/visuals/mesh", Vu, np.arange(len(Vu)).reshape(-1, 3))
    UsdShade.MaterialBindingAPI.Apply(vis.GetPrim()).Bind(look)
    col = _mesh(stage, f"/{name}/base_link/collisions/hull", V, F, purpose=UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr("convexHull")
    root.GetPrim().CreateAttribute("ig:nativeBB", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*(V.max(0) - V.min(0)).tolist()))
    stage.GetRootLayer().Save()
    return str(path.resolve())


def initialize_now(og, scene, objs):
    """방금 더한 물체를 지금 초기화(팀원 simfoundry/cutting/rule.py PlaneCuttingRule._initialize_now 와 같은 식, OmniGibson 3.x).
    안 하면 다음 단계에 초기화되며 keep_still 로 준 속도를 잃고, og.sim.save() 도 거부한다."""
    from omnigibson.macros import gm

    sim = og.sim
    sim.pi.update_simulation(elapsedStep=0, currentTime=sim.current_time)
    for obj in objs:
        obj.initialize()
        if obj in sim._objects_to_initialize:
            sim._objects_to_initialize.remove(obj)
    sim.update_handles()
    if gm.ENABLE_TRANSITION_RULES:
        scene.transition_rule_api.refresh_all_rules()


def add_rigid(og, scene, specs):
    """specs: [{name, usd, pos(월드), quat(xyzw), lin_vel, ang_vel}] → USDObject 들(한 번에 더함)."""
    import torch as th
    from omnigibson.objects import USDObject

    objs = [USDObject(name=s["name"], usd_path=s["usd"], category=s.get("category", "food"), fixed_base=False) for s in specs]
    og.sim.batch_add_objects(objs, [scene] * len(objs))
    initialize_now(og, scene, objs)
    for o, s in zip(objs, specs):
        o.set_position_orientation(position=th.tensor(np.asarray(s["pos"], float), dtype=th.float32),
                                   orientation=th.tensor(np.asarray(s.get("quat", [0, 0, 0, 1]), float), dtype=th.float32))
        o.set_linear_velocity(th.tensor(np.asarray(s.get("lin_vel", [0, 0, 0]), float), dtype=th.float32))
        o.set_angular_velocity(th.tensor(np.asarray(s.get("ang_vel", [0, 0, 0]), float), dtype=th.float32))
    return objs


def split_components(p, h):
    """점들을 한 변 h 격자에 모아 26-이웃으로 이어진 덩어리 번호(n,) — 큰 덩어리부터 0, 1, 2 …(11-25 reports/72)."""
    from scipy import ndimage

    q = np.floor((p - p.min(axis=0)) / h).astype(np.int64)
    occ = np.zeros(tuple(q.max(axis=0) + 1), bool)
    occ[q[:, 0], q[:, 1], q[:, 2]] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3, 3), bool))
    cl = lab[q[:, 0], q[:, 1], q[:, 2]] - 1
    order = np.argsort(-np.bincount(cl, minlength=n), kind="stable")
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    return rank[cl]


def pieces_to_rigid(og, scene, sim, world_from_sim, out_dir, *, albedo=None, min_n=200, prefix="piece", category="food", spacing=None):
    """썬 MPM 식재료 → 조각 묶음마다 PhysX 강체. 돌려줌: (물체 목록, 기록 목록, 뺀 작은 묶음, 입자별 강체 묶음 번호(−1 = 강체 안 됨)).

    11-25(reports/72): spacing(입자 간격)을 주면 솔버 묶음마다 '공간에서 이어진 덩어리'(1.5 × spacing 격자, 26-이웃)로 다시 나눠 덩어리마다 강체로 —
    칼에 붙어 들린 편이 도마 위 몸통과 같은 솔버 번호로 묶이면 볼록 껍질 하나가 둘을 잇는 키 큰 덩어리가 됐다(2판 오이 seed 1 몸통 높이 73 mm,
    칼에 붙은 입자 36개가 섞인 편 높이 116 mm). min_n 보다 작은 덩어리는 강체로 안 만듦(small). 이름: 가장 큰 덩어리 = <prefix>_<묶음>, 나머지 <prefix>_<묶음>_<k>."""
    x_s = sim.positions()
    v_s = sim.v.numpy()
    R, t = world_from_sim[:3, :3], world_from_sim[:3, 3]
    xw = x_s @ R.T + t
    vw = v_s @ R.T
    groups = sim.piece_groups()[sim.pieces()]
    comp_all = np.full(len(xw), -1, np.int64)
    parts = []  # (솔버 묶음, 덩어리 번호, 입자 번호들)
    for gid in np.unique(groups):
        idx = np.flatnonzero(groups == gid)
        if spacing and len(idx) >= min_n:
            cl = split_components(xw[idx], 1.5 * float(spacing))
            parts += [(int(gid), k, idx[cl == k]) for k in range(int(cl.max()) + 1)]
        else:
            parts.append((int(gid), 0, idx))
    specs, recs, small = [], [], []
    for gid, k, idx in parts:
        n = len(idx)
        if n < min_n:
            small.append({"group": gid, "component": k, "n": int(n)})
            continue
        uid = len(recs)
        comp_all[idx] = uid
        p = xw[idx]
        c = p.mean(axis=0)
        local = p - c
        # 11-26 reports/80: 입자가 한 점·한 판에 몰린 덩어리(칼에 붙어 찢긴 조각에서 겹친 입자 수백 개)는 볼록 껍질이 안 생김 → 강체로 안 만들고 small 로
        if len(np.unique(np.round(local, 7), axis=0)) < 4 or np.ptp(local, axis=0).min() < 1e-4:
            small.append({"group": gid, "component": k, "n": int(n), "degenerate": True})
            continue
        try:
            V, F = _hull(local)
        except Exception as e:
            small.append({"group": gid, "component": k, "n": int(n), "degenerate": True, "error": str(e)[:80]})
            continue
        mass, Idiag, axes = _mass_props(local, float(sim.particle_mass))
        # 색 = 겉(볼록 껍질 꼭짓점) 입자 색의 가운데값 — 전체 평균은 속살이 많아 껍질 색이 사라짐(11-24 r4)
        rgb = np.median(albedo[idx][_hull_vertex_ids(local)], axis=0) if albedo is not None else (0.9, 0.9, 0.8)
        name = f"{prefix}_{gid}" if k == 0 else f"{prefix}_{gid}_{k}"
        usd = write_rigid_usd(Path(out_dir) / f"{name}.usd", name, category, V, F, rgb, mass, Idiag, axes)
        specs.append({"name": name, "usd": usd, "category": category, "pos": c, "quat": [0, 0, 0, 1],
                      "lin_vel": vw[idx].mean(axis=0)})
        ext = np.ptp(local, axis=0)
        recs.append({"name": name, "group": uid, "solver_group": gid, "component": k, "n": int(n), "mass_g": 1e3 * mass,
                     "centre_w": c.tolist(), "extent_mm": (1e3 * ext).round(2).tolist(), "hull_vertices": int(len(V)), "usd": usd,
                     "particles": idx.astype(np.int32), "local": local.astype(np.float32)})
    objs = add_rigid(og, scene, specs) if specs else []
    return objs, recs, small, comp_all


def food_rigid_usd(path, name, x_rest_our, m_particle, rgb, category="food"):
    """썰기 전 통 식재료(우리 좌표 입자 틀)를 강체 USD 로. 링크 원점 = 우리 좌표 원점(도마 면 가운데) → 놓을 자세 = 우리 좌표 → 월드."""
    V, F = _hull(np.asarray(x_rest_our, float))
    c = np.asarray(x_rest_our, float).mean(axis=0)
    mass, Idiag, axes = _mass_props(np.asarray(x_rest_our, float) - c, m_particle)
    # 무게중심을 링크 원점 밖에 두려면 CenterOfMass 를 c 로: write_rigid_usd 는 원점 = 무게중심이라 꼭짓점을 c 만큼 옮겨 쓰고, 놓을 때 c 를 더한다
    usd = write_rigid_usd(path, name, category, V - c, F, rgb, mass, Idiag, axes)
    return usd, c, mass


def yaw_of(quat_xyzw):
    x, y, z, w = map(float, quat_xyzw)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ---- 11-25 reports/77: 칼에 붙은 편을 강체 단계에서도 칼에 붙여 두기(끊어지는 접착 이음목) -----------------------------------------
# MPM 단계에서는 물막 접착(film3, warp_fixes.ADHESION_FILM3)으로 편이 칼 옆면에 붙어 칼과 함께 들린다. 강체로 바꾸면 접착이 없어져
# 붙어 있던 편이 저절로 떨어졌다(11-24~25 판). → 붙어 있던 조각을 칼에 고정 이음목으로 붙이되, 이음목이 버티는 힘을 MPM 접착의 정지 버팀
# (미끄럼 방향: τ + μ·p_a, 모두 × wet × 닿은 넓이)으로 두어 그보다 큰 힘(털기·긁기)이 걸리면 끊어지게 한다(PhysX breakForce).

def blade_half_thickness(z, edge=0.0002, spine=0.0015, bevel=0.045):
    """칼날 반두께 [m] — 날 끝(z 0)에서 bevel 높이까지 고르게 두꺼워짐(팀원 KnifeGeometry 와 같은 식)."""
    return 0.5 * (edge + (spine - edge) * np.clip(np.asarray(z, float) / bevel, 0.0, 1.0))


def knife_stuck_pieces(recs, K_world, *, half_len, height, spacing, z_board, wet, tau=30.0, p_a=250.0, mu=0.1,
                       edge=0.0002, spine=0.0015, bevel=0.045, min_n=20, lift=0.002, g=9.81):
    """썰기 끝(강체로 바꾼 때) 칼 자세 K_world(칼 틀: x 칼날 길이, y 두께 = 면 법선, z 날 끝 → 칼등)에서 '칼에 붙은 조각'을 찾는다.

    붙음 = 칼 옆면에서 1.5 입자 간격 + 0.5 mm 안의 입자가 min_n 개 이상이고, 도마에서 lift 넘게 떠 있음.
    붙은 넓이 A = 그 입자들을 칼 면(x, z)에 비춰 입자 간격 칸으로 센 수 × 간격². 버팀 F = (τ + μ·p_a) × wet × A
    (film3 의 정지 버팀 중 가장 약한 미끄럼 방향, μ = 칼질 뒤 칼 마찰). F 가 제 무게의 1.05 배보다 작으면 붙이지 않음(어차피 미끄러져 떨어짐)."""
    Ki = np.linalg.inv(np.asarray(K_world, float))
    out = []
    for k, r in enumerate(recs):
        P = np.asarray(r["centre_w"], float) + np.asarray(r["local"], float)
        L = P @ Ki[:3, :3].T + Ki[:3, 3]
        ht = blade_half_thickness(L[:, 2], edge, spine, bevel)
        near = ((np.abs(np.abs(L[:, 1]) - ht) < 1.5 * spacing + 0.0005) & (np.abs(L[:, 0]) < half_len + 0.002)
                & (L[:, 2] > -0.002) & (L[:, 2] < height + 0.002))
        zmin = float(P[:, 2].min() - z_board)
        if int(near.sum()) < min_n or zmin <= lift:
            continue
        cells = np.unique(np.floor(L[near][:, [0, 2]] / spacing).astype(np.int64), axis=0)
        A = len(cells) * spacing * spacing
        F = (tau + mu * p_a) * wet * A
        W = 1e-3 * float(r["mass_g"]) * g
        side = int(np.sign(np.median(L[near, 1])))
        # 11-25 밤 reports/78: 미끄러지는 접착(glue_slide_to_knife)용 — 칼 면에 닿은 칸 가운데(칼 틀 x·z)와 닿은 자리 가운데(칼 면 위)
        cxz = (cells + 0.5) * spacing
        pc = np.array([float(cxz[:, 0].mean()), 0.0, float(cxz[:, 1].mean())])
        pc[1] = side * float(blade_half_thickness(pc[2], edge, spine, bevel))
        out.append({"k": k, "name": r["name"], "side": side, "area_mm2": round(1e6 * A, 1),
                    "break_N": float(F), "pull_N": float(p_a * wet * A), "weight_N": float(W), "mass_kg": 1e-3 * float(r["mass_g"]), "z_on_blade_mm": [round(1e3 * float(L[near, 2].min()), 1), round(1e3 * float(L[near, 2].max()), 1)],
                    "above_board_mm": round(1e3 * zmin, 1), "glue": bool(F > 1.05 * W),
                    "cells_xz": cxz.tolist(), "patch_K": pc.tolist(), "hold_shear_Pa": float((tau + mu * p_a) * wet), "hold_pull_Pa": float(p_a * wet)})
    return out


def _pose_np(obj):
    p, q = obj.get_position_orientation()
    return np.asarray(p.detach().cpu().numpy() if hasattr(p, "detach") else p, float), np.asarray(q.detach().cpu().numpy() if hasattr(q, "detach") else q, float)


def glue_to_knife(og, knife, objs, stuck):
    """stuck(knife_stuck_pieces) 중 glue 인 조각을 지금 자리 그대로 칼에 고정 이음목으로 붙인다(breakForce = 버팀 F, breakTorque = F × √(A/π)).
    이음목은 두 몸 사이 충돌을 끔(UsdPhysics 기본). 돌려줌: 붙인 목록(조각 번호·이음목 경로·칼 틀에서 본 처음 자세)."""
    import torch as th
    from omnigibson.utils.usd_utils import create_joint
    from scipy.spatial.transform import Rotation

    import os

    scale = float(os.environ.get("GLUE_SCALE", "1"))  # 진단용: 버팀 배율(0 = 안 끊어짐)
    kp, kq = _pose_np(knife.root_link)
    Rk = Rotation.from_quat(kq)
    glued = []
    for s in stuck:
        if not s["glue"]:
            continue
        o = objs[s["k"]]
        op, oq = _pose_np(o.root_link)
        rel_p = Rk.inv().apply(op - kp)
        rel_q = (Rk.inv() * Rotation.from_quat(oq)).as_quat()
        F = float(s["break_N"])
        Tq = F * math.sqrt(s["area_mm2"] * 1e-6 / math.pi)
        path = f"{knife.root_link.prim_path}/glue_{o.name}"
        # 11-25 17:40: PhysX breakForce 는 쥔 칼의 미세 떨림(작은 진폭·높은 진동수 = 큰 순간 가속도)까지 세어 1.7 g 편이 붙이자마자 끊겼다(33 ms).
        #   → PhysX 이음목은 안 끊어지게(진단용 GLUE_PHYSX_BREAK=1 이면 예전처럼) 두고, 끊기 판정은 GlueMonitor(30 ms 평균 가속도로 하중 계산)가 한다.
        phys_break = os.environ.get("GLUE_PHYSX_BREAK") == "1"
        create_joint(prim_path=path, joint_type="FixedJoint", body0=knife.root_link.prim_path, body1=o.root_link.prim_path,
                     joint_frame_in_parent_frame_pos=th.tensor(rel_p, dtype=th.float32),
                     joint_frame_in_parent_frame_quat=th.tensor(rel_q, dtype=th.float32),
                     joint_frame_in_child_frame_pos=th.zeros(3, dtype=th.float32),
                     joint_frame_in_child_frame_quat=th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
                     break_force=(F * scale if phys_break and scale > 0 else None), break_torque=(Tq * scale if phys_break and scale > 0 else None))
        glued.append({"k": s["k"], "name": s["name"], "path": path, "rel_p": rel_p, "rel_q": rel_q, "break_N": F * (scale if scale > 0 else 1e9),
                      "pull_N": float(s["pull_N"]) * (scale if scale > 0 else 1e9), "break_Nm": Tq, "mass_kg": float(s["mass_kg"]), "side": int(s["side"])})
    return glued


class GlueMonitor:
    """붙인 편의 접착 하중을 물리 단계마다 세어, MPM 접착 버팀을 넘으면 그 이음목을 끈다(11-25 reports/77).

    붙어 있는 동안 편은 칼과 한 몸이므로 편 가운데 자리 = 칼 자세 × 칼 틀에서 본 자리. 그 자리의 가속도를 window 단계(기본 4 = 33 ms)
    평균으로 구해(쥔 칼의 미세 떨림은 걸러짐) 접착이 내야 할 힘 f = m·(a − g) 를 칼 틀로 나눈다:
      면을 따라(칼 x·z) |f_면| > break_N(τ + μ·p_a)·wet·A → 미끄러져 떨어짐,  면에서 떼는 쪽 −side·f_y > pull_N(p_a·wet·A) → 떨어짐.
    (MPM film3 의 정지 버팀. 점성 항 c_t·c_n 은 빼고 '짧은 튐은 안 셈'을 평균 창으로 어림.)"""

    def __init__(self, og, knife, glued, dt, window=4, g=9.81):
        self.og, self.knife, self.glued, self.dt, self.window = og, knife, glued, float(dt), int(window)
        self.g = np.array([0.0, 0.0, -g])
        self.hist = []  # 단계마다 (칼 위치, 칼 회전)
        self.broken = {}  # 이름 → (단계, 면 하중 N, 떼는 하중 N)
        self.k = 0
        self.peak = {gl["name"]: [0.0, 0.0] for gl in glued}

    def update(self):
        from scipy.spatial.transform import Rotation

        self.k += 1
        kp, kq = _pose_np(self.knife.root_link)
        Rk = Rotation.from_quat(kq).as_matrix()
        self.hist.append((kp, Rk))
        w = self.window
        if len(self.hist) < w + 2:
            return
        self.hist = self.hist[-(w + 2):]
        for gl in self.glued:
            if gl["name"] in self.broken:
                continue
            x = np.array([p_ + R_ @ gl["rel_p"] for p_, R_ in self.hist])  # 편 가운데의 지난 w+2 자리
            acc = (x[2:] - 2 * x[1:-1] + x[:-2]) / (self.dt * self.dt)
            a_ = acc.mean(axis=0)
            f = gl["mass_kg"] * (a_ - self.g)  # 접착이 편에 줘야 하는 힘(월드)
            fk = Rk.T @ f  # 칼 틀: x 칼 길이, y 면 법선, z 날 끝 → 칼등
            f_face = float(np.hypot(fk[0], fk[2]))
            f_pull = float(-gl["side"] * fk[1])  # +면에 붙은 편을 칼 쪽(−y)으로 당겨야 하는 힘 = 떼는 하중
            pk = self.peak[gl["name"]]
            pk[0], pk[1] = max(pk[0], f_face), max(pk[1], f_pull)
            if f_face > gl["break_N"] or f_pull > gl["pull_N"]:
                prim = self.og.sim.stage.GetPrimAtPath(gl["path"])
                if prim and prim.IsValid():
                    prim.GetAttribute("physics:jointEnabled").Set(False)
                self.broken[gl["name"]] = (self.k, round(f_face, 4), round(f_pull, 4))

    def summary(self):
        return {"broken": {k_: {"step": v[0], "face_N": v[1], "pull_N": v[2]} for k_, v in self.broken.items()},
                "peak_load_N": {k_: [round(v[0], 4), round(v[1], 4)] for k_, v in self.peak.items()}}


def glue_state(knife, objs, glued, tol_mm=2.0, tol_deg=5.0):
    """붙인 조각마다 칼 틀에서 처음 자세와 얼마나 어긋났나 → attached(어긋남이 tol 안). 이음목이 끊기면 조각이 미끄러져 떨어지므로 어긋남으로 판정."""
    from scipy.spatial.transform import Rotation

    kp, kq = _pose_np(knife.root_link)
    Rk = Rotation.from_quat(kq)
    out = []
    for gl in glued:
        op, oq = _pose_np(objs[gl["k"]].root_link)
        dp = 1e3 * float(np.linalg.norm(Rk.inv().apply(op - kp) - gl["rel_p"]))
        dq = float(np.degrees((Rotation.from_quat(gl["rel_q"]).inv() * (Rk.inv() * Rotation.from_quat(oq))).magnitude()))
        out.append({"name": gl["name"], "drift_mm": round(dp, 2), "drift_deg": round(dq, 2), "attached": bool(dp < tol_mm and dq < tol_deg)})
    return out


def unglue(og, glued):
    """남은 접착 이음목을 끈다(칼을 놓기 전에 — 칼을 순간 이동하면 이음목이 조각을 홱 끌어당기므로)."""
    for gl in glued:
        prim = og.sim.stage.GetPrimAtPath(gl["path"])
        if prim and prim.IsValid():
            prim.GetAttribute("physics:jointEnabled").Set(False)
    if og.sim.is_playing():
        og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)


# ---- 11-25 밤 reports/78: 미끄러지는 접착 이음(칼 면을 따라 미끄러짐·손가락·그릇 테·도마 충격까지 같은 식으로) -------------------
# 77 판(glue_to_knife + GlueMonitor)은 '고정 이음목 + 칼 가속도로 센 관성 하중이 버팀을 넘으면 끔' 규칙이었다. 칼 가속도만 세므로
# 손가락·그릇 테가 편을 밀어도 하중으로 안 잡혔고(붙은 편은 칼과 한 몸이라 그대로 밀고 감), 넘는 순간 편이 통째로 '자유'가 됐다.
# 여기서는 편과 칼 사이를 PhysX D6 이음(6자유도 이음)으로 잇고, 칼 면을 따르는 두 방향(칼 길이 x·날 끝→칼등 z)에 용수철·감쇠 드라이브를
# 두되 그 힘 상한 = MPM 물막 접착(film3)의 정지 미끄럼 버팀 (τ + μ·p_a)·wet·A 로 둔다. 관성이든 손가락·테·도마 접촉이든 편을 미는 힘은
# PhysX 가 풀고, 드라이브가 상한에 닿으면 편이 칼 면을 따라 미끄러진다(SlideMonitor 가 닻(목표 자리)을 편 쪽으로 옮겨 되돌아가지 않게 —
# 탄소성 마찰 모형). 미끄러져 칼 면을 벗어난 만큼 닿은 넓이 A 가 줄어 버팀도 줄고, 넓이가 거의 없어지면(칼날 밖) 이음을 끈다.
# 면에서 떼는 방향(y)은 한쪽 한계(칼 속으로는 못 들어감) + 용수철(상한 = 흡인 p_a·wet·A): 당김이 흡인을 넘으면 물막이 끊겼다고 보고 끈다.
# 돌림은 잠금(편이 칼 면에서 벗겨지듯 도는 '박리'는 없음 — 한계).

def face_fraction(cells_xz, sx, sz, *, x0, x1, z0, top_fn):
    """칼 면에 닿은 칸들(칼 틀 x·z)을 (sx, sz) 만큼 옮겼을 때 칼 면(x0..x1, z0..top_fn(x)) 안에 남는 비율."""
    c = np.asarray(cells_xz, float)
    x, z = c[:, 0] + sx, c[:, 1] + sz
    inside = (x >= x0) & (x <= x1) & (z >= z0) & (z <= np.asarray(top_fn(x), float))
    return float(inside.mean()) if len(c) else 0.0


def glue_slide_to_knife(og, knife, objs, stuck, *, f_n=25.0, zeta=0.7, kn_mul=4.0, edge=0.0002, spine=0.0015, bevel=0.045):
    """stuck(knife_stuck_pieces) 중 glue 인 조각을 지금 자리 그대로 칼에 '미끄러지는 접착 이음'(D6)으로 붙인다.

    이음 틀: 칼 링크 틀 안 닿은 자리 가운데(patch_K)에, 축 = 칼 틀(x 칼 길이, y 면 법선, z 날 끝→칼등)을 칼 쐐기 기울기만큼 x 축으로 돌림
    (z 축이 칼 면을 따라감). 미끄럼 두 축(transX·transZ): 용수철 k_s = m(2π f_n)²·감쇠 2ζ·m·2πf_n, 힘 상한 = 미끄럼 버팀.
    면 법선(transY): 칼 쪽으로는 한계(못 들어감), 떼는 쪽은 용수철 k_n = kn_mul·k_s(상한 = 흡인). 돌림 세 축 잠금.
    f_n(기본 25 Hz)은 편–칼 이음의 고유 진동수 — 쥔 칼의 미세 떨림(물리 단계 120 Hz 수준)은 걸러지고 털기·두드리기(수 Hz)는 그대로 전해진다.
    → 붙인 목록(SlideMonitor 가 씀)."""
    from pxr import Gf, Sdf, UsdPhysics
    from scipy.spatial.transform import Rotation

    stage = og.sim.stage
    kp, kq = _pose_np(knife.root_link)
    Rk = Rotation.from_quat(kq)
    glued = []
    for s in stuck:
        if not s["glue"]:
            continue
        o = objs[s["k"]]
        op, oq = _pose_np(o.root_link)
        Ro = Rotation.from_quat(oq)
        side = int(s["side"])
        cK = np.asarray(s["patch_K"], float)
        slope = 0.5 * (spine - edge) / bevel if cK[2] < bevel else 0.0  # 칼 면의 기울기 dy/dz(쐐기)
        R0 = Rotation.from_euler("x", -side * math.atan(slope))  # 이음 z 축 = (0, side·slope, 1) 방향 = 칼 면을 따라
        jw = kp + Rk.apply(cK)
        R1 = Ro.inv() * Rk * R0
        l1 = Ro.inv().apply(jw - op)
        path = f"{knife.root_link.prim_path}/slide_{o.name}"
        j = UsdPhysics.Joint.Define(stage, path)
        j.CreateBody0Rel().SetTargets([Sdf.Path(knife.root_link.prim_path)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(o.root_link.prim_path)])
        q0, q1 = R0.as_quat(), R1.as_quat()
        j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, cK)))
        j.CreateLocalRot0Attr(Gf.Quatf(float(q0[3]), float(q0[0]), float(q0[1]), float(q0[2])))
        j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, l1)))
        j.CreateLocalRot1Attr(Gf.Quatf(float(q1[3]), float(q1[0]), float(q1[1]), float(q1[2])))
        j.CreateCollisionEnabledAttr(False)
        j.CreateJointEnabledAttr(True)
        prim = j.GetPrim()
        for ax in ("rotX", "rotY", "rotZ"):  # 잠금(low > high)
            L_ = UsdPhysics.LimitAPI.Apply(prim, ax)
            L_.CreateLowAttr(1.0)
            L_.CreateHighAttr(-1.0)
        L_ = UsdPhysics.LimitAPI.Apply(prim, "transY")  # 칼 속으로는 못 들어감, 떼는 쪽은 열림
        L_.CreateLowAttr(0.0 if side > 0 else -0.05)
        L_.CreateHighAttr(0.05 if side > 0 else 0.0)
        m = float(s["mass_kg"])
        w = 2.0 * math.pi * f_n
        k_s, c_s = m * w * w, 2.0 * zeta * m * w
        k_n, c_n = kn_mul * k_s, 2.0 * zeta * m * w * math.sqrt(kn_mul)
        A0 = 1e-6 * float(s["area_mm2"])
        F_s, F_n = float(s["hold_shear_Pa"]) * A0, float(s["hold_pull_Pa"]) * A0
        drives = {}
        for ax, kk, cc, ff in (("transX", k_s, c_s, F_s), ("transZ", k_s, c_s, F_s), ("transY", k_n, c_n, F_n)):
            D = UsdPhysics.DriveAPI.Apply(prim, ax)
            D.CreateTypeAttr("force")
            D.CreateStiffnessAttr(float(kk))
            D.CreateDampingAttr(float(cc))
            D.CreateMaxForceAttr(float(ff))
            D.CreateTargetPositionAttr(0.0)
            D.CreateTargetVelocityAttr(0.0)
            drives[ax] = D
        glued.append({"k": s["k"], "name": s["name"], "path": path, "side": side, "mass_kg": m, "cK": cK, "R0": R0.as_matrix(),
                      "l1": l1, "k_s": k_s, "k_n": k_n, "A0": A0, "hold_s": float(s["hold_shear_Pa"]), "hold_n": float(s["hold_pull_Pa"]),
                      "cells_xz": np.asarray(s["cells_xz"], float), "drives": drives, "t": np.zeros(2), "F_set": (F_s, F_n),
                      "break_N": F_s, "pull_N": F_n, "rel_p": Rk.inv().apply(op - kp), "rel_q": (Rk.inv() * Ro).as_quat()})
    if glued and og.sim.is_playing():
        og.sim.pi.update_simulation(elapsedStep=0, currentTime=og.sim.current_time)
    return glued


class SlideMonitor:
    """미끄러지는 접착 이음 감시(물리 단계마다, 11-25 밤 reports/78).

    편의 이음 자리가 칼의 이음 자리에서 얼마나 옮겨졌나(이음 틀 x·y·z = 미끄럼 sx, 떼어짐 sy, 미끄럼 sz)를 재서
    - 미끄럼: 용수철 늘어남 |t − s|·k_s 가 버팀 F_s(A) 를 넘으면 닻 t 를 편 쪽으로 옮김(늘어남 = F_s/k_s 로) → 되돌아가지 않는 미끄러짐.
    - 닿은 넓이: 붙을 때 닿은 칸들을 미끄럼만큼 옮겨 칼 면 안에 남는 비율 × A0 → 버팀(미끄럼·흡인 상한)을 넓이에 맞춰 고침.
    - 끄기: 남은 넓이 < frac_release·A0(칼날 밖으로 미끄러져 나감) 또는 떼는 쪽 늘어남·k_n > 흡인(물막 끊김).
    broken(끈 것: 단계·까닭·미끄럼 mm)·path(편 미끄럼 자취)·peak 를 남김. GlueMonitor 와 같은 broken 이름을 써서 털기 과제 코드가 그대로 쓴다."""

    def __init__(self, og, knife, objs, glued, dt, *, face, frac_release=0.15, gap_extra=0.0003, pull_steps=4):
        self.og, self.knife, self.objs, self.glued, self.dt = og, knife, objs, glued, float(dt)
        self.face = face  # dict(x0, x1, z0, top_fn)
        self.frac_release, self.gap_extra = float(frac_release), float(gap_extra)
        # 떼어짐(물막 끊김)은 pull_steps 단계(기본 4 = 33 ms) 내리 이어질 때만 — 11-26 새벽 양파 seed 1: 강체로 바꾸는 순간 겹친 이웃 편을 PhysX 가
        #   밀어내며 붙인 편이 1.5 mm 떠 3 단계 만에 풀림(같은 상태 다른 판에선 안 풀림). 77 의 33 ms 평균 창과 같은 취지.
        self.pull_steps = int(pull_steps)
        self.pull_run = {gl["name"]: 0 for gl in glued}
        self.broken = {}
        self.k = 0
        self.phase = ""
        self.trace = {gl["name"]: [] for gl in glued}  # (단계, sx mm, sz mm, sy mm, 넓이 비율, 늘어남 힘 N)
        self.peak = {gl["name"]: [0.0, 0.0] for gl in glued}  # 미끄럼 방향 힘·떼는 힘 최대 [N]
        self.slip_events = {gl["name"]: 0 for gl in glued}

    def _disable(self, gl):
        prim = self.og.sim.stage.GetPrimAtPath(gl["path"])
        if prim and prim.IsValid():
            prim.GetAttribute("physics:jointEnabled").Set(False)

    def update(self):
        from scipy.spatial.transform import Rotation

        self.k += 1
        if self.k % 4 == 0:
            if not hasattr(self, "phase_log"):
                self.phase_log = []
            self.phase_log.append((self.k, self.phase))
        if len(self.broken) == len(self.glued):
            return
        kp, kq = _pose_np(self.knife.root_link)
        RkT = Rotation.from_quat(kq).as_matrix().T
        for gl in self.glued:
            if gl["name"] in self.broken:
                continue
            op, oq = _pose_np(self.objs[gl["k"]].root_link)
            jw1 = op + Rotation.from_quat(oq).as_matrix() @ gl["l1"]
            d = gl["R0"].T @ (RkT @ (jw1 - kp) - gl["cK"])  # 이음 틀에서 본 편 자리 옮김
            sx, sy, sz = map(float, d)
            frac = face_fraction(gl["cells_xz"], sx, sz, **self.face)
            A = frac * gl["A0"]
            F_s, F_n = gl["hold_s"] * A, gl["hold_n"] * A
            e = gl["t"] - np.array([sx, sz])
            f_sh = gl["k_s"] * float(np.linalg.norm(e))
            f_pull = gl["k_n"] * max(0.0, gl["side"] * sy)
            pk = self.peak[gl["name"]]
            pk[0], pk[1] = max(pk[0], min(f_sh, F_s)), max(pk[1], f_pull)
            why = None
            over = gl["side"] * sy > F_n / gl["k_n"] + self.gap_extra
            self.pull_run[gl["name"]] = self.pull_run[gl["name"]] + 1 if over else 0
            if frac < self.frac_release:
                why = "slid_off"
            elif self.pull_run[gl["name"]] >= self.pull_steps:
                why = "pulled_off"
            if why:
                self._disable(gl)
                self.broken[gl["name"]] = (self.k, why, round(1e3 * sx, 2), round(1e3 * sz, 2), self.phase)
                self.trace[gl["name"]].append((self.k, round(1e3 * sx, 2), round(1e3 * sz, 2), round(1e3 * sy, 3), round(frac, 3), round(f_sh, 4)))
                continue
            if f_sh > F_s and F_s > 0:  # 미끄러짐: 닻을 편 쪽으로(늘어남 = 버팀/k_s)
                t_new = np.array([sx, sz]) + e * (F_s / f_sh)
                gl["t"] = t_new
                gl["drives"]["transX"].GetTargetPositionAttr().Set(float(t_new[0]))
                gl["drives"]["transZ"].GetTargetPositionAttr().Set(float(t_new[1]))
                self.slip_events[gl["name"]] += 1
            Fs0, Fn0 = gl["F_set"]
            if abs(F_s - Fs0) > 0.03 * max(Fs0, 1e-9):  # 넓이가 바뀐 만큼 상한 고침
                gl["drives"]["transX"].GetMaxForceAttr().Set(float(F_s))
                gl["drives"]["transZ"].GetMaxForceAttr().Set(float(F_s))
                gl["drives"]["transY"].GetMaxForceAttr().Set(float(F_n))
                gl["F_set"] = (F_s, F_n)
            if self.k % 4 == 0:
                self.trace[gl["name"]].append((self.k, round(1e3 * sx, 2), round(1e3 * sz, 2), round(1e3 * sy, 3), round(frac, 3), round(f_sh, 4)))

    def attached(self):
        return [gl["name"] for gl in self.glued if gl["name"] not in self.broken]

    def summary(self):
        return {"broken": {k_: {"step": v[0], "why": v[1], "slip_x_mm": v[2], "slip_z_mm": v[3], "phase": v[4]} for k_, v in self.broken.items()},
                "peak_load_N": {k_: [round(v[0], 4), round(v[1], 4)] for k_, v in self.peak.items()},
                "slip_steps": dict(self.slip_events),
                "last": {k_: (v[-1] if v else None) for k_, v in self.trace.items()}}


def unglue_slide(og, glued):
    """남은 미끄러지는 접착 이음을 끈다(칼을 놓기 전에)."""
    unglue(og, glued)


def _box_mesh(lo, hi):
    """축 맞춘 상자 꼭짓점·삼각형(바깥 법선)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return _hull(V)


def write_tray_usd(path, name, size_x, size_y, wall_h, wall_t=0.006, floor_t=0.006, rgb=(0.95, 0.95, 0.93), roughness=0.35, end_h=None, gap_py=0.0):
    """곧은 벽이 있는 네모 접시(11-25 밤 reports/78 '접시에 문질러 떼기'): 바닥 + 벽 4 개 = 볼록 상자 충돌체 5 개, 한 링크.
    틀 원점 = 바닥 아랫면 가운데(탁자 면에 놓음). 긴 쪽(±x) 벽 윗면 = wall_h(탁자 위), 짧은 쪽(±y) = end_h(없으면 wall_h).
    둥근 테에 곧은 칼날을 대면 두 점에서만 닿아(현) 가운데 편을 못 긁으므로 곧은 벽. 칼을 기울여 긁으면 날 전체·손잡이가 접시 안으로
    내려가므로(11-25 밤 s1: 18 cm 접시 짧은 벽에 칼·손잡이가 걸려 36 mm 앞에서 멈춤) 접시를 칼날보다 길게, 짧은 쪽은 낮은 턱으로.
    gap_py: 긴 벽(±x)의 +y 끝 이만큼 [m]은 낮은 턱(end_h) — 칼 뒤꿈치 너머 손잡이·로봇 손이 벽에 안 걸리게(s3~s7)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from simfoundry.cutting.og_cosim import _identity_ops, _mesh, _omni_pbr, srgb_to_linear

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    _identity_ops(root)
    root.GetPrim().CreateAttribute("ig:category", Sdf.ValueTypeNames.String).Set("plate")
    UsdGeom.Scope.Define(stage, f"/{name}/Looks")
    look = _omni_pbr(stage, f"/{name}/Looks/ceramic", srgb_to_linear(np.clip(np.asarray(rgb, float), 0, 1)), roughness=roughness)
    link = UsdGeom.Xform.Define(stage, f"/{name}/base_link")
    _identity_ops(link)
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr(1.0)
    UsdGeom.Xform.Define(stage, f"/{name}/base_link/visuals")
    UsdGeom.Xform.Define(stage, f"/{name}/base_link/collisions")
    hx, hy = 0.5 * size_x, 0.5 * size_y
    eh = end_h or wall_h
    yb = hy - gap_py
    boxes = {"floor": ([-hx, -hy, 0.0], [hx, hy, floor_t]),
             "wall_px": ([hx - wall_t, -hy, 0.0], [hx, yb, wall_h]), "wall_nx": ([-hx, -hy, 0.0], [-hx + wall_t, yb, wall_h]),
             "wall_py": ([-hx, hy - wall_t, 0.0], [hx, hy, eh]), "wall_ny": ([-hx, -hy, 0.0], [hx, -hy + wall_t, eh])}
    if gap_py > 0:
        boxes["lip_px"] = ([hx - wall_t, yb, 0.0], [hx, hy, eh])
        boxes["lip_nx"] = ([-hx, yb, 0.0], [-hx + wall_t, hy, eh])
    for nm, (lo, hi) in boxes.items():
        V, F = _box_mesh(lo, hi)
        Vu = V[F.reshape(-1)]
        vis = _mesh(stage, f"/{name}/base_link/visuals/{nm}", Vu, np.arange(len(Vu)).reshape(-1, 3))
        UsdShade.MaterialBindingAPI.Apply(vis.GetPrim()).Bind(look)
        col = _mesh(stage, f"/{name}/base_link/collisions/{nm}", V, F, purpose=UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr("convexHull")
    root.GetPrim().CreateAttribute("ig:nativeBB", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(float(size_x), float(size_y), float(wall_h)))
    stage.GetRootLayer().Save()
    return str(path.resolve())


def write_bowl_usd(path, name, radius, wall_h, n_seg=28, wall_t=0.005, floor_t=0.006, rgb=(0.94, 0.94, 0.92), roughness=0.3):
    """둥근 그릇(11-26 reports/79 '썬 조각을 그릇에 옮기기'): 바닥 원판(볼록 다각 기둥) + 벽 n_seg 조각(돌린 상자, 각각 볼록 충돌체), 한 링크.
    틀 원점 = 바닥 아랫면 가운데(탁자 면에 놓음). radius = 벽 바깥 반지름, 벽 윗면 = wall_h(탁자 위)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from simfoundry.cutting.og_cosim import _identity_ops, _mesh, _omni_pbr, srgb_to_linear

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    _identity_ops(root)
    root.GetPrim().CreateAttribute("ig:category", Sdf.ValueTypeNames.String).Set("bowl")
    UsdGeom.Scope.Define(stage, f"/{name}/Looks")
    look = _omni_pbr(stage, f"/{name}/Looks/ceramic", srgb_to_linear(np.clip(np.asarray(rgb, float), 0, 1)), roughness=roughness)
    link = UsdGeom.Xform.Define(stage, f"/{name}/base_link")
    _identity_ops(link)
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr(1.0)
    UsdGeom.Xform.Define(stage, f"/{name}/base_link/visuals")
    UsdGeom.Xform.Define(stage, f"/{name}/base_link/collisions")
    parts = {}
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    disc = np.array([[radius * np.cos(t), radius * np.sin(t), z] for t in ang for z in (0.0, floor_t)])
    parts["floor"] = _hull(disc)
    half = np.pi / n_seg
    for k in range(n_seg):  # 벽 한 조각 = 안·바깥 반지름 사이, 각도 ±half(조금 겹치게) 의 볼록 사다리꼴 기둥
        t0, t1 = 2 * np.pi * k / n_seg - 1.15 * half, 2 * np.pi * k / n_seg + 1.15 * half
        P = np.array([[r_ * np.cos(t_), r_ * np.sin(t_), z] for t_ in (t0, t1) for r_ in (radius - wall_t, radius) for z in (0.0, wall_h)])
        parts[f"wall{k:02d}"] = _hull(P)
    for nm, (V, F) in parts.items():
        Vu = V[F.reshape(-1)]
        vis = _mesh(stage, f"/{name}/base_link/visuals/{nm}", Vu, np.arange(len(Vu)).reshape(-1, 3))
        UsdShade.MaterialBindingAPI.Apply(vis.GetPrim()).Bind(look)
        col = _mesh(stage, f"/{name}/base_link/collisions/{nm}", V, F, purpose=UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr("convexHull")
    root.GetPrim().CreateAttribute("ig:nativeBB", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(2 * radius, 2 * radius, float(wall_h)))
    stage.GetRootLayer().Save()
    return str(path.resolve())
