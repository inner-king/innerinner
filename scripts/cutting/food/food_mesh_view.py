"""로봇 에피소드 안에서 식재료를 메쉬로 그린다 — 팀원 FoodPoints(입자 공) 대신 (2026-11-25, 계획 B5, reports/69).

- 메쉬는 에피소드 전에 food_mesh_prep.py 가 만든다(계획된 칼질 평면으로 미리 나눈 조각: 껍질 = 꼭짓점 색, 단면 = 과육·씨 질감).
- 두 가지로 따라 움직인다.
  ① 입자 따라가기(썰기 중): 조각마다 자기 입자에 강체 맞춤(Kabsch) + 남는 변형 일부(residual)를 꼭짓점에 옮김.
     og_mesh_render.py(끝난 실행을 그리는 것)의 deform·torn_off·fold_torn 과 같은 식(그 파일은 실행 스크립트라 import 못 해 옮겨 씀).
  ② 강체 따라가기(놓기·조각 옮기기): 넘겨준 PhysX 강체 자세로 메쉬를 통째로 옮김(그 강체의 볼록 껍질 그림은 숨김).
- Isaac Sim(OmniGibson) 프로세스 안에서 쓴다(pxr). 계산은 numpy/scipy, 그리는 것은 USD Mesh 점·법선 갱신.
"""
import json
import time
from pathlib import Path

import numpy as np


def kabsch(Pa, Pb):
    ca, cb = Pa.mean(0), Pb.mean(0)
    U, _, Vt_ = np.linalg.svd((Pa - ca).T @ (Pb - cb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt_.T @ U.T))])
    R = Vt_.T @ D @ U.T
    return R, cb - ca @ R.T


def torn_off(Xr, P, r_link=0.003, tear=0.005):
    """og_mesh_render.torn_off 와 같음: 조각 입자가 찢겨 멀리 갈라졌으면 가장 큰 덩어리와 따로 움직인 덩어리를 표시(없으면 None)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pr = cKDTree(P).query_pairs(r_link, output_type="ndarray")
    cl = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(P), len(P))), directed=False)[1]
    sz = np.bincount(cl)
    if len(sz) < 2:
        return None
    c0 = int(np.argmax(sz))
    R, t = kabsch(Xr[cl == c0], P[cl == c0])
    off = np.zeros(len(P), bool)
    for c in range(len(sz)):
        m = cl == c
        if c != c0 and np.linalg.norm((P[m] - (Xr[m] @ R.T + t)).mean(0)) > tear:
            off |= m
    return off if off.any() else None


def fold_torn(V, nb, w, keep, P):
    """og_mesh_render.fold_torn 과 같음: 찢겨 나간 덩어리 쪽 꼭짓점을 남은 입자 중 가까운 것 자리로 접어 넣음(넓이 0 → 안 보임)."""
    wk = (w * keep[nb]).sum(1) / np.maximum(w.sum(1), 1e-12)
    torn = wk < 0.5
    if not torn.any():
        return V
    j = np.where(keep[nb[torn]], np.arange(nb.shape[1])[None, :], nb.shape[1]).min(1)
    has = j < nb.shape[1]
    tgt = np.empty((int(torn.sum()), 3))
    tgt[has] = P[nb[torn][has, j[has]]]
    if (~has).any():
        from scipy.spatial import cKDTree

        kp = np.flatnonzero(keep)
        tgt[~has] = P[kp[cKDTree(P[kp]).query(V[torn][~has])[1]]]
    V = V.copy()
    V[torn] = tgt
    return V


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for c in range(3):
        np.add.at(N, F[:, c], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def _pose(obj):
    """OmniGibson 물체 자세 → (R 3×3, t 3)."""
    from scipy.spatial.transform import Rotation

    p, q = obj.get_position_orientation()
    p = p.cpu().numpy() if hasattr(p, "cpu") else np.asarray(p)
    q = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
    return Rotation.from_quat(np.asarray(q, float)).as_matrix(), np.asarray(p, float)


class FoodMesh:
    """식재료 조각 메쉬(USD). update 는 refresh() — 실행기가 화면을 찍기 직전에 부른다."""

    def __init__(self, stage, path, mesh_dir, *, residual=0.6, r_max=0.004, skin_rough=0.38, flesh_rough=0.30, flesh_spec=0.5):
        from pxr import UsdGeom

        self.stage, self.path, self.residual, self.r_max = stage, path, float(residual), float(r_max)
        self.root = UsdGeom.Xform.Define(stage, path)
        mesh_dir = Path(mesh_dir)
        info = json.loads((mesh_dir / "mesh.json").read_text())
        A = np.load(mesh_dir / "mesh.npz")
        self.info = info
        self.pieces = []
        mats = {}
        for i, parts in enumerate(info["pieces"]):
            idx = A[f"p{i}_idx"].astype(np.int64)
            subs, V0s = [], []
            for pt in parts:
                nm = pt["name"]
                V0 = A[f"{nm}_V"].astype(float)
                F = A[f"{nm}_F"].astype(np.int32)
                V0s.append(V0)
                if len(F) == 0:  # 삼각형 0개인 빈 단면(og_mesh_render 와 같은 까닭)
                    continue
                if pt["kind"] == "skin_vc":
                    if "skin_vc" not in mats:
                        mats["skin_vc"] = self._vc_material(f"{path}/Looks/skin_vc", skin_rough)
                    prim = self._make_mesh(f"{path}/{nm}", V0, F, None, mats["skin_vc"], colors=A[f"{nm}_C"].astype(np.float32))
                else:
                    key = pt["tex"]
                    if key not in mats:
                        mats[key] = self._omni_pbr_tex(f"{path}/Looks/{Path(key).stem}", mesh_dir / "tex" / key, flesh_rough, flesh_spec)
                    prim = self._make_mesh(f"{path}/{nm}", V0, F, A[f"{nm}_uv"].astype(np.float32), mats[key])
                subs.append((prim, F, int(pt["v0"]), int(pt["n"])))
            self.pieces.append(dict(idx=idx, Xr=A[f"p{i}_Xr"].astype(float), nb=A[f"p{i}_nb"].astype(np.int64), w=A[f"p{i}_w"].astype(float),
                                    V0=np.concatenate(V0s), subs=subs, V=None, body=None, local=None, hidden=False))
        self.source = None  # 입자 따라가기: () → 월드 입자 자리 (N, 3)
        self.rigid = False
        self.n_refresh, self.t_refresh = 0, 0.0

    # --- USD ---
    def _vc_material(self, path, rough):
        from pxr import Sdf, UsdShade

        mat = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, f"{path}/surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        rd = UsdShade.Shader.Define(self.stage, f"{path}/color")
        rd.CreateIdAttr("UsdPrimvarReader_float3")
        rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
        res = rd.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(res)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
        mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        return mat

    def _omni_pbr_tex(self, path, tex, rough, spec):
        from pxr import Sdf, UsdShade

        mat = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, f"{path}/Shader")
        sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(tex).resolve())))
        sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(rough))
        sh.CreateInput("specular_level", Sdf.ValueTypeNames.Float).Set(float(spec))
        o = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
        for so in (mat.CreateSurfaceOutput("mdl"), mat.CreateDisplacementOutput("mdl"), mat.CreateVolumeOutput("mdl")):
            so.ConnectToSource(o)
        return mat

    def _make_mesh(self, path, V, F, uv, mat, colors=None):
        from pxr import Sdf, UsdGeom, UsdShade, Vt

        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(V, np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(F, np.int32).reshape(-1)))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        m.CreateDoubleSidedAttr(False)
        if uv is not None:
            st = UsdGeom.PrimvarsAPI(m.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
            st.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(uv, np.float32)))
        if colors is not None:  # sRGB → 선형
            lin = np.where(colors <= 0.04045, colors / 12.92, ((colors + 0.055) / 1.055) ** 2.4).astype(np.float32)
            pv = UsdGeom.PrimvarsAPI(m.GetPrim()).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
            pv.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(lin)))
        m.CreateNormalsAttr()
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        m.CreateExtentAttr()
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
        return m

    def _set(self, pc, V):
        from pxr import Gf, Vt

        pc["V"] = V
        for prim, F, v0, n in pc["subs"]:
            Vp = V[v0:v0 + n].astype(np.float32)
            prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(Vp)))
            prim.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(vertex_normals(Vp, F).astype(np.float32)))
            used = Vp[np.unique(F)]
            prim.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*used.min(0).tolist()), Gf.Vec3f(*used.max(0).tolist())]))

    def _hide_piece(self, pc):
        from pxr import UsdGeom

        for prim, *_ in pc["subs"]:
            UsdGeom.Imageable(prim.GetPrim()).MakeInvisible()
        pc["hidden"] = True

    # --- 움직이기 ---
    def deform(self, pc, P_all, lab_all=None):
        P = P_all[pc["idx"]]
        keep = None
        if lab_all is not None:  # 11-26 reports/81: 맞춤 입자가 물리 조각 둘에 걸치면(마지막 편 구역에 몸통 입자 11~44 %) 많은 쪽만으로 맞춤
            lb = np.asarray(lab_all)[pc["idx"]]
            u_, c_ = np.unique(lb, return_counts=True)
            if len(u_) > 1 and c_.max() >= 8:
                keep = lb == u_[c_.argmax()]
        R, t = kabsch(pc["Xr"], P) if keep is None else kabsch(pc["Xr"][keep], P[keep])
        if len(P) >= 8 and np.mean(np.linalg.norm(P - (pc["Xr"] @ R.T + t), axis=1) >= self.r_max) > 0.1:
            off = torn_off(pc["Xr"], P)
            if off is not None:
                keep = ~off if keep is None else (keep & ~off)
                if keep.sum() >= 4:
                    R, t = kabsch(pc["Xr"][keep], P[keep])
        V = pc["V0"] @ R.T + t
        if self.residual > 0:
            res = P - (pc["Xr"] @ R.T + t)
            ok = np.linalg.norm(res, axis=1) < self.r_max
            if keep is not None:
                ok &= keep
            ww = pc["w"] * ok[pc["nb"]]
            V = V + self.residual * (ww[..., None] * res[pc["nb"]]).sum(1) / np.maximum(ww.sum(1, keepdims=True), 1e-9)
        if keep is not None:
            V = fold_torn(V, pc["nb"], pc["w"], keep, P)
        return V

    def follow_particles(self, source, labels=None):
        """source: () → 월드 입자 자리(솔버 입자 순서). 다음 refresh 부터 입자를 따라감.
        labels: () → 입자별 물리 조각 번호(11-26 reports/81, 주면 맞춤 입자 가운데 많은 조각 쪽만으로 맞춤)."""
        self.source, self.labels, self.rigid = source, labels, False

    def bind_rigid(self, objs, groups_of_particles, group_of_obj):
        """썰기 뒤 넘겨주기: 메쉬 조각마다 입자 대부분이 든 솔버 묶음 → 그 강체에 지금 자리 그대로 붙임. 강체가 안 된 묶음(작은 묶음)의 조각은 숨김.
        groups_of_particles: (N,) 입자별 솔버 묶음 번호, group_of_obj: [묶음 번호(objs 순서)]. 돌려줌: 조각별 붙은 물체 번호(-1 = 숨김)."""
        if self.source is not None and not self.rigid:
            self.refresh()  # 지금 입자 자리로(마지막 화면 뒤에도 물리가 몇 단계 더 갔을 수 있음)
        where = {int(gv): k for k, gv in enumerate(group_of_obj)}
        poses = [_pose(o) for o in objs]
        bound = []
        for pc in self.pieces:
            g_ = np.asarray(groups_of_particles)[pc["idx"]]
            if len(g_):  # 가장 많은 묶음(−1 = 강체 안 된 입자, 11-25: 음수도 셀 수 있게 unique)
                u_, c_ = np.unique(g_, return_counts=True)
                gid = int(u_[c_.argmax()])
            else:
                gid = -1
            k = where.get(gid, -1)
            bound.append(k)
            if k < 0:
                self._hide_piece(pc)
                continue
            R, t = poses[k]
            pc["body"], pc["local"] = objs[k], (pc["V"] - t) @ R  # Rᵀ(v − t)
        self.rigid, self.source = True, None
        return bound

    def bind_rigid_local(self, obj, V_local_of_piece):
        """놓기 과제: 통 식재료 강체 하나에 조각(보통 하나)을 강체 틀 좌표 꼭짓점으로 붙임. V_local_of_piece: 조각 번호 → (n_v, 3)."""
        for i, pc in enumerate(self.pieces):
            if i in V_local_of_piece:
                pc["body"], pc["local"] = obj, np.asarray(V_local_of_piece[i], float)
            else:
                self._hide_piece(pc)
        self.rigid, self.source = True, None

    def refresh(self):
        t0 = time.perf_counter()
        if self.rigid:
            for pc in self.pieces:
                if pc["body"] is None or pc["hidden"]:
                    continue
                R, t = _pose(pc["body"])
                self._set(pc, pc["local"] @ R.T + t)
        elif self.source is not None:
            P = np.asarray(self.source(), float)
            lab = self.labels() if getattr(self, "labels", None) is not None else None
            for pc in self.pieces:
                self._set(pc, self.deform(pc, P, lab))
        self.n_refresh += 1
        self.t_refresh += time.perf_counter() - t0

    def hide(self):
        self.root.GetPrim().SetActive(False)
        self.source, self.rigid = None, False

    def stats(self):
        return {"refreshes": self.n_refresh, "refresh_ms_mean": round(1e3 * self.t_refresh / max(self.n_refresh, 1), 2),
                "pieces": len(self.pieces), "vertices": int(sum(len(pc["V0"]) for pc in self.pieces))}


def hide_rigid_visuals(stage, objs):
    """넘겨준 강체(food_handoff 볼록 껍질)의 그림을 숨김 — 메쉬가 대신 그려짐. 충돌·잡기는 그대로."""
    from pxr import Usd, UsdGeom

    n = 0
    for o in objs:
        root = stage.GetPrimAtPath(o.prim_path)
        for prim in Usd.PrimRange(root) if root and root.IsValid() else []:
            if prim.GetName() == "visuals":  # write_rigid_usd 의 /<이름>/base_link/visuals (OmniGibson 이 붙인 경로 아래 어디든)
                UsdGeom.Imageable(prim).MakeInvisible()
                n += 1
    return n
