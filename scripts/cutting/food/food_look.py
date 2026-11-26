"""로봇 데이터 생성 화면의 겉모습 — '현실 장면' 설정과 겉모습 무작위화 (2026-11-26, reports/81).

food_robot_episodes.py --look {base,real,random} 가 쓴다. 물리(MPM·PhysX 충돌체)는 하나도 바꾸지 않고 그리기만 바꾼다.
  - base  : 아무것도 안 함(지금까지 데이터와 같은 화면 — 빈 흰 배경, 황갈색 도마, OmniGibson 기본 하늘 조명).
  - real  : 팀원 10-25 주방 녹화에 맞춘 한 벌 — 흰 플라스틱 도마, 밝은 회색 화강암 조리대(탁자 둘레까지 넓게), 흰 타일 벽 두 면,
            천장 조명(그림자) + 실내 사진 돔 조명(반사·은은한 빛), 하늘 조명 색은 흰색으로.
  - random: 에피소드마다 도마·조리대·벽 무늬와 색조, 돔 사진(학습용 20 장 중)·세기·방향, 천장 조명 자리·세기·색온도, 앞 카메라 자리를 뽑는다.
            뽑은 값(dict)은 에피소드 json 의 "look" 에 남겨 다시 만들 수 있다.
만드는 것(USD, /World/look 아래, 충돌 없음): 조리대 판(탁자 자리까지 덮고 탁자 윗면보다 1 mm 아래), 도마 상자(충돌 도마와 같은 자리·크기),
벽 둘(카메라 반대쪽 −x·+y), 천장 구 조명. 원래 탁자·도마 그림은 숨긴다(충돌은 그대로).
무늬 그림: scripts/cutting/food/make_look_textures.py → data/looks/textures (실제 크기 size_m 기준으로 UV 를 매김 → 타일·나뭇결 크기가 실물 크기).
돔 사진: data/backgrounds/polyhaven(Poly Haven 실내 HDRI 의 JPG, CC0, LIST.txt 의 train_random 만 무작위에, test_unseen 은 안 씀).
"""
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # 저장소 뿌리
TEX = Path(os.environ.get("FOOD_LOOK_TEXTURES", ROOT / "Data" / "food" / "looks" / "textures"))  # make_look_textures.py 가 만듦
HDRI = Path(os.environ.get("FOOD_LOOK_HDRI", ROOT / "Data" / "food" / "hdri"))  # Poly Haven 실내 HDRI JPG + LIST.txt(없으면 돔 그림은 그대로)

# 팀원 주방(10-25 녹화 사진에서 눈으로 고름) — 수치는 첫 추정값, 실물 사진·카메라를 받으면 맞춘다.
# 11-26 07:10 미리 보기 1차(돔 900·천장 45000 머리 위)가 하얗게 날아가 돔 500·천장 30000·비스듬히(그림자 보이게)로 낮춤
REAL = {
    "kind": "real",
    "board": {"tex": "board_white_plastic", "tint": [1.0, 1.0, 1.0], "rough": 0.55},
    "counter": {"tex": "counter_granite_light", "tint": [1.0, 1.0, 1.0], "rough": 0.45},
    "wall": {"tex": "wall_tile_white", "tint": [1.0, 1.0, 1.0], "rough": 0.25},
    "dome": {"hdri": "small_empty_room_1", "intensity": 500.0, "rot_deg": 30.0, "color": [1.0, 1.0, 1.0]},
    "ceiling": {"dxy": [0.35, -0.25], "h": 1.0, "radius": 0.15, "intensity": 30000.0, "kelvin": 4500.0},
    "cam": {"front_dpos": [0.0, 0.0, 0.0], "front_dlook": [0.0, 0.0, 0.0]},
}


def _hdri_split():
    tr, te = [], []
    p = HDRI / "LIST.txt"
    if p.exists():
        for ln in p.read_text().splitlines():
            s = ln.split()
            if len(s) == 2 and not ln.startswith("#"):
                (tr if s[0] == "train_random" else te).append(s[1])
    return tr, te


def sample_look(kind, seed):
    """kind = base/real/random → look dict(JSON 저장 가능). base 는 None."""
    if kind == "base":
        return None
    if kind == "real":
        return json.loads(json.dumps(REAL))
    rng = np.random.default_rng(int(seed) + 31337)
    idx = json.loads((TEX / "index.json").read_text())
    pick = lambda k, w=None: str(rng.choice([n for n, v in idx.items() if v["kind"] == k], p=w))  # noqa: E731
    boards = [n for n, v in idx.items() if v["kind"] == "board"]
    wb = np.array([0.4 if n == "board_white_plastic" else 0.6 / (len(boards) - 1) for n in boards])
    tr, _ = _hdri_split()
    tint = lambda s: [float(v) for v in np.clip(1.0 + rng.normal(0, s, 3), 0.75, 1.2)]  # noqa: E731
    return {
        "kind": "random", "seed": int(seed),
        "board": {"tex": str(rng.choice(boards, p=wb / wb.sum())), "tint": tint(0.05), "rough": float(rng.uniform(0.4, 0.75))},
        "counter": {"tex": pick("counter"), "tint": tint(0.06), "rough": float(rng.uniform(0.3, 0.7))},
        "wall": {"tex": pick("wall"), "tint": tint(0.06), "rough": float(rng.uniform(0.2, 0.6))},
        "dome": {"hdri": str(rng.choice(tr)) if tr else "small_empty_room_1", "intensity": float(500.0 * rng.uniform(0.5, 1.7)),
                 "rot_deg": float(rng.uniform(0, 360)), "color": [1.0, 1.0, 1.0]},
        "ceiling": {"dxy": [float(rng.uniform(-0.6, 0.6)), float(rng.uniform(-0.5, 0.6))], "h": float(rng.uniform(0.9, 1.6)),
                    "radius": float(rng.uniform(0.1, 0.35)), "intensity": float(30000.0 * rng.uniform(0.4, 1.6)),
                    "kelvin": float(rng.uniform(2700, 6500))},
        "cam": {"front_dpos": [float(v) for v in rng.uniform(-0.02, 0.02, 3)],
                "front_dlook": [float(v) for v in rng.uniform(-0.015, 0.015, 3)]},
    }


def _srgb_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


class SceneLook:
    """장면에 겉모습을 입힌다. table_h: 탁자 윗면 z, table_xy/size: 탁자 가운데·크기(x, y), board_size: (x, y, 두께), board_z0: 도마 밑면 z.
    front: (카메라 이름, 기본 눈 자리, 기본 바라보는 점) — 무작위 카메라 흔들기용. look_at: 팀원 T.look_at."""

    def __init__(self, og, env, cams, *, table_h, table_xy, table_size, board_xy, board_size, board_z0, front, look_at):
        self.og, self.env, self.cams, self.look_at = og, env, cams, look_at
        self.stage = og.sim.stage
        self.th, self.txy, self.tsz = float(table_h), np.asarray(table_xy, float), np.asarray(table_size, float)
        self.bxy, self.bsz, self.bz0 = np.asarray(board_xy, float), np.asarray(board_size, float), float(board_z0)
        self.front = front
        self.idx = json.loads((TEX / "index.json").read_text()) if (TEX / "index.json").exists() else {}  # base 는 무늬 없이도 돎
        self.built = False
        self.hidden = []  # 숨긴 원래 그림 prim
        self.dome0 = None  # 원래 하늘 조명(세기, 색, 그림, 돌림)
        self.current = None

    # ---------------- USD ----------------
    def _material(self, path, tex, tint, rough, metallic=0.0):
        from pxr import Sdf, UsdShade

        mat = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, f"{path}/Shader")
        sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str((TEX / self.idx[tex]["file"]).resolve())))
        sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(tuple(float(v) for v in tint))
        sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(rough))
        sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(metallic))
        o = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
        for so in (mat.CreateSurfaceOutput("mdl"), mat.CreateDisplacementOutput("mdl"), mat.CreateVolumeOutput("mdl")):
            so.ConnectToSource(o)
        return mat, sh

    def _box(self, path, lo, hi, tex_size, top_fit=False):
        """축 정렬 상자 메쉬(면마다 꼭짓점 4개, 면 법선). UV = 면 위 자리[m] ÷ 무늬 실제 크기(top_fit 이면 윗면은 무늬 한 장을 꼭 맞춤)."""
        from pxr import Sdf, UsdGeom, Vt

        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        sx, sy = tex_size
        P, N, UV, F = [], [], [], []
        faces = [  # (법선, 네 꼭짓점 고르기)
            ((0, 0, 1), [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
            ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
            ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
            ((-1, 0, 0), [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
            ((0, 1, 0), [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
            ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
        ]
        for n, corners in faces:
            b = len(P)
            for c in corners:
                p = lo + (hi - lo) * np.array(c, float)
                P.append(p)
                N.append(n)
                if n[2] != 0:  # 윗·아랫면: (x, y)
                    u, v = (p[0] - lo[0], p[1] - lo[1])
                    if top_fit:
                        u, v = u / (hi[0] - lo[0]), v / (hi[1] - lo[1])
                    else:
                        u, v = u / sx, v / sy
                elif n[0] != 0:  # x 면: (y, z)
                    u, v = (p[1] - lo[1]) / sx, (p[2] - lo[2]) / sy
                else:  # y 면: (x, z)
                    u, v = (p[0] - lo[0]) / sx, (p[2] - lo[2]) / sy
                UV.append((u, v))
            F += [b, b + 1, b + 2, b, b + 2, b + 3]
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(P, np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F) // 3, 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(F, np.int32)))
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(N, np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        st = UsdGeom.PrimvarsAPI(m.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray.FromNumpy(np.asarray(UV, np.float32)))
        m.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([lo, hi]).astype(np.float32)))
        return m

    def _hide_originals(self):
        from pxr import Usd, UsdGeom

        for name in ("table", "board"):
            obj = self.env.scene.object_registry("name", name)
            if obj is None:
                continue
            for p in Usd.PrimRange(self.stage.GetPrimAtPath(obj.prim_path)):
                if "/visuals" in str(p.GetPath()) and p.IsA(UsdGeom.Gprim):
                    UsdGeom.Imageable(p).MakeInvisible()
                    self.hidden.append(p)

    def build(self):
        """처음 한 번: 상자·벽·조명 prim 과 재질 자리를 만든다(무늬는 apply 가 정함)."""
        from pxr import Gf, UsdGeom, UsdLux, UsdShade

        UsdGeom.Xform.Define(self.stage, "/World/look")
        th, (tx, ty), (sx, sy) = self.th, self.txy, self.tsz
        # 조리대: 탁자 둘레까지 넓게(x −1.0 ~ +1.2, y −0.9 ~ +0.95 m), 윗면 = 탁자 윗면 − 1 mm(탁자 그림은 숨김)
        self.x_wall, self.y_wall = tx - 1.0, ty + 0.95
        self.g_counter = self._box("/World/look/counter", (self.x_wall, ty - 0.9, th - 0.04), (tx + 1.2, self.y_wall, th - 0.001), (1.6, 1.2))
        bx, by = self.bxy
        bsx, bsy, bt = self.bsz
        self.g_board = self._box("/World/look/board", (bx - bsx / 2, by - bsy / 2, self.bz0), (bx + bsx / 2, by + bsy / 2, self.bz0 + bt),
                                 (bsx, bsy), top_fit=True)
        self.g_wall_y = self._box("/World/look/wall_back", (self.x_wall, self.y_wall, th - 0.04), (tx + 1.2, self.y_wall + 0.02, th + 1.1), (2.4, 1.2))
        self.g_wall_x = self._box("/World/look/wall_left", (self.x_wall - 0.02, ty - 0.9, th - 0.04), (self.x_wall, self.y_wall, th + 1.1), (2.4, 1.2))
        self.mats = {}
        for key, g in (("board", self.g_board), ("counter", self.g_counter), ("wall", self.g_wall_y)):
            self.mats[key] = self._material(f"/World/look/Looks/{key}", REAL[key]["tex"], REAL[key]["tint"], REAL[key]["rough"])
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(self.mats[key][0])
        UsdShade.MaterialBindingAPI.Apply(self.g_wall_x.GetPrim()).Bind(self.mats["wall"][0])
        self.light = UsdLux.SphereLight.Define(self.stage, "/World/look/ceiling")
        self.light.CreateRadiusAttr(0.2)
        self.light.CreateIntensityAttr(45000.0)
        self.light.CreateEnableColorTemperatureAttr(True)
        self.light.CreateColorTemperatureAttr(4800.0)
        self.light_xf = UsdGeom.Xformable(self.light.GetPrim()).AddTranslateOp()
        self.light_xf.Set(Gf.Vec3d(float(tx), float(ty), float(th + 1.25)))
        self.built = True

    def _set_tex(self, key, spec):
        from pxr import Sdf

        _, sh = self.mats[key]
        sh.GetInput("diffuse_texture").Set(Sdf.AssetPath(str((TEX / self.idx[spec["tex"]]["file"]).resolve())))
        sh.GetInput("diffuse_tint").Set(tuple(float(v) for v in spec["tint"]))
        sh.GetInput("reflection_roughness_constant").Set(float(spec["rough"]))

    def _set_visible(self, on):
        from pxr import UsdGeom

        img = UsdGeom.Imageable(self.stage.GetPrimAtPath("/World/look"))
        (img.MakeVisible if on else img.MakeInvisible)()
        for p in self.hidden:
            (UsdGeom.Imageable(p).MakeInvisible if on else UsdGeom.Imageable(p).MakeVisible)()

    def apply(self, look):
        """look(dict) 을 입힌다. None 이면 원래 화면으로 되돌림."""
        from pxr import Gf

        sky = getattr(self.og.sim, "skybox", None)
        if self.dome0 is None and sky is not None:
            self.dome0 = (sky.intensity, tuple(float(v) for v in sky.color), sky.texture_file_path)
        if look is None:
            if self.built:
                self._set_visible(False)
            if sky is not None and self.dome0 is not None:
                sky.intensity, sky.color, sky.texture_file_path = self.dome0[0], self.dome0[1], self.dome0[2]
            self._set_front((0, 0, 0), (0, 0, 0))
            self.current = None
            return
        if not self.built:
            self.build()
            self._hide_originals()
        self._set_visible(True)
        for key in ("board", "counter", "wall"):
            self._set_tex(key, look[key])
        c = look["ceiling"]
        self.light.GetRadiusAttr().Set(float(c["radius"]))
        self.light.GetIntensityAttr().Set(float(c["intensity"]))
        self.light.GetColorTemperatureAttr().Set(float(c["kelvin"]))
        self.light_xf.Set(Gf.Vec3d(float(self.txy[0] + c["dxy"][0]), float(self.txy[1] + c["dxy"][1]), float(self.th + c["h"])))
        d = look["dome"]
        if sky is not None:
            sky.intensity = float(d["intensity"])
            sky.color = tuple(float(v) for v in d.get("color", (1.0, 1.0, 1.0)))
            if (HDRI / f"{d['hdri']}.jpg").exists():
                sky.texture_file_path = str((HDRI / f"{d['hdri']}.jpg").resolve())
            try:
                import omnigibson.utils.transform_utils as TU
                import torch as th_

                sky.set_position_orientation(orientation=TU.euler2quat(th_.tensor([0.0, 0.0, math.radians(d["rot_deg"])])))
            except Exception as e:  # 돌림은 덤 — 안 되면 그대로
                print(f"[food_look] 돔 돌림 생략: {e}", flush=True)
        cam = look.get("cam", {})
        self._set_front(cam.get("front_dpos", (0, 0, 0)), cam.get("front_dlook", (0, 0, 0)))
        self.current = look

    def _set_front(self, dpos, dlook):
        import torch as th_

        if self.front is None or self.front[0] not in self.cams:
            return
        name, eye, tgt = self.front
        e = np.asarray(eye, float) + np.asarray(dpos, float)
        t = np.asarray(tgt, float) + np.asarray(dlook, float)
        self.cams[name].set_position_orientation(position=th_.tensor(e, dtype=th_.float32),
                                                 orientation=th_.tensor(np.asarray(self.look_at(e, t), float), dtype=th_.float32))
