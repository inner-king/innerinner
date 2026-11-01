# SPDX-License-Identifier: Apache-2.0
"""CPU renderer for the cutting solver: no OpenGL, no Isaac Sim, so a headless box can make a video.

The food is splatted as small spheres into a depth buffer, which is then smoothed and
shaded from its own screen-space normals, so a dense particle block reads as a solid. The
cutting board and the knife are ray-cast exactly, and the food casts a soft shadow on the
board.

The look follows the real clips in Data/tofu_reference (see that folder's sources.json):
tofu's outer skin is a warm off-white with pores and the imprint of the pressing cloth,
while a cut face is lighter, finer and wetter. Particles carry both from where they started
(``tofu_look``), so a face that a cut exposes looks like a cut face.
"""

import math
from dataclasses import dataclass

import numpy as np

from simfoundry.cutting.knife import KnifeGeometry, KnifePose

# Median colour of the tofu's front face in the HDMD close-up (neutral light), and a cut face
# a little lighter and less yellow, as in the Chef'd and HDMD clips.
SKIN_RGB = np.array([229.0, 221.0, 211.0]) / 255.0
CUT_RGB = np.array([238.0, 233.0, 223.0]) / 255.0
STEEL_RGB = np.array([186.0, 191.0, 198.0]) / 255.0
HANDLE_RGB = np.array([34.0, 28.0, 26.0]) / 255.0


def _normalize(v):
    return v / np.linalg.norm(v)


@dataclass(frozen=True)
class Style:
    """Scene dressing and lights.

    Attributes:
        background_top: Background colour at the top of the frame.
        background_bottom: Background colour at the bottom.
        board_rgb: Colour of the cutting board's top.
        board_thickness: Thickness of the board [m]; its edge shows in low views.
        shadow: Darkening of the board under the food (0 = none).
        key_light: Direction towards the main light.
        fill_light: Direction towards a weaker fill light.
        ambient: Ambient light level.
        wrap: Diffuse wrap of the food, for the soft, slightly translucent look of tofu.
        blade_alpha: Opacity of the blade (1 = solid steel).
        outline: Brightness factor on boundaries between separate pieces.
        steel_rgb: Colour of the blade's flat; under a large soft light, as in the Chef'd
            clip, polished steel shows almost white.
    """

    background_top: tuple = (0.97, 0.97, 0.98)
    background_bottom: tuple = (0.86, 0.87, 0.89)
    board_rgb: tuple = (214.0 / 255.0, 181.0 / 255.0, 140.0 / 255.0)
    board_thickness: float = 0.018
    shadow: float = 0.55
    key_light: tuple = (-0.4, -0.6, 1.0)
    fill_light: tuple = (0.8, 0.3, 0.5)
    ambient: float = 0.30
    wrap: float = 0.35
    blade_alpha: float = 1.0
    outline: float = 0.72
    steel_rgb: tuple = tuple(float(c) for c in STEEL_RGB)


STUDIO = Style()
# A dark kitchen like the Chef'd clip: dark backdrop, warm light from the front left.
KITCHEN = Style(
    background_top=(0.16, 0.12, 0.11),
    background_bottom=(0.30, 0.20, 0.15),
    board_rgb=(206.0 / 255.0, 168.0 / 255.0, 128.0 / 255.0),
    key_light=(-0.5, -0.8, 0.9),
    fill_light=(0.9, -0.2, 0.3),
    ambient=0.34,
)


def tofu_look(x0, skin, seed=0, cloth_period=6.0e-3, cloth=0.045):
    """Albedo and gloss of each particle, from where it started.

    Args:
        x0: Initial particle positions, shape (N, 3).
        skin: Whether each particle started on the block's outer skin, shape (N,).
        seed: Seed of the pores and mottling.
        cloth_period: Spacing of the pressing-cloth imprint on the skin [m].
        cloth: How much darker the imprint's lines are; 0 leaves them out (a skin texture
            taken from a photo carries whatever the real skin shows).

    Returns:
        (albedo (N, 3), gloss (N,)).
    """
    rng = np.random.default_rng(seed)
    n = len(x0)
    mottle = np.zeros(n)
    for _ in range(6):
        k = rng.normal(size=3)
        k = k / np.linalg.norm(k) * rng.uniform(150.0, 450.0)  # wavelengths of 1.4-4 cm
        mottle += np.sin(x0 @ k + rng.uniform(0.0, 2.0 * math.pi))
    mottle /= 6.0
    albedo = CUT_RGB[None, :] * (1.0 + 0.015 * mottle)[:, None]
    gloss = np.full(n, 0.55)
    # Pores: small and faint, as in the clips; one particle each, so keep them sparse.
    pores = rng.random(n) < 0.03
    albedo[pores] *= rng.uniform(0.93, 0.97, size=(int(pores.sum()), 1))

    s = np.asarray(skin, dtype=bool)
    if s.any():
        lo, hi = x0.min(axis=0), x0.max(axis=0)
        dist = np.concatenate([x0[s] - lo, hi - x0[s]], axis=1)  # distance to the six faces
        axis = np.argmin(dist, axis=1) % 3
        skin_albedo = SKIN_RGB[None, :] * (1.0 + 0.03 * mottle[s])[:, None]
        # The cloth's imprint: faint lines along both directions of each face, wavering.
        lines = np.zeros(int(s.sum()))
        xs = x0[s]
        for a in range(3):
            on = axis == a
            if not on.any():
                continue
            b, c = [d for d in range(3) if d != a]
            wob = 0.25 * np.sin(xs[on, c] * 250.0 + a) + 0.25 * np.sin(xs[on, b] * 180.0 + 2 * a)
            fb = np.abs(((xs[on, b] / cloth_period + wob) % 1.0) - 0.5)
            fc = np.abs(((xs[on, c] / cloth_period - wob) % 1.0) - 0.5)
            lines[on] = np.maximum(fb > 0.42, fc > 0.42)
        skin_albedo *= (1.0 - cloth * lines)[:, None]
        skin_pores = rng.random(int(s.sum())) < 0.05
        skin_albedo[skin_pores] *= rng.uniform(0.88, 0.95, size=(int(skin_pores.sum()), 1))
        albedo[s] = skin_albedo
        gloss[s] = 0.15
    return np.clip(albedo, 0.0, 1.0), gloss


# Weight of the specular highlight on food: colour = albedo * diffuse + SPEC_GAIN * gloss * specular.
SPEC_GAIN = 0.45


def food_light(style: Style, nrm, forward):
    """The light the renderer puts on food: (diffuse, specular) for unit normals ``nrm``.

    Colour = albedo * diffuse + SPEC_GAIN * gloss * specular, seen along ``forward``.
    ``real2sim.photo_albedo`` divides a photograph by the same light.
    """
    key_dir = _normalize(np.asarray(style.key_light, dtype=float))
    fill_dir = _normalize(np.asarray(style.fill_light, dtype=float))
    key = np.clip((nrm @ key_dir + style.wrap) / (1.0 + style.wrap), 0.0, 1.0)
    fill = np.clip(nrm @ fill_dir, 0.0, 1.0)
    half = _normalize(key_dir - np.asarray(forward, dtype=float))
    spec = np.clip(nrm @ half, 0.0, 1.0) ** 60
    return style.ambient + 0.80 * key + 0.25 * fill, spec


class Camera:
    """Pinhole camera looking from ``eye`` at ``target``.

    ``from_opencv`` builds one from a calibrated camera instead, such as the one SimFoundry's
    Pipeline A recovers for a video frame. Pixel coordinates are continuous: the image spans
    [0, width] x [0, height], and pixel i covers [i, i + 1].
    """

    def __init__(self, eye, target, up=(0.0, 0.0, 1.0), fov_deg=35.0, width=1280, height=720):
        eye = np.asarray(eye, dtype=float)
        forward = _normalize(np.asarray(target, dtype=float) - eye)
        right = _normalize(np.cross(forward, np.asarray(up, dtype=float)))
        focal = 0.5 * height / math.tan(math.radians(fov_deg) / 2.0)
        self._setup(eye, forward, right, np.cross(right, forward), width, height, focal, focal, width / 2.0, height / 2.0)

    @classmethod
    def from_opencv(cls, cam_to_world, K, width, height, calib_size=None):
        """A camera from its pose and intrinsic matrix, in the OpenCV convention.

        Args:
            cam_to_world: 4x4 pose whose rotation columns are the camera's x (right), y (down)
                and z (forward) axes in the world.
            K: 3x3 intrinsic matrix (fx, fy, cx, cy), for an image of ``calib_size`` pixels.
            width: Width of the rendered image; K is scaled to it.
            height: Height of the rendered image.
            calib_size: (width, height) that K belongs to; the rendered size if None.
        """
        T = np.asarray(cam_to_world, dtype=float)
        K = np.asarray(K, dtype=float)
        sx, sy = (1.0, 1.0) if calib_size is None else (width / calib_size[0], height / calib_size[1])
        cam = cls.__new__(cls)
        cam._setup(T[:3, 3], _normalize(T[:3, 2]), _normalize(T[:3, 0]), -_normalize(T[:3, 1]), width, height,
                   K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy)
        return cam

    def _setup(self, eye, forward, right, up, width, height, fx, fy, cx, cy):
        self.eye, self.forward, self.right, self.up = eye, forward, right, up
        self.width, self.height = width, height
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.focal = 0.5 * (fx + fy)  # sizes the particle splats
        px, py = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        dirs = (
            forward[None, None, :]
            + ((px - cx) / fx)[..., None] * right[None, None, :]
            - ((py - cy) / fy)[..., None] * up[None, None, :]
        )
        self.rays = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        # Camera-space depth of a hit at ray parameter t is t * cos_fwd.
        self.cos_fwd = self.rays @ forward

    def project(self, pts):
        """Pixel coordinates and camera-space depth of world points, shape (N, 3)."""
        d = pts - self.eye
        z = d @ self.forward
        px = self.cx + self.fx * (d @ self.right) / z
        py = self.cy - self.fy * (d @ self.up) / z
        return px, py, z


def _box_blur(img, mask, radius):
    """Mean of ``img`` over ``mask`` pixels in a (2r+1)^2 window, by summed-area tables."""
    def window_sum(a):
        p = np.pad(a, ((radius + 1, radius), (radius + 1, radius)))
        s = p.cumsum(0).cumsum(1)
        k = 2 * radius + 1
        return s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]

    m = mask.astype(float)
    num = window_sum(np.where(mask, img, 0.0))
    den = window_sum(m)
    return np.where(mask, num / np.maximum(den, 1e-9), img)


def _ray_box(origin, dirs, lo, hi):
    """Slab test of rays against an axis-aligned box in the rays' own frame.

    Returns:
        (t, face): ray parameter of the hit (inf = miss) and the index of the axis whose
        slab the ray entered through (the hit face's normal axis).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t0 = (lo - origin) * inv
        t1 = (hi - origin) * inv
    near = np.minimum(t0, t1)
    near = np.where(np.isnan(near), -np.inf, near)
    far = np.maximum(t0, t1)
    far = np.where(np.isnan(far), np.inf, far)
    tmin = near.max(axis=-1)
    tmax = far.min(axis=-1)
    hit = (tmax >= tmin) & (tmax > 0.0)
    return np.where(hit, np.maximum(tmin, 0.0), np.inf), near.argmax(axis=-1)


class Renderer:
    """Renders frames of a ``MPMCuttingSim``.

    Args:
        camera: The camera.
        board_z: Height of the cutting board's top face.
        board_rect: (x0, y0, x1, y1) extent of the board.
        knife: Blade geometry, for drawing.
        particle_radius: Radius of a particle splat [m]; about the particle spacing.
        particle_spacing: Initial particle spacing [m]. When given, only particles on the
            surface of their piece are drawn, which is several times faster.
        blade_alpha: Opacity of the blade; overrides the style's when given.
        style: Lights and scene dressing (``STUDIO`` or ``KITCHEN``).
        look: (albedo, gloss) per particle from ``tofu_look``; plain cut-face colour if None.
        photo: Optional photograph of the scene from this camera, (H, W, 3) uint8, drawn
            instead of the backdrop and the board. The board is then the one in the photo: it
            is not drawn, and only the food's shadow is laid on its top face within
            ``board_rect``.
    """

    def __init__(
        self,
        camera,
        board_z,
        board_rect,
        knife: KnifeGeometry,
        particle_radius,
        particle_spacing=None,
        blade_alpha=None,
        style: Style = STUDIO,
        look=None,
        photo=None,
    ):
        self.cam = camera
        self.board_z = board_z
        self.board_rect = board_rect
        self.knife = knife
        self.radius = particle_radius
        self.spacing = particle_spacing
        self.style = style
        self.blade_alpha = style.blade_alpha if blade_alpha is None else blade_alpha
        self.look = look
        if photo is not None:
            photo = np.asarray(photo)
            if photo.shape[:2] != (camera.height, camera.width):
                raise ValueError(f"photo is {photo.shape[1]}x{photo.shape[0]}, camera {camera.width}x{camera.height}")
            photo = photo[..., :3].astype(float) / 255.0
        self.photo = photo
        self.key = _normalize(np.asarray(style.key_light, dtype=float))
        self.fill = _normalize(np.asarray(style.fill_light, dtype=float))

    def _background(self):
        if self.photo is not None:
            return self.photo.copy()
        h = self.cam.height
        t = np.linspace(0.0, 1.0, h)[:, None, None]
        top = np.asarray(self.style.background_top)
        bottom = np.asarray(self.style.background_bottom)
        return np.broadcast_to(top + (bottom - top) * t, (h, self.cam.width, 3)).copy()

    def _shadow_map(self, x):
        """Soft shadow and contact darkening of the food on the board, on a 0.5 mm grid."""
        from scipy.ndimage import gaussian_filter

        x0, y0, x1, y1 = self.board_rect
        cell = 0.5e-3
        nx, ny = int((x1 - x0) / cell) + 1, int((y1 - y0) / cell) + 1
        L = self.key
        if L[2] <= 0.05:
            return None
        proj = x - ((x[:, 2] - self.board_z) / L[2])[:, None] * L[None, :]
        ix = ((proj[:, 0] - x0) / cell).astype(int)
        iy = ((proj[:, 1] - y0) / cell).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        cast = np.zeros((nx, ny))
        cast[ix[ok], iy[ok]] = 1.0
        low = x[:, 2] < self.board_z + 4e-3
        jx = ((x[low, 0] - x0) / cell).astype(int)
        jy = ((x[low, 1] - y0) / cell).astype(int)
        ok2 = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
        contact = np.zeros((nx, ny))
        contact[jx[ok2], jy[ok2]] = 1.0
        soft = gaussian_filter(np.clip(gaussian_filter(cast, 1.5) * 4.0, 0.0, 1.0), 4.0)
        ao = gaussian_filter(contact, 8.0)
        return np.clip(0.8 * soft + 0.9 * np.clip(ao * 3.0, 0.0, 1.0), 0.0, 1.0), cell, nx, ny

    def _board(self, shadow):
        cam = self.cam
        x0, y0, x1, y1 = self.board_rect
        if self.photo is not None:
            return self._photo_board(shadow)
        lo = np.array([x0, y0, self.board_z - self.style.board_thickness])
        hi = np.array([x1, y1, self.board_z])
        t, face = _ray_box(cam.eye, cam.rays, lo, hi)
        hit = cam.eye[None, None, :] + np.where(np.isfinite(t), t, 0.0)[..., None] * cam.rays
        depth = np.where(np.isfinite(t), t * cam.cos_fwd, np.inf)
        base = np.asarray(self.style.board_rgb)
        grain = (
            0.5 + 0.5 * np.sin(hit[..., 1] * 520.0 + 3.0 * np.sin(hit[..., 0] * 37.0))
        ) * 0.6 + 0.4 * (0.5 + 0.5 * np.sin(hit[..., 1] * 2300.0 + 5.0 * np.sin(hit[..., 0] * 90.0)))
        color = base[None, None, :] * (0.90 + 0.10 * grain[..., None])
        top = face == 2
        light = self.style.ambient + (1.0 - self.style.ambient) * max(self.key[2], 0.0)
        color = color * np.where(top, light, 0.7 * light)[..., None]
        if shadow is not None:
            sm, cell, nx, ny = shadow
            ix = np.clip(((hit[..., 0] - x0) / cell).astype(int), 0, nx - 1)
            iy = np.clip(((hit[..., 1] - y0) / cell).astype(int), 0, ny - 1)
            dark = np.where(top & np.isfinite(t), sm[ix, iy], 0.0)
            color = color * (1.0 - self.style.shadow * dark)[..., None]
        return depth, color

    def _photo_board(self, shadow):
        """The photographed board's top face: depth for occlusion, and the photo under the shadow."""
        cam = self.cam
        x0, y0, x1, y1 = self.board_rect
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.board_z - cam.eye[2]) / cam.rays[..., 2]
        hit = cam.eye[None, None, :] + np.where(t > 0.0, t, 0.0)[..., None] * cam.rays
        on = (t > 0.0) & (hit[..., 0] >= x0) & (hit[..., 0] <= x1) & (hit[..., 1] >= y0) & (hit[..., 1] <= y1)
        depth = np.where(on, t * cam.cos_fwd, np.inf)
        color = self.photo.copy()
        if shadow is not None:
            sm, cell, nx, ny = shadow
            ix = np.clip(((hit[..., 0] - x0) / cell).astype(int), 0, nx - 1)
            iy = np.clip(((hit[..., 1] - y0) / cell).astype(int), 0, ny - 1)
            dark = np.where(on, sm[ix, iy], 0.0)
            color = color * (1.0 - self.style.shadow * dark)[..., None]
        return depth, color

    def _knife(self, pose: KnifePose):
        cam = self.cam
        g = self.knife
        e_u, e_v, e_w = pose.axes()
        o = np.asarray(pose.origin, dtype=float)
        H, W = cam.height, cam.width
        depth = np.full((H, W), np.inf)
        color = np.zeros((H, W, 3))
        # Blade: the plane v = 0, in the outline the solver uses (KnifeGeometry.edge_height and
        # spine_height): for a chef's knife, an edge curving up to the tip and a spine
        # curving down to meet it.
        denom = cam.rays @ e_v
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((o - cam.eye) @ e_v) / denom
        hit = cam.eye[None, None, :] + t[..., None] * cam.rays - o
        u = hit @ e_u
        w = hit @ e_w
        above_edge = w - g.edge_height(u)
        on_blade = (t > 0) & (np.abs(u) <= g.half_length) & (above_edge >= 0.0) & (w <= g.spine_height(u))
        facing = np.abs(denom)
        n_view = np.where((denom > 0)[..., None], -e_v, e_v)
        key = np.clip(n_view @ self.key, 0.0, 1.0)
        bevel = np.clip(above_edge / g.bevel_height, 0.0, 1.0)
        brushed = 1.0 + 0.025 * np.sin(u * 1700.0 + 2.0 * np.sin(w * 420.0))
        shade = (0.42 + 0.35 * facing + 0.35 * key) * brushed
        steel_rgb = np.asarray(self.style.steel_rgb, dtype=float)
        steel = steel_rgb[None, None, :] * (1.06 - 0.10 * bevel[..., None]) * shade[..., None]
        steel = np.where((above_edge < 0.6e-3)[..., None], np.array([0.94, 0.95, 0.97]), steel)
        depth = np.where(on_blade, t * cam.cos_fwd, depth)
        color = np.where(on_blade[..., None], np.clip(steel, 0.0, 1.0), color)
        alpha = np.where(on_blade, self.blade_alpha, 1.0)
        # Bolster and handle: boxes past the heel, their tops level with the spine, in the knife
        # frame; the handle's three rivets show on its sides.
        R = np.stack([e_u, e_v, e_w], axis=0)
        ro = R @ (cam.eye - o)
        rd = cam.rays @ R.T
        top = g.height - 0.001
        parts = (
            (np.array([-g.half_length - 0.012, -0.004, max(0.0, g.height - 0.03)]), np.array([-g.half_length, 0.004, top]), "bolster"),
            (np.array([-g.half_length - 0.012 - g.handle_length, -0.010, max(0.0, g.height - 0.026)]),
             np.array([-g.half_length - 0.012, 0.010, top]), "handle"),
        )
        for lo, hi, name in parts:
            th, _ = _ray_box(ro, rd, lo, hi)
            dh = th * cam.cos_fwd
            closer = dh < depth
            if name == "bolster":
                shade_b = (0.5 + 0.3 * facing + 0.3 * key)[..., None]
                c = np.clip(steel_rgb[None, None, :] * shade_b, 0.0, 1.0)
            else:
                c = HANDLE_RGB[None, None, :] * (0.8 + 0.4 * facing[..., None])
                p = ro[None, None, :] + np.where(np.isfinite(th), th, 0.0)[..., None] * rd
                mid_w = 0.5 * (lo[2] + hi[2])
                rivet = np.zeros(th.shape, dtype=bool)
                for f in (0.2, 0.5, 0.8):
                    cu = hi[0] - f * g.handle_length
                    rivet |= (p[..., 0] - cu) ** 2 + (p[..., 2] - mid_w) ** 2 < 0.0025 ** 2
                rivet &= np.abs(np.abs(p[..., 1]) - 0.010) < 1e-4
                c = np.where(rivet[..., None], np.array([0.85, 0.86, 0.88]) * (0.7 + 0.3 * facing[..., None]), c)
            color = np.where(closer[..., None], c, color)
            depth = np.where(closer, dh, depth)
            alpha = np.where(closer, 1.0, alpha)
        return depth, color, alpha

    def _surface(self, x, piece):
        """Particles on the surface of their own piece; the rest are hidden and need no splat."""
        h = self.spacing
        idx = np.floor((x - x.min(0)) / h).astype(np.int64) + 1
        shape = idx.max(0) + 2
        occ = np.zeros(tuple(shape), dtype=np.int16)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = piece + 1
        own = piece + 1
        surf = np.zeros(len(x), dtype=bool)
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            n = occ[idx[:, 0] + d[0], idx[:, 1] + d[1], idx[:, 2] + d[2]]
            surf |= n != own
        return surf

    def _food(self, x, ids):
        """Depth buffer of the particles and, per pixel, the index (into ``x``) of the nearest one."""
        cam = self.cam
        H, W = cam.height, cam.width
        px, py, z = cam.project(x)
        keep = z > 1e-3
        px, py, z, ids = px[keep], py[keep], z[keep], ids[keep]
        r_px = cam.focal * self.radius / z
        rmax = int(math.ceil(r_px.max())) if len(r_px) else 1
        cx = np.floor(px).astype(int)
        cy = np.floor(py).astype(int)
        idx_all, dep_all, id_all = [], [], []
        for oy in range(-rmax, rmax + 1):
            for ox in range(-rmax, rmax + 1):
                ddx = cx + ox + 0.5 - px
                ddy = cy + oy + 0.5 - py
                q = (ddx * ddx + ddy * ddy) / (r_px * r_px)
                ok = (q < 1.0) & (cx + ox >= 0) & (cx + ox < W) & (cy + oy >= 0) & (cy + oy < H)
                if not ok.any():
                    continue
                idx_all.append((cy + oy)[ok] * W + (cx + ox)[ok])
                dep_all.append(z[ok] - self.radius * np.sqrt(1.0 - q[ok]))
                id_all.append(ids[ok])
        depth = np.full(H * W, np.inf)
        nearest = np.full(H * W, -1)
        if idx_all:
            idx = np.concatenate(idx_all)
            dep = np.concatenate(dep_all)
            pid = np.concatenate(id_all)
            # One float64 sort key orders by pixel, then depth (depths are well under 10 m).
            order = np.argsort(idx * 10.0 + dep, kind="stable")
            idx, dep, pid = idx[order], dep[order], pid[order]
            first = np.ones(len(idx), dtype=bool)
            first[1:] = idx[1:] != idx[:-1]
            depth[idx[first]] = dep[first]
            nearest[idx[first]] = pid[first]
        return depth.reshape(H, W), nearest.reshape(H, W)

    def food_pixels(self, x, piece):
        """What the camera sees of the food, as ``render`` shades it.

        Returns:
            (depth (H, W), nearest (H, W) index into ``x`` of the particle drawn at each pixel,
            -1 where no food shows, normal (H, W, 3) the pixel is shaded with, ids of the
            particles that were splatted).
        """
        cam = self.cam
        ids = np.arange(len(x))
        if self.spacing is not None and len(x):
            ids = ids[self._surface(x, piece)]
        fd, near = self._food(x[ids], ids)
        food = near >= 0
        nrm = np.zeros(fd.shape + (3,))
        if food.any():
            # Screen-space normals of the smoothed depth buffer.
            smooth = _box_blur(np.where(food, fd, 0.0), food, 2)
            pts = cam.eye[None, None, :] + (smooth / np.maximum(cam.cos_fwd, 1e-6))[..., None] * cam.rays
            gx = np.zeros_like(pts)
            gy = np.zeros_like(pts)
            gx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
            gy[1:-1, :] = pts[2:, :] - pts[:-2, :]
            nrm = np.cross(gy, gx)
            nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
            flip = (nrm * cam.rays).sum(-1) > 0
            nrm[flip] *= -1.0
        return fd, near, nrm, ids

    def render(self, x, piece, knife_pose=None, overlay=None, dark=None):
        """Renders one frame.

        Args:
            x: Particle positions, shape (N, 3).
            piece: Piece index per particle, shape (N,); boundaries between pieces are outlined.
            knife_pose: Where to draw the knife, or None.
            overlay: Optional list of text lines drawn in the top-left corner.
            dark: Optional mask of particles to draw darker, e.g. the faces of an open slit.

        Returns:
            An (H, W, 3) uint8 image.
        """
        cam = self.cam
        st = self.style
        n = len(x)
        if dark is None:
            dark = np.zeros(n, dtype=bool)
        if self.look is not None:
            albedo, gloss = self.look
        else:
            albedo, gloss = np.tile(CUT_RGB, (n, 1)), np.full(n, 0.5)
        fd, near, nrm, ids = self.food_pixels(x, piece)

        img = self._background()
        depth = np.full((cam.height, cam.width), np.inf)
        bd, bc = self._board(self._shadow_map(x[ids]) if st.shadow > 0 and n else None)
        sel = bd < depth
        img[sel], depth[sel] = bc[sel], bd[sel]

        food = near >= 0
        if food.any():
            k = np.where(food, near, 0)
            fid = np.where(food, piece[k], -1)
            diffuse, spec = food_light(st, nrm, cam.forward)
            col = albedo[k] * diffuse[..., None] + (SPEC_GAIN * gloss[k] * spec)[..., None]
            # Boundaries between separate pieces, and the faces of open slits, read darker.
            edge = np.zeros_like(food)
            edge[1:, :] |= fid[1:, :] != fid[:-1, :]
            edge[:, 1:] |= fid[:, 1:] != fid[:, :-1]
            col = np.where((edge & food)[..., None], col * st.outline, col)
            col = np.where((dark[k] & food)[..., None], col * 0.6, col)
            sel = food & (fd < depth)
            img[sel], depth[sel] = col[sel], fd[sel]

        if knife_pose is not None:
            kd, kc, ka = self._knife(knife_pose)
            sel = kd < depth
            a = ka[sel][:, None]
            img[sel] = a * kc[sel] + (1.0 - a) * img[sel]
            depth[sel] = kd[sel]

        out = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        if overlay:
            out = _draw_text(out, overlay, light=np.mean(st.background_top) < 0.5)
        return out


def _draw_text(img, lines, light=False):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return img
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    y = 14
    for line in lines:
        draw.text((18, y), line, fill=(235, 235, 235) if light else (30, 30, 34), font=font)
        y += 32
    return np.asarray(im)
