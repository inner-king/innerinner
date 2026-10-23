# SPDX-License-Identifier: Apache-2.0
"""CPU renderer for the cutting solver: no OpenGL, no Isaac Sim, so a headless box can make a video.

The food is splatted as small spheres into a depth buffer, which is then smoothed and
shaded from its own screen-space normals, so a dense particle block reads as a solid with
clean cut faces. The cutting board and the knife are ray-cast exactly. Piece boundaries
are outlined so a closed cut is still visible.
"""

import math

import numpy as np

from simfoundry.cutting.knife import KnifeGeometry, KnifePose

TOFU_RGB = np.array([241.0, 236.0, 220.0]) / 255.0
STEEL_RGB = np.array([176.0, 182.0, 190.0]) / 255.0
HANDLE_RGB = np.array([48.0, 38.0, 34.0]) / 255.0


def _normalize(v):
    return v / np.linalg.norm(v)


class Camera:
    """Pinhole camera looking from ``eye`` at ``target``."""

    def __init__(self, eye, target, up=(0.0, 0.0, 1.0), fov_deg=35.0, width=1280, height=720):
        self.eye = np.asarray(eye, dtype=float)
        self.forward = _normalize(np.asarray(target, dtype=float) - self.eye)
        self.right = _normalize(np.cross(self.forward, np.asarray(up, dtype=float)))
        self.up = np.cross(self.right, self.forward)
        self.width, self.height = width, height
        self.focal = 0.5 * height / math.tan(math.radians(fov_deg) / 2.0)
        px, py = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        dirs = (
            self.forward[None, None, :]
            + ((px - width / 2.0) / self.focal)[..., None] * self.right[None, None, :]
            - ((py - height / 2.0) / self.focal)[..., None] * self.up[None, None, :]
        )
        self.rays = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        # Camera-space depth of a hit at ray parameter t is t * cos_fwd.
        self.cos_fwd = self.rays @ self.forward

    def project(self, pts):
        """Pixel coordinates and camera-space depth of world points, shape (N, 3)."""
        d = pts - self.eye
        z = d @ self.forward
        px = self.width / 2.0 + self.focal * (d @ self.right) / z
        py = self.height / 2.0 - self.focal * (d @ self.up) / z
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
    """Slab test of rays against an axis-aligned box in the rays' own frame; returns t (inf = miss)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t0 = (lo - origin) * inv
        t1 = (hi - origin) * inv
    tmin = np.nanmax(np.minimum(t0, t1), axis=-1)
    tmax = np.nanmin(np.maximum(t0, t1), axis=-1)
    hit = (tmax >= tmin) & (tmax > 0.0)
    return np.where(hit, np.maximum(tmin, 0.0), np.inf)


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
        blade_alpha: Opacity of the blade, below 1 to see the cut through it.
        light_dir: Direction towards the key light.
    """

    def __init__(
        self,
        camera,
        board_z,
        board_rect,
        knife: KnifeGeometry,
        particle_radius,
        particle_spacing=None,
        blade_alpha=0.6,
        light_dir=(-0.4, -0.6, 1.0),
    ):
        self.cam = camera
        self.board_z = board_z
        self.board_rect = board_rect
        self.knife = knife
        self.radius = particle_radius
        self.spacing = particle_spacing
        self.blade_alpha = blade_alpha
        self.light = _normalize(np.asarray(light_dir, dtype=float))
        rng = np.random.default_rng(3)
        self.piece_tint = 1.0 + rng.uniform(-0.035, 0.035, size=(64, 3))
        self.piece_tint[0] = 1.0

    def _background(self):
        h = self.cam.height
        t = np.linspace(0.0, 1.0, h)[:, None, None]
        top = np.array([0.97, 0.97, 0.98])
        bottom = np.array([0.86, 0.87, 0.89])
        return np.broadcast_to(top + (bottom - top) * t, (h, self.cam.width, 3)).copy()

    def _board(self):
        cam = self.cam
        dz = cam.rays[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.board_z - cam.eye[2]) / dz
        hit = cam.eye[None, None, :] + t[..., None] * cam.rays
        x0, y0, x1, y1 = self.board_rect
        ok = (t > 0) & (hit[..., 0] >= x0) & (hit[..., 0] <= x1) & (hit[..., 1] >= y0) & (hit[..., 1] <= y1)
        depth = np.where(ok, t * cam.cos_fwd, np.inf)
        grain = 0.5 + 0.5 * np.sin(hit[..., 1] * 900.0 + 3.0 * np.sin(hit[..., 0] * 140.0))
        base = np.array([0.80, 0.64, 0.45])
        color = base[None, None, :] * (0.90 + 0.10 * grain[..., None])
        return depth, color * (0.55 + 0.45 * max(self.light[2], 0.0))

    def _knife(self, pose: KnifePose):
        cam = self.cam
        g = self.knife
        e_u, e_v, e_w = pose.axes()
        o = np.asarray(pose.origin, dtype=float)
        H, W = cam.height, cam.width
        depth = np.full((H, W), np.inf)
        color = np.zeros((H, W, 3))
        # Blade: the plane v = 0, with a rounded tip.
        denom = cam.rays @ e_v
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((o - cam.eye) @ e_v) / denom
        hit = cam.eye[None, None, :] + t[..., None] * cam.rays - o
        u = hit @ e_u
        w = hit @ e_w
        tip = 0.35 * g.height
        w_max = np.where(
            u > g.half_length - tip,
            g.height * np.sqrt(np.clip(1.0 - ((u - (g.half_length - tip)) / tip) ** 2, 0.0, 1.0)),
            g.height,
        )
        on_blade = (t > 0) & (np.abs(u) <= g.half_length) & (w >= 0.0) & (w <= w_max)
        facing = np.abs(denom)
        shade = 0.55 + 0.45 * facing
        bevel = np.clip(w / g.bevel_height, 0.0, 1.0)
        steel = STEEL_RGB[None, None, :] * (1.08 - 0.12 * bevel[..., None]) * shade[..., None]
        steel = np.where((w < 0.6e-3)[..., None], np.array([0.93, 0.94, 0.96]), steel)
        depth = np.where(on_blade, t * cam.cos_fwd, depth)
        color = np.where(on_blade[..., None], steel, color)
        alpha = np.where(on_blade, self.blade_alpha, 1.0)
        # Handle: a box past the heel, in the knife frame.
        R = np.stack([e_u, e_v, e_w], axis=0)
        ro = R @ (cam.eye - o)
        rd = cam.rays @ R.T
        lo = np.array([-g.half_length - g.handle_length, -0.011, 0.18 * g.height])
        hi = np.array([-g.half_length, 0.011, 0.78 * g.height])
        th = _ray_box(ro, rd, lo, hi)
        dh = th * cam.cos_fwd
        closer = dh < depth
        color = np.where(closer[..., None], HANDLE_RGB[None, None, :] * (0.8 + 0.2 * facing[..., None]), color)
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

    def _food(self, x, piece, dark):
        cam = self.cam
        H, W = cam.height, cam.width
        if self.spacing is not None and len(x):
            keep = self._surface(x, piece)
            x, piece, dark = x[keep], piece[keep], dark[keep]
        px, py, z = cam.project(x)
        keep = z > 1e-3
        px, py, z, piece, dark = px[keep], py[keep], z[keep], piece[keep], dark[keep]
        r_px = cam.focal * self.radius / z
        rmax = int(math.ceil(r_px.max())) if len(r_px) else 1
        cx = np.floor(px).astype(int)
        cy = np.floor(py).astype(int)
        idx_all, dep_all, pid_all, dark_all = [], [], [], []
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
                pid_all.append(piece[ok])
                dark_all.append(dark[ok])
        depth = np.full(H * W, np.inf)
        pid = np.full(H * W, -1)
        drk = np.zeros(H * W, dtype=bool)
        if idx_all:
            idx = np.concatenate(idx_all)
            dep = np.concatenate(dep_all)
            pc = np.concatenate(pid_all)
            dk = np.concatenate(dark_all)
            # One float64 sort key orders by pixel, then depth (depths are well under 10 m).
            order = np.argsort(idx * 10.0 + dep, kind="stable")
            idx, dep, pc, dk = idx[order], dep[order], pc[order], dk[order]
            first = np.ones(len(idx), dtype=bool)
            first[1:] = idx[1:] != idx[:-1]
            depth[idx[first]] = dep[first]
            pid[idx[first]] = pc[first]
            drk[idx[first]] = dk[first]
        return depth.reshape(H, W), pid.reshape(H, W), drk.reshape(H, W)

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
        img = self._background()
        depth = np.full((cam.height, cam.width), np.inf)
        bd, bc = self._board()
        sel = bd < depth
        img[sel], depth[sel] = bc[sel], bd[sel]

        if dark is None:
            dark = np.zeros(len(x), dtype=bool)
        fd, fid, fdark = self._food(x, piece, dark)
        food = np.isfinite(fd)
        if food.any():
            # Shade from screen-space normals of the smoothed depth buffer, per piece so that
            # two halves pressed together still show the cut between them.
            smooth = _box_blur(np.where(food, fd, 0.0), food, 2)
            pts = cam.eye[None, None, :] + (smooth / np.maximum(cam.cos_fwd, 1e-6))[..., None] * cam.rays
            gx = np.zeros_like(pts)
            gy = np.zeros_like(pts)
            gx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
            gy[1:-1, :] = pts[2:, :] - pts[:-2, :]
            n = np.cross(gy, gx)
            n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
            flip = (n * cam.rays).sum(-1) > 0
            n[flip] *= -1.0
            lam = np.clip(n @ self.light, 0.0, 1.0)
            half = _normalize(self.light - cam.forward)
            spec = np.clip(n @ half, 0.0, 1.0) ** 40
            tint = self.piece_tint[np.clip(fid, 0, 63)]
            col = TOFU_RGB[None, None, :] * tint * (0.42 + 0.58 * lam)[..., None] + 0.10 * spec[..., None]
            # Outline piece boundaries and depth jumps.
            edge = np.zeros_like(food)
            jump = 2.5 * self.radius
            fdf = np.where(food, fd, 1e9)
            diff_y = (fid[1:, :] != fid[:-1, :]) | (np.abs(fdf[1:, :] - fdf[:-1, :]) > jump)
            diff_x = (fid[:, 1:] != fid[:, :-1]) | (np.abs(fdf[:, 1:] - fdf[:, :-1]) > jump)
            edge[1:, :] |= diff_y
            edge[:, 1:] |= diff_x
            col = np.where(((edge | fdark) & food)[..., None], col * 0.55, col)
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
            out = _draw_text(out, overlay)
        return out


def _draw_text(img, lines):
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
        draw.text((18, y), line, fill=(30, 30, 34), font=font)
        y += 32
    return np.asarray(im)
