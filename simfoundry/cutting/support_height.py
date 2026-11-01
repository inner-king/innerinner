# SPDX-License-Identifier: Apache-2.0
"""Height of a knife's lowest edge point above a support plane: the one definition the tofu policy uses.

The policy does not command the hand's absolute height. It commands h, the height of the lowest point
of the knife's cutting edge above the support plane (the cutting board), and learns u = ln(h + EPS_M).
The data converter and the evaluation client both call these functions, so a height means the same
thing when a demonstration is converted and when a policy is executed.

Why the lowest point of the whole edge, heel to tip, and not "the part over the food": the blade is
what meets the board, so the lowest point of the edge is what the board stops. A point of the edge
that lies beyond the food still sets the height.

Frames: the hand pose and the support height must share one frame with +z up, and the support plane
is horizontal. Nothing here converts between frames; the caller does.

Why the inverse is closed form: for a fixed hand orientation, moving the hand in z moves every point
of the edge by the same amount, so the lowest point does not change and h changes one for one with
the hand's z. ``hand_z_for_support_height`` therefore solves for z exactly, with no iteration.

Why EPS_M is fixed: u = ln(h + ε) is undefined at h = -ε, and the value of ε is part of what the
policy learns. It is fixed before any result is seen, and changing it changes every target.

The knife's pose relative to the hand (``knife_in_eef``) cannot be taken from the setup file: the value
there matches only the first episode of a process, because the knife is grasped again for each
episode. ``fit_knife_in_eef`` estimates it from logged hand poses and the logged edge heights.

Only numpy is imported at module level (scipy inside ``fit_knife_in_eef``), so a converter outside
the repository can use this module without Isaac Sim, torch or OmniGibson.
"""

import numpy as np

from simfoundry.cutting.knife import CHEF_KNIFE, KnifeGeometry

#: ε of u = ln(h + ε) [m]. Fixed before any result was seen.
EPS_M = 1.0e-3

#: Points along the edge. The generator logs ``edge_z`` with this many, so the two agree.
EDGE_SAMPLES = 100


def edge_points(geometry: KnifeGeometry = CHEF_KNIFE, samples: int = EDGE_SAMPLES) -> np.ndarray:
    """The cutting edge in the knife frame, from heel to tip.

    Args:
        geometry: The blade. Its ``edge_height`` is the only source of the edge's shape.
        samples: Number of points along the edge.

    Returns:
        ndarray of shape (3, samples): the e_u, e_v and e_w coordinates of each point.
    """
    u = np.linspace(-geometry.half_length, geometry.half_length, samples)
    return np.stack([u, np.zeros(samples), geometry.edge_height(u)])


def lowest_edge_z(eef, knife_in_eef, geometry: KnifeGeometry = CHEF_KNIFE, samples: int = EDGE_SAMPLES):
    """Height of the edge's lowest point.

    The knife's pose in the world is K = eef @ knife_in_eef. Its edge points are K's origin plus the
    edge in K's axes, so the z of each is K[2, 3] + K[2, :3] @ (edge point), and the lowest is the
    minimum of those over the edge.

    Args:
        eef: Hand pose(s), shape (..., 4, 4), +z up.
        knife_in_eef: The knife's pose in the hand frame, shape (4, 4).
        geometry: The blade.
        samples: Points along the edge.

    Returns:
        Height of the lowest edge point [m], shape (...).
    """
    K = np.asarray(eef, dtype=float) @ np.asarray(knife_in_eef, dtype=float)
    along = K[..., 2, :3] @ edge_points(geometry, samples)
    return K[..., 2, 3] + along.min(axis=-1)


def support_height(eef, knife_in_eef, support_z, geometry: KnifeGeometry = CHEF_KNIFE, samples: int = EDGE_SAMPLES):
    """Height h of the edge's lowest point above the support plane [m].

    Args:
        eef: Hand pose(s), shape (..., 4, 4), +z up.
        knife_in_eef: The knife's pose in the hand frame, shape (4, 4).
        support_z: Height of the support plane [m], a scalar or broadcastable to eef's batch shape.
        geometry: The blade.
        samples: Points along the edge.

    Returns:
        h [m], shape (...). Negative when the edge is below the plane.
    """
    return lowest_edge_z(eef, knife_in_eef, geometry, samples) - np.asarray(support_z, dtype=float)


def hand_z_for_support_height(
    eef, knife_in_eef, support_z, height, geometry: KnifeGeometry = CHEF_KNIFE, samples: int = EDGE_SAMPLES
):
    """The hand's z that puts the edge's lowest point at ``height`` above the support plane.

    Moving the hand in z moves h one for one (see the module docstring), so the answer is the current
    z plus the difference between the target and the current h. Nothing is iterated.

    Args:
        eef: Hand pose(s), shape (..., 4, 4), +z up.
        knife_in_eef: The knife's pose in the hand frame, shape (4, 4).
        support_z: Height of the support plane [m].
        height: Target h [m], a scalar or broadcastable to eef's batch shape.
        geometry: The blade.
        samples: Points along the edge.

    Returns:
        The hand's z [m], shape (...). The orientation and x, y are left as they are.
    """
    eef = np.asarray(eef, dtype=float)
    current = support_height(eef, knife_in_eef, support_z, geometry, samples)
    return eef[..., 2, 3] + (np.asarray(height, dtype=float) - current)


def log_height(height, eps: float = EPS_M) -> np.ndarray:
    """u = ln(h + eps), the value the policy learns.

    Args:
        height: h [m], a scalar or array. Every h + eps must be positive.
        eps: The offset ε [m]. Use EPS_M unless an experiment says otherwise.

    Returns:
        u, with the shape of ``height``.

    Raises:
        ValueError: If any h + eps <= 0. The message gives how many, and the smallest h.
    """
    h = np.asarray(height, dtype=float)
    shifted = h + eps
    bad = shifted <= 0.0
    if np.any(bad):
        raise ValueError(
            f"{int(np.count_nonzero(bad))} height(s) make h + eps <= 0 (eps = {eps:g} m); "
            f"the smallest h is {float(h.min()):g} m"
        )
    return np.log(shifted)


def height_from_log(u, eps: float = EPS_M) -> np.ndarray:
    """Inverse of ``log_height``: h = exp(u) - eps [m]."""
    return np.exp(np.asarray(u, dtype=float)) - eps


def hand_z_from_log_height(
    eef, knife_in_eef, support_z, u, eps: float = EPS_M, geometry: KnifeGeometry = CHEF_KNIFE, samples: int = EDGE_SAMPLES
):
    """The hand's z for a policy output u = ln(h + eps).

    Args:
        eef: Hand pose(s), shape (..., 4, 4), +z up.
        knife_in_eef: The knife's pose in the hand frame, shape (4, 4).
        support_z: Height of the support plane [m].
        u: The policy's output, a scalar or broadcastable to eef's batch shape.
        eps: The offset ε [m].
        geometry: The blade.
        samples: Points along the edge.

    Returns:
        The hand's z [m], shape (...).
    """
    height = height_from_log(u, eps)
    return hand_z_for_support_height(eef, knife_in_eef, support_z, height, geometry, samples)


def state_with_support(state, support_z) -> np.ndarray:
    """A float32 state vector with the support height appended as its last entry.

    The dataset stores float32 states, so the policy's observation is float32 too. The input is not
    modified.

    Args:
        state: A 1-D state vector.
        support_z: Height of the support plane [m].

    Returns:
        float32 ndarray of length len(state) + 1.

    Raises:
        ValueError: If ``state`` is not 1-D.
    """
    state = np.asarray(state)
    if state.ndim != 1:
        raise ValueError(f"state must be 1-D, got shape {state.shape}")
    return np.append(state.astype(np.float32), np.float32(support_z))


#: The two layouts of a log-support dataset's state: the hand only, or the hand with the support height last.
STATE_SUPPORT_MODES = ("none", "append")


def support_state_mode(meta) -> str:
    """Whether a log-support dataset's state ends with the support height s.

    s is the same number in every demonstration, so its normalised std is 0 and a policy that reads it divides by
    1e-8 at serve time. The default layout therefore leaves it out.

    Args:
        meta: The parsed ``meta/support_height.json`` of the dataset.

    Returns:
        ``"append"`` or ``"none"``. A dataset written before the key existed has s appended (the only layout it
        could have), so a missing key reads as ``"append"``.

    Raises:
        ValueError: If the key holds neither mode.
    """
    mode = meta.get("state_support", "append")
    if mode not in STATE_SUPPORT_MODES:
        raise ValueError(f"state_support {mode!r} is neither 'none' nor 'append'")
    return mode


def support_drop_near_constant(meta) -> bool:
    """Whether a log-support dataset's state leaves out the two finger entries.

    The converter's --drop-near-constant removes the finger columns (their std is 0.0008), so the state is the pose
    only: seven entries, eight with s appended.

    Args:
        meta: The parsed ``meta/support_height.json`` of the dataset.

    Returns:
        True or False. A dataset written before the key existed kept the fingers, so a missing key reads as False.

    Raises:
        ValueError: If the key holds something other than a boolean.
    """
    drop = meta.get("drop_near_constant", False)
    if not isinstance(drop, bool):
        raise ValueError(f"drop_near_constant {drop!r} is not a boolean")
    return drop


def fit_knife_in_eef(
    eefs,
    edge_z,
    k0,
    *,
    geometry: KnifeGeometry = CHEF_KNIFE,
    samples: int = EDGE_SAMPLES,
    restarts: int = 12,
    seed: int = 0,
):
    """Estimate the knife's pose in the hand frame from logged hand poses and edge heights.

    Least squares over the six parameters of knife_in_eef (a rotation vector and a translation),
    starting from ``k0``. The first start is exactly ``k0``; the others perturb it with
    ``numpy.random.default_rng(seed)``. The start with the smallest cost wins.

    Only some of the six parameters are well determined. Trust the prediction, not the matrix: the
    heights that K predicts for hand poses like the ones it was fitted on are what it is good for,
    and a K that differs from the truth in an unobservable direction can still predict them well.

    Args:
        eefs: Actual hand poses, shape (n, 4, 4), +z up, in the same frame as ``edge_z``.
        edge_z: The lowest edge height at each pose [m], shape (n,).
        k0: Starting knife_in_eef, shape (4, 4).
        geometry: The blade.
        samples: Points along the edge.
        restarts: Number of starts, including the one at ``k0``. At least 1.
        seed: Seed of the random starts.

    Returns:
        (K, rms): K is the fitted knife_in_eef, shape (4, 4). rms is the root-mean-square residual
        [m] of the predicted edge heights at the fitted K.

    Raises:
        ValueError: If the shapes or ``restarts`` are not as documented.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    eefs = np.asarray(eefs, dtype=float)
    edge_z = np.asarray(edge_z, dtype=float)
    k0 = np.asarray(k0, dtype=float)
    if eefs.ndim != 3 or eefs.shape[1:] != (4, 4):
        raise ValueError(f"eefs must have shape (n, 4, 4), got {eefs.shape}")
    if edge_z.shape != (eefs.shape[0],):
        raise ValueError(f"edge_z must have shape ({eefs.shape[0]},), got {edge_z.shape}")
    if restarts < 1:
        raise ValueError(f"restarts must be at least 1, got {restarts}")

    def knife_from(x):
        K = np.eye(4)
        K[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
        K[:3, 3] = x[3:]
        return K

    def residual(x):
        return lowest_edge_z(eefs, knife_from(x), geometry, samples) - edge_z

    x_start = np.concatenate([Rotation.from_matrix(k0[:3, :3]).as_rotvec(), k0[:3, 3]])
    rng = np.random.default_rng(seed)
    best = None
    for k in range(restarts):
        x0 = x_start.copy()
        if k:
            x0[:3] = (Rotation.from_rotvec(rng.normal(size=3) * 0.5) * Rotation.from_rotvec(x_start[:3])).as_rotvec()
            x0[3:] += rng.normal(size=3) * 0.03
        fit = least_squares(residual, x0, x_scale=[0.1] * 3 + [0.01] * 3, xtol=1e-12, ftol=1e-12)
        if best is None or fit.cost < best.cost:
            best = fit
    rms = float(np.sqrt(np.mean(best.fun**2)))
    return knife_from(best.x), rms
