# SPDX-License-Identifier: Apache-2.0
"""Turn the cut pieces into closed triangle meshes, so they can go on as ordinary assets.

The MPM solver ends with particles. A rigid-body simulator (PhysX in OmniGibson) and the
scene editors want meshes, so each piece is voxelised at its particle spacing, the
occupancy is smoothed, and its 0.5 level set is extracted with marching cubes. The
smoothing closes the gaps a jittered particle lattice leaves; the mesh is watertight
because the grid is padded on every side.
"""

import numpy as np


def piece_meshes(x, piece, spacing, smoothing=0.6):
    """Closed surface mesh of every piece.

    Args:
        x: Particle positions, shape (N, 3) [m].
        piece: Piece index per particle, shape (N,).
        spacing: Initial particle spacing [m]; the voxel size.
        smoothing: Gaussian smoothing of the occupancy, in voxels.

    Returns:
        Dict piece index -> trimesh.Trimesh in world coordinates.
    """
    import trimesh
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    meshes = {}
    for p in np.unique(piece):
        pts = x[piece == p]
        if len(pts) < 8:
            continue
        lo = pts.min(0) - 3.0 * spacing
        idx = np.floor((pts - lo) / spacing).astype(int)
        shape = idx.max(0) + 4
        occ = np.zeros(tuple(shape), dtype=np.float32)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
        field = gaussian_filter(occ, smoothing)
        # Voxels holding a particle sit at about field value 0.5 on a flat face after smoothing.
        level = 0.5 * field[idx[:, 0], idx[:, 1], idx[:, 2]].mean()
        verts, faces, _, _ = marching_cubes(field, level=level, spacing=(spacing,) * 3)
        mesh = trimesh.Trimesh(vertices=verts + lo + 0.5 * spacing, faces=faces, process=True)
        if mesh.volume < 0:
            mesh.invert()
        meshes[int(p)] = mesh
    return meshes


def export_pieces(x, piece, spacing, out_dir, density=None):
    """Writes ``piece_<i>.obj`` per piece plus a JSON summary; returns the summary.

    Args:
        x: Particle positions, shape (N, 3) [m].
        piece: Piece index per particle, shape (N,).
        spacing: Initial particle spacing [m].
        out_dir: Directory to write into (created if missing).
        density: Material density [kg/m^3], to report each mesh's mass.
    """
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for p, mesh in sorted(piece_meshes(x, piece, spacing).items()):
        path = out / f"piece_{p:02d}.obj"
        mesh.export(path)
        entry = dict(
            piece=p, file=path.name, faces=len(mesh.faces), watertight=bool(mesh.is_watertight),
            volume_cm3=round(float(mesh.volume) * 1e6, 3),
            particle_volume_cm3=round(float((piece == p).sum()) * spacing**3 * 1e6, 3),
            centroid=np.round(mesh.centroid, 5).tolist(),
        )
        if density is not None:
            entry["mass_g"] = round(float(mesh.volume) * density * 1e3, 2)
        summary.append(entry)
    (out / "pieces.json").write_text(json.dumps(summary, indent=2))
    return summary
