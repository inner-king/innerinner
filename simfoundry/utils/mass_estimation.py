# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mass of a reconstructed rigid object from two VLM answers and its mesh.

Stage 11 asked its VLM for an object's mass directly, from the object's photo, its bounding box and a volume. Scored
against measured masses (28 YCB objects, Calli et al., arXiv:1502.03143, Table II; repro/README.md, chapter 29),
Qwen3.5-9B answered that question off by a median factor of 1.48 and within a factor 2 of the truth for 64% of the
objects, with a sealed soup can at 50 g for 349 g, and three answers cut off at the token limit, which stops the
stage. Physical-property work (image2mass, NeRF2Physics) splits mass into a density and a geometry instead; this asks
two questions:

- typical (prompt_typical_mass): what a real object of this kind and size weighs, the VLM's prior for the category;
- material (prompt_material_mass): what the object is made of and how. It gives the density, whether the object is
  solid or a thin shell, the shell's wall thickness and any contents, and the mass follows from the mesh
  (material_mass_g), so the VLM does no arithmetic on a volume it is handed.

Neither alone did better than the direct question on YCB. The material mass held within a factor 2 of the typical
one (combine_masses), or whichever of the two answered, did: a mass for all 28 objects, a median factor of 1.42, 82%
within a factor 2 and a worst factor of 4.8 against 7.2. The repository's original prompt, before chapter 20 of the
README changed it, was off by a median factor of 2.07 with 43% within a factor 2. The two questions were fixed, and
the combination defined, before the benchmark ran.

That holds for Qwen3.5-9B, Pipeline A's VLM. Qwen3-VL-8B, Pipeline B's, gave whole tools with plastic handles the
density of steel (a screwdriver at 604 g for 70 g), and with it the combination did no better than the direct
question (71% within a factor 2 both, worst factor 12.1 against 8.1). Score another VLM with
repro/mass_friction/bench_mass.py before relying on this.
"""

import logging
import re

import numpy as np
import trimesh

from simfoundry.utils.prompt_utils import parse_json_response, welded_volume

log = logging.getLogger(__name__)

#: A shell mesh filling less of its convex hull than this is a container or sheet reconstructed with both faces of its
#: wall (a cup, a plate); a fuller one is a closed package or bottle modelled by its outside alone. The same split
#: mass_prompt_volumes uses for a compact solid.
THIN_FILL = 0.4

#: Bulk density ranges (kg/m^3) by material family. They are a common-sense check on a mass, not a material database:
#: a value outside its range is worth a look, not a rejection, so the ranges are wide on purpose, and the values are
#: provisional, rounded from standard handbook densities (the per-line notes are typical values, not table entries).
#: The family is the first line with a keyword that names the material answer (lower-cased; see _density_family), so the
#: order matters: stainless steel is steel, plastic-coated rubber is rubber, and foam rubber is foam.
MATERIAL_DENSITY_FAMILIES = (
    ("steel", ("steel", "iron", "stainless"), (7000, 8100)),  # steel ~7850, cast iron ~7200
    ("aluminum", ("aluminum", "aluminium"), (2500, 2900)),  # ~2700
    ("copper", ("copper", "brass", "bronze"), (7400, 9000)),  # copper ~8960, brass ~8500
    ("glass", ("glass",), (2200, 2800)),  # soda-lime ~2500
    ("ceramic", ("ceramic", "porcelain", "stoneware", "clay"), (1800, 2900)),  # fired clay and porcelain ~2300-2500
    # Before rubber, so "foam rubber" and "plastic foam" are foam.
    ("foam", ("foam", "sponge"), (10, 400)),  # polystyrene and polyurethane foam, tens of kg/m^3
    ("rubber", ("rubber", "silicone"), (900, 1600)),  # natural rubber ~920, silicone ~1100
    ("plastic", ("plastic", "polymer", "acrylic", "nylon", "abs", "polypropylene", "polyethylene", "pvc", "pet"),
     (850, 2300)),  # PE ~950, PP ~900, PET ~1350, nylon ~1140, acrylic ~1180
    ("wood", ("wood", "bamboo"), (150, 1300)),  # dense hardwoods ~700-900, balsa ~150
    ("paper", ("paper", "cardboard"), (200, 1300)),  # paper ~800, cardboard ~650
    ("stone", ("stone", "concrete", "marble", "granite"), (1800, 3100)),  # granite and marble ~2700, concrete ~2300
    ("fabric", ("fabric", "cloth", "textile", "leather"), (100, 1600)),  # loose and fibrous; cotton fibre ~1500
    ("food", ("fruit", "vegetable", "tofu", "meat", "fish", "cheese", "bread", "dough", "food"), (250, 1300)),
    ("liquid", ("water", "juice", "oil", "liquid"), (700, 1450)),  # water 1000, oils ~900
)
#: Any solid whose family the material answer does not name: below 50 kg/m^3 or above 12000 (heavier than lead) is
#: doubtful for an ordinary object.
GLOBAL_DENSITY_RANGE = (50, 12000)


def prompt_typical_mass(obj_phrase, bounding_box_cm):
    box = " x ".join(f"{v:.1f}cm" for v in bounding_box_cm)
    return f"""Shown is a picture of a {obj_phrase}. Its bounding box is {box}. Estimate the mass of this object as a real object of this kind and size weighs. Think it through first: say what the object most likely is and what it is made of, whether it is empty or a closed package, can or bottle that holds contents (then most of its mass is the contents), and what such objects typically weigh. Also give the friction coefficient of its surface sliding on a wooden table.
    Answer, after your reasoning, in the following format: json
    {{
        "mass": <kg>,
        "friction": <value>
    }}
    """


def prompt_material_mass(obj_phrase, bounding_box_cm):
    box = " x ".join(f"{v:.1f}cm" for v in bounding_box_cm)
    return f"""Shown is a picture of a {obj_phrase}. Its bounding box is {box}. Its mass will be computed from its 3D model, so do not guess a mass: describe how it is built. Think it through first: say what the object most likely is and what it is made of, and whether it is
- "solid": material all the way through its outer surface (a fruit, a ball, a block, a tool, scissors, a marker or pen), with the average density of everything inside that surface, or
- "shell": a thin wall or sheet around empty space or contents (a cup, mug, bowl, plate, tray, box, can, bottle, pitcher), with the wall's thickness.
If it is a closed package, can or bottle that holds contents, give the contents' density and how full it is.
    Answer, after your reasoning, in the following format: json
    {{
        "material": "<main material>",
        "density": <g/cm^3>,
        "form": "solid" or "shell",
        "wall_thickness_mm": <mm for a shell, else null>,
        "contents_density": <g/cm^3, or null if nothing inside>,
        "contents_fill": <0 to 1, or null if nothing inside>,
        "friction": <coefficient of its surface sliding on a wooden table>
    }}
    """


def enclosed_volume(tm):
    """
    Volume (m^3) the mesh's surface encloses, or None when the surface has holes.

    The mesh's own volume when trimesh calls it closed, else welded_volume's (a textured mesh duplicates its UV-seam
    vertices), else, when merging coincident vertices leaves no boundary edge but trimesh still calls the surface
    open (non-manifold edges; every YCB scan that failed was such), the magnitude of the merged mesh's signed volume.

    Args:
        tm (trimesh.Trimesh): the scaled mesh

    Returns:
        float or None: the volume (m^3)
    """
    if tm.is_volume:
        return float(tm.volume)
    vol = welded_volume(tm)
    if vol is not None:
        return vol
    welded = trimesh.Trimesh(vertices=tm.vertices, faces=tm.faces, process=True)
    if len(trimesh.grouping.group_rows(welded.edges_sorted, require_count=1)):
        return None
    trimesh.repair.fix_winding(welded)
    return abs(float(welded.volume))


def mesh_geometry(tm, obb_extent):
    """
    What material_mass_g needs from a scaled mesh.

    Args:
        tm (trimesh.Trimesh): the scaled mesh
        obb_extent (array): its oriented bounding-box extents (m)

    Returns:
        dict: extent_cm, volume_cm3 (enclosed, or None), area_cm2, hull_cm3, and fill (volume / hull volume, or None)
    """
    vol = enclosed_volume(tm)
    hull = float(tm.convex_hull.volume)
    return {"extent_cm": [float(x) * 100 for x in obb_extent], "volume_cm3": None if vol is None else vol * 1e6,
            "area_cm2": float(tm.area) * 1e4, "hull_cm3": hull * 1e6, "fill": None if vol is None else vol / hull}


def _num(value):
    """A number out of a VLM's JSON field: 0.9, "0.9", "0.9-1.2" (the mean) or "95%" (a fraction); else None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value))]
    if not numbers:
        return None
    mean = sum(numbers[:2]) / len(numbers[:2])
    return mean / 100 if "%" in str(value) else mean


def material_mass_g(answer, geometry):
    """
    Mass (g) from the material answer and the mesh.

    A solid weighs its density times the volume its surface encloses (65% of the box when the surface has holes,
    stage 11's fallback). A shell weighs its wall, density times thickness times the wall's area: half the mesh's area
    for a mesh that fills less than THIN_FILL of its hull (both faces of the wall reconstructed), the whole area for a
    fuller one (a closed package or bottle modelled by its outside). Contents fill the volume the surface encloses (the
    hull, for an open container).

    Args:
        answer (dict): the parsed answer to prompt_material_mass
        geometry (dict): mesh_geometry's output

    Returns:
        float or None: the mass (g), or None when the answer lacks a density
    """
    rho = _num(answer.get("density"))
    if not rho or rho <= 0:
        return None
    vol = geometry["volume_cm3"] if geometry["volume_cm3"] is not None else 0.65 * float(np.prod(geometry["extent_cm"]))
    if not str(answer.get("form", "solid")).lower().startswith("shell"):
        return rho * vol
    t_cm = (_num(answer.get("wall_thickness_mm")) or 1.0) / 10
    thin = geometry["fill"] is not None and geometry["fill"] < THIN_FILL
    wall = rho * t_cm * (geometry["area_cm2"] / 2 if thin else geometry["area_cm2"])
    cd, cf = _num(answer.get("contents_density")), _num(answer.get("contents_fill"))
    if cf is not None and cf > 1:
        cf = cf / 100
    contents = cd * cf * (geometry["hull_cm3"] if thin else vol) if cd and cf else 0.0
    return wall + contents


def combine_masses(typical_g, material_g):
    """The material mass held within a factor 2 of the typical one; whichever exists when the other does not."""
    if typical_g and material_g:
        return min(max(material_g, typical_g / 2), typical_g * 2)
    return material_g or typical_g


def _names_material(keyword, text):
    """Whether a keyword names the material: it starts a word, and three letters or fewer make a whole word."""
    pattern = r"\b" + re.escape(keyword) + (r"\b" if len(keyword) <= 3 else "")
    return re.search(pattern, text) is not None


def _density_family(material):
    """
    The first family in MATERIAL_DENSITY_FAMILIES that the material answer names.

    The answer is lower-cased, and a keyword must start a word, so "wooden" is wood and "steelwork" is steel. A keyword
    of three letters or fewer must be a whole word, so "absorbent" is not abs and "PET bottle" is pet. A bare substring
    would misfire: "carpet" would be plastic through "pet", "foil" liquid through "oil".

    Args:
        material (dict or None): the material answer (prompt_material_mass)

    Returns:
        2-tuple: (family name or None, (lo, hi) range in kg/m^3); the global range when no family matches
    """
    text = str((material or {}).get("material") or "").lower()
    for family, keywords, bounds in MATERIAL_DENSITY_FAMILIES:
        if any(_names_material(keyword, text) for keyword in keywords):
            return family, bounds
    return None, GLOBAL_DENSITY_RANGE


def mass_flags(mass_kg, typical_g, material_g, material, geometry):
    """
    Flags on the mass stage 11 writes, for review. The mass itself is never changed.

    Two checks. The clamp: combine_masses holds the material mass within a factor 2 of the typical one, so a mass
    pinned at that edge means the two questions disagreed by more than that: "low" when the material mass was under
    half the typical one (the mass was raised), "high" when it was over twice (the mass was lowered). The density: the
    mass over the mesh's enclosed volume should fall in the bulk density range of the material's family. That range is
    provisional and deliberately wide (MATERIAL_DENSITY_FAMILIES). A shell's mass over its enclosed volume is not a
    material density, since most of that volume is empty, and an open mesh has no enclosed volume: neither is checked
    (density_checked False, density_outside_range None). A zero volume counts as no volume, so a degenerate mesh cannot
    divide by zero. An unknown material is checked against GLOBAL_DENSITY_RANGE.

    Args:
        mass_kg (float): the mass written to the scene (kg)
        typical_g (float or None): the typical-mass answer (g)
        material_g (float or None): the material mass (g)
        material (dict or None): the material answer (prompt_material_mass)
        geometry (dict): mesh_geometry's output

    Returns:
        dict: clamped ("low", "high" or None); density_kg_m3 (the mass over the enclosed volume, or None without one);
        density_family (the family name or None); density_range_kg_m3 ([lo, hi], the range applied); density_checked
        (bool); density_outside_range (bool, or None when not checked); flags (a list of "clamped_low", "clamped_high",
        "density_low", "density_high", in that order). The values are plain Python types, so json.dumps accepts them.
    """
    clamped = None
    if typical_g and material_g:  # combine_masses clamps only when both exist, so a flag means the mass was held
        if material_g < typical_g / 2:
            clamped = "low"
        elif material_g > typical_g * 2:
            clamped = "high"
    family, (lo, hi) = _density_family(material)
    shell = str((material or {}).get("form", "solid")).lower().startswith("shell")
    volume_cm3 = geometry["volume_cm3"]
    density = None if not volume_cm3 else float(mass_kg) * 1e6 / float(volume_cm3)  # kg/cm^3 times cm^3 per m^3
    checked = not shell and bool(volume_cm3)
    flags = [f"clamped_{clamped}"] if clamped else []
    outside = None
    if checked:
        outside = density < lo or density > hi
        if density < lo:
            flags.append("density_low")
        elif density > hi:
            flags.append("density_high")
    return {"clamped": clamped, "density_kg_m3": density, "density_family": family,
            "density_range_kg_m3": [lo, hi], "density_checked": checked, "density_outside_range": outside,
            "flags": flags}


def ask_json(vlm, prompt, image_path, print_results=False):
    """
    One deterministic VLM question and its parsed JSON answer.

    Returns:
        2-tuple: (answer dict or None, error string or None). An answer cut off at the token limit or with no JSON
        in it is an error here, not an exception: the other question may still give a mass.
    """
    result = vlm(prompt=prompt, image_paths=image_path, temperature=0, top_p=0, seed=0, print_results=print_results)
    try:
        answer = parse_json_response(vlm.get_result_text(result))
    except Exception as exc:  # noqa: BLE001 - reported, and the caller falls back
        return None, repr(exc)
    if isinstance(answer, list):  # Qwen3.5 sometimes wraps its one answer in a JSON array
        answer = next((a for a in answer if isinstance(a, dict)), None)
    if not isinstance(answer, dict):
        return None, "no JSON object in the answer"
    return answer, None


def estimate_rigid_mass(vlm, obj_phrase, image_path, tm, obb_extent, print_results=False):
    """
    Mass (kg) and friction of a rigid object from the typical and material questions.

    Args:
        vlm: the stage's VLM client (simfoundry.models.vlm.Gemini)
        obj_phrase (str): the object's description from stage 5
        image_path (str): its stage 6 photo
        tm (trimesh.Trimesh): its mesh at stage 11's scale
        obb_extent (array): the mesh's oriented bounding-box extents (m)
        print_results (bool): print each answer

    Returns:
        dict: mass_kg (None when neither question gave a mass), friction (the material answer's, else the typical
        answer's, else None), typical_g, material_g, the material answer, the geometry and any errors
    """
    geometry = mesh_geometry(tm, obb_extent)
    bbox_cm = np.asarray(obb_extent) * 100
    typical, typical_err = ask_json(vlm, prompt_typical_mass(obj_phrase, bbox_cm), image_path, print_results)
    material, material_err = ask_json(vlm, prompt_material_mass(obj_phrase, bbox_cm), image_path, print_results)
    typical_g = None if typical is None or _num(typical.get("mass")) is None else _num(typical.get("mass")) * 1000
    try:
        material_g = None if material is None else material_mass_g(material, geometry)
    except Exception as exc:  # noqa: BLE001 - a malformed answer must not stop the stage; typical_g still counts
        material_g, material_err = None, repr(exc)
    mass_g = combine_masses(typical_g, material_g)
    friction = next((f for f in (_num((material or {}).get("friction")), _num((typical or {}).get("friction")))
                     if f is not None and f > 0), None)
    errors = {k: v for k, v in (("typical", typical_err), ("material", material_err)) if v}
    if errors:
        log.warning(f"Mass of {obj_phrase}: {errors}")
    return {"mass_kg": None if mass_g is None else mass_g / 1000, "friction": friction, "typical_g": typical_g,
            "material_g": material_g, "material": material, "geometry": geometry, "errors": errors}
