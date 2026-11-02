# SPDX-License-Identifier: Apache-2.0
"""Material presets for the MPM cutting solver.

The elastic constants of tofu come from mechanical tests on retail tofu. Its fracture
energy does not: no measured J/m^2 value for tofu was found, so the presets use an
order-of-magnitude estimate for soft food gels and say so. Treat ``fracture_energy`` as
the first parameter to calibrate against a real cut.

Tofu is strongly viscoelastic, so the presets carry a Maxwell branch: ``youngs_modulus`` is
the stiffness under quick loading and ``relaxed_modulus`` what is left after a few
``relaxation_time``s. Boes et al. tested in confined compression, whose slope is the P-wave
modulus lambda + 2 mu, not E: their fast peak for firm tofu (18.9 kPa at 30 %, i.e. about
63 kPa) is E ~ 29 kPa at nu = 0.4, in line with St. Pierre et al.'s 26.4 kPa, while their
equilibrium stress is 7.7 times lower. The presets take the measured modulus as the quick one
and a more cautious factor of 4 for the relaxed one; the 2 s relaxation time is an estimate.

Sources:
    St. Pierre et al., "The mechanical and sensory signature of plant-based and animal
    meat", npj Science of Food 8, 94 (2024): firm 26.4 +/- 4.0 kPa and extra-firm
    27.5 +/- 5.6 kPa stiffness from tension, compression and shear; neo-Hookean shear
    modulus about 9 kPa; extra-firm fails in tension slightly above 10 % stretch.
    Boes, Simon, Holthusen, Kuhl, "The mechanics and physics of tofu", J. Mech. Phys.
    Solids 213 (2026), bioRxiv 10.64898/2025.12.10.693552: peak compressive stress at
    30 % compression of 3.74 (silken), 18.91 (firm) and 42.84 kPa (extra-firm); fluid
    volume fraction 0.92 / 0.87 / 0.86; strongly viscoelastic.

The tensile strength is the stress at that 10 % stretch under quick loading, E / 10: measured
for extra-firm tofu only, and carried over to the other presets as an estimate.

Vegetables (carrot, cucumber, zucchini, onion) come from the materials report (final_veg/materials/report_ko.md,
sections 4-1 and 4-2; values.csv). Their names, synonyms, tissue kinds (``TISSUE_KINDS``) and ranges are a
손으로 쓴 목록(사용자가 사전 지식으로 허용, 2026-11-02): a hand-written list. The project does not allow
hand-written category lists otherwise; the user allowed these as prior knowledge on 2026-11-02. Where a vegetable
has no value for a property, the value is a stand-in
from its tissue kind, and the note says which one (``tissue_material``).
"""

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SoftSolidMaterial:
    """Isotropic soft solid cut by the MPM solver. SI units throughout.

    Attributes:
        name: Preset name.
        youngs_modulus: Small-strain Young's modulus E [Pa].
        poisson_ratio: Poisson's ratio. Below the incompressible 0.5 on purpose: the
            explicit time step shrinks with the bulk modulus, and tofu is a porous gel
            that drains water under slow loading.
        density: Mass density [kg/m^3].
        viscosity: Kelvin-Voigt viscosity [Pa s]. Mostly numerical damping; tofu is
            strongly viscoelastic, but with relaxation times this solver does not model.
        fracture_energy: Critical energy release rate G_c [J/m^2]. The crack under the
            knife runs as far as the work of the knife's push pays for at G_c per unit of
            new area, so a steady cut takes a push of G_c times the length of edge in the
            food (less when the edge also slides along itself).
        knife_friction: Coulomb friction between blade and material.
        board_friction: Coulomb friction between material and cutting board.
        contact_friction: Coulomb friction between two cut pieces.
        relaxed_modulus: Young's modulus left after full relaxation [Pa]; 0 makes the
            material purely elastic with ``youngs_modulus``.
        relaxation_time: Relaxation time of the Maxwell branch [s].
        initiation_length: Depth of crack whose fracture energy the knife must push into the
            food before the food first gives way under a stroke [m]. It sets how far a straight
            press dents the food first; slicing divides it like G_c. An estimate.
        tensile_strength: Opening stress at which the uncut ligament below a partial cut tears
            [Pa]; 0 never tears it, and cracks then grow only under the knife.
    """

    name: str
    youngs_modulus: float
    poisson_ratio: float
    density: float
    viscosity: float
    fracture_energy: float
    knife_friction: float = 0.3
    board_friction: float = 0.5
    contact_friction: float = 0.3
    relaxed_modulus: float = 0.0
    relaxation_time: float = 0.0
    initiation_length: float = 2.0e-3
    tensile_strength: float = 0.0

    @property
    def shear_modulus(self) -> float:
        """Lame mu under quick loading [Pa]."""
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def viscoelastic(self) -> bool:
        return 0.0 < self.relaxed_modulus < self.youngs_modulus and self.relaxation_time > 0.0

    @property
    def equilibrium_shear_modulus(self) -> float:
        """Shear modulus of the elastic (equilibrium) branch [Pa]."""
        if not self.viscoelastic:
            return self.shear_modulus
        return self.relaxed_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def maxwell_shear_modulus(self) -> float:
        """Shear modulus of the Maxwell branch, which relaxes [Pa]; 0 if elastic."""
        return self.shear_modulus - self.equilibrium_shear_modulus

    @property
    def critical_opening(self) -> float:
        """Crack opening at which a cohesive crack with a linear softening law carries nothing [m].

        2 G_c / tensile_strength: the law falls from the strength to zero over this opening and
        takes G_c of work per unit area to do it. 0 if the material never tears.
        """
        if self.tensile_strength <= 0.0:
            return 0.0
        return 2.0 * self.fracture_energy / self.tensile_strength

    @property
    def lame_lambda(self) -> float:
        """Lame lambda [Pa]."""
        nu = self.poisson_ratio
        return self.youngs_modulus * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @property
    def wave_speed(self) -> float:
        """Dilatational wave speed sqrt((lambda + 2 mu) / rho) [m/s], which bounds the time step."""
        return math.sqrt((self.lame_lambda + 2.0 * self.shear_modulus) / self.density)


# Density: tofu is 86-92 % water by volume and its soy solids are denser than water, and
# it sinks in water; 1060 kg/m^3 is an estimate, not a measurement.
FIRM_TOFU = SoftSolidMaterial(
    name="firm_tofu",
    youngs_modulus=26.4e3,
    poisson_ratio=0.40,
    density=1060.0,
    viscosity=2.0,
    fracture_energy=3.0,
    relaxed_modulus=26.4e3 / 4.0,
    relaxation_time=2.0,
    tensile_strength=26.4e3 / 10.0,
)

# Silken tofu scaled from firm by the ratio of the peak compressive stresses in Boes et al.
# (3.74 / 18.91); its fracture energy is lowered by the same reasoning, also an estimate.
SILKEN_TOFU = SoftSolidMaterial(
    name="silken_tofu",
    youngs_modulus=26.4e3 * 3.74 / 18.91,
    poisson_ratio=0.40,
    density=1040.0,
    viscosity=1.0,
    fracture_energy=1.0,
    relaxed_modulus=26.4e3 * 3.74 / 18.91 / 3.0,
    relaxation_time=2.0,
    tensile_strength=26.4e3 * 3.74 / 18.91 / 10.0,
)

# St. Pierre et al. measured almost the same stiffness for firm and extra-firm tofu, while
# Boes et al. measured more than twice the stress; the preset keeps the measured modulus and
# a higher (estimated) fracture energy, which is what a cook notices.
EXTRA_FIRM_TOFU = SoftSolidMaterial(
    name="extra_firm_tofu",
    youngs_modulus=27.5e3,
    poisson_ratio=0.40,
    density=1070.0,
    viscosity=2.0,
    fracture_energy=6.0,
    relaxed_modulus=27.5e3 / 4.0,
    relaxation_time=2.0,
    tensile_strength=27.5e3 / 10.0,
)

# Firm tofu pressed under a weight to drain it, as in the HDMD clip (45 minutes). Boes et al.
# found stress rising more than tenfold for six percent less water; four times firm tofu is an
# estimate, not a measurement.
PRESSED_TOFU = SoftSolidMaterial(
    name="pressed_tofu",
    youngs_modulus=4.0 * 26.4e3,
    poisson_ratio=0.40,
    density=1080.0,
    viscosity=2.0,
    fracture_energy=6.0,
    relaxed_modulus=26.4e3,
    relaxation_time=2.0,
    tensile_strength=4.0 * 26.4e3 / 10.0,
)

# Vegetables, from the materials report (final_veg/materials/report_ko.md, 4-1 candidates and 4-2 rules; values.csv).
# Hand-written list: the names, synonyms, tissue kinds and ranges in this module are hand-written lists that the user
# allowed as prior knowledge on 2026-11-02 (the project's rule against hand-written category lists).
#
# Every vegetable below shares the common rule: viscosity 2.0 Pa s, elastic (relaxed modulus 0), tensile strength E/10
# (the tofu rule; no value for vegetables), initiation length 2 mm (the tofu value; R9 decides it), board friction 0.5 and
# contact friction 0.3 (the solver's defaults; nothing measured).


def _vegetable(name, youngs_modulus, poisson_ratio, density, fracture_energy, knife_friction):
    """A vegetable's material under the common rule above, from the values the caller gives (a preset or a draw)."""
    return SoftSolidMaterial(
        name=name,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
        density=density,
        viscosity=2.0,
        fracture_energy=fracture_energy,
        knife_friction=knife_friction,
        board_friction=0.5,
        contact_friction=0.3,
        relaxed_modulus=0.0,
        relaxation_time=0.0,
        initiation_length=2.0e-3,
        tensile_strength=youngs_modulus / 10.0,
    )


# Carrot: E is the geometric mean of three sources (Williams 2005, Agrawal 1997, Gołacki 2007). G_c is the geometric mean of
# the scissor tests (344-440). Friction is a substitute: the geometric mean of cucumber's 0.36 and onion's 0.477 is 0.414,
# and the table gives it to two decimals as 0.41. Density is the arithmetic mean of 1003-1040 (1021.5, set to 1020).
CARROT = _vegetable("carrot", 6.0e6, 0.35, 1020.0, 389.0, 0.41)
# Cucumber: E 0.85-1.4 MPa (Jahangiri 2016, Tanaka 2018), the default 1.1 MPa. G_c 125 is the model constant of Jamdagni
# and Jia (the knife-energy rule gives 119); its origin is not stated, so it is not an independent check. Friction is the mean
# of the stainless-steel tilt tests (0.34-0.38).
CUCUMBER = _vegetable("cucumber", 1.1e6, 0.35, 995.0, 125.0, 0.36)
# Zucchini: E 0.38-5.3 MPa, and the two values disagree, so E is low-confidence. Friction is a substitute: cucumber's 0.36.
ZUCCHINI = _vegetable("zucchini", 1.8e6, 0.35, 960.0, 705.0, 0.36)
# Onion: E from the scale tissue only (1.4 and 2.2 MPa; the whole-bulb Hertz value includes the air between scales and is
# left out). G_c is the geometric mean of the robot papers' 302 and 675. Friction 0.477 is the cross-section value of the
# robot paper.
ONION = _vegetable("onion", 1.8e6, 0.35, 1075.0, 451.0, 0.477)

# The four vegetables' ranges, from which sample_material draws (the report's ranges). E and G_c are drawn log-uniformly;
# nu and density uniformly, over the same ranges (research 2's rule, 2026-11-02, the one tissue kinds use too).
VEGETABLE_RANGES = {
    "carrot": dict(youngs_modulus=(4.6e6, 9.9e6), fracture_energy=(344.0, 440.0), density=(1003.0, 1040.0),
                   poisson_ratio=(0.24, 0.45)),
    "cucumber": dict(youngs_modulus=(0.85e6, 1.4e6), fracture_energy=(60.0, 238.0), density=(980.0, 1010.0),
                     poisson_ratio=(0.24, 0.45)),
    "zucchini": dict(youngs_modulus=(0.38e6, 5.3e6), fracture_energy=(622.0, 800.0), density=(910.0, 1010.0),
                     poisson_ratio=(0.24, 0.45)),
    "onion": dict(youngs_modulus=(1.4e6, 2.2e6), fracture_energy=(302.0, 675.0), density=(1040.0, 1110.0),
                  poisson_ratio=(0.24, 0.45)),
}

VEGETABLES = {m.name: m for m in (CARROT, CUCUMBER, ZUCCHINI, ONION)}

PRESETS = {m.name: m for m in (FIRM_TOFU, SILKEN_TOFU, EXTRA_FIRM_TOFU, PRESSED_TOFU, CARROT, CUCUMBER, ZUCCHINI, ONION)}

# Tissue kinds: the stand-in for a vegetable the table does not have (report 4-2 rule 3). A kind's default is the geometric
# mean of its members' E, G_c and knife friction, and the arithmetic mean of their density and nu.
TISSUE_KINDS = {
    "root_tuber": ("carrot",),  # the report also names potato and radish; only carrot has values in the table
    "watery_fruit": ("cucumber", "zucchini"),
    "bulb_scale": ("onion",),
    "gel": ("firm_tofu",),
}
# The spread of a stand-in's E and G_c, as x and /: the report's rule 4, from Singh 2016's same-kind differences (2.1-2.6x).
TISSUE_SPREAD = 2.6
# The knife friction range of a vegetable and of a tissue kind: the report's friction values taken together (4-1, 0.30-0.60).
# Both draws read this one constant.
KNIFE_FRICTION_RANGE = (0.30, 0.60)
# A stand-in's other properties, drawn uniformly over the whole table's range: no value exists for the kind itself, so the
# range is the vegetables' (report rule 3; nu 0.24-0.45 is the onion's measured range, density 910-1110 the union of the
# four vegetables' ranges).
TISSUE_OTHER_RANGES = dict(poisson_ratio=(0.24, 0.45), density=(910.0, 1110.0), knife_friction=KNIFE_FRICTION_RANGE)
# Gel is firm tofu, not a vegetable, so the vegetables' ranges do not apply: nu 0.40 and friction 0.30 are fixed (the
# tofu presets' values), and density is the report's tofu range (Kong 1982: 1034-1083).
GEL_OTHER_RANGES = dict(poisson_ratio=(0.40, 0.40), density=(1034.0, 1083.0), knife_friction=(0.30, 0.30))

_MEMBERS = {"firm_tofu": FIRM_TOFU, **VEGETABLES}
# Names that find a preset, lower case. Hand-written list (see the top of this section): the report's synonyms, the common
# names of the vegetables, and the tofu's names. A name not here finds nothing; the caller then looks at the tissue kinds.
_NAMES = {
    "carrot": CARROT, "daucus carota": CARROT, "당근": CARROT,
    "cucumber": CUCUMBER, "cucumis sativus": CUCUMBER, "오이": CUCUMBER,
    "zucchini": ZUCCHINI, "courgette": ZUCCHINI, "cucurbita pepo": ZUCCHINI, "애호박": ZUCCHINI,
    "onion": ONION, "allium cepa": ONION, "양파": ONION,
    "tofu": FIRM_TOFU, "bean curd": FIRM_TOFU, "두부": FIRM_TOFU,
}


def material_for_name(name):
    """The preset a name stands for, ignoring case, or None when the name is not in the hand-written list.

    A None is not a guess: the caller turns to the tissue kind (``TISSUE_KINDS``) instead.

    Args:
        name: A name such as "Carrot", "Daucus carota", "courgette", "두부".

    Returns:
        The SoftSolidMaterial, or None.
    """
    return _NAMES.get(name.lower())


def _geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _mean(values):
    return sum(values) / len(values)


def _members(kind):
    return [_MEMBERS[m] for m in TISSUE_KINDS[kind]]


def _relaxation(members):
    """The viscoelastic part of a tissue kind: the members' mean ratio of relaxed to fast modulus, and their mean relaxation
    time. The vegetables are elastic (both 0), so only gel carries relaxation, and it keeps firm tofu's values."""
    ratio = _mean([m.relaxed_modulus / m.youngs_modulus for m in members])
    return ratio, _mean([m.relaxation_time for m in members])


def tissue_material(kind):
    """The default material of a tissue kind, and the note that says what stands in for it.

    Args:
        kind: A key of ``TISSUE_KINDS``.

    Returns:
        (material, note): the material, with its name set to the kind, and the note "대신함(kind, members)".
    """
    members = _members(kind)
    youngs = _geomean([m.youngs_modulus for m in members])
    ratio, time = _relaxation(members)
    material = replace(
        _vegetable(
            kind,
            youngs,
            _mean([m.poisson_ratio for m in members]),
            _mean([m.density for m in members]),
            _geomean([m.fracture_energy for m in members]),
            _geomean([m.knife_friction for m in members]),
        ),
        relaxed_modulus=ratio * youngs,
        relaxation_time=time,
    )
    return material, f"대신함({kind}, {', '.join(TISSUE_KINDS[kind])})"


def tissue_range(kind):
    """The range of each property a tissue kind is drawn from: (low, high) per property.

    E and G_c are the default times and divided by ``TISSUE_SPREAD``; the other properties are ``TISSUE_OTHER_RANGES``,
    or ``GEL_OTHER_RANGES`` for gel.

    Args:
        kind: A key of ``TISSUE_KINDS``.

    Returns:
        A dict from the material's field name to (low, high).
    """
    material, _ = tissue_material(kind)
    ranges = dict(GEL_OTHER_RANGES if kind == "gel" else TISSUE_OTHER_RANGES)
    ranges["youngs_modulus"] = (material.youngs_modulus / TISSUE_SPREAD, material.youngs_modulus * TISSUE_SPREAD)
    ranges["fracture_energy"] = (material.fracture_energy / TISSUE_SPREAD, material.fracture_energy * TISSUE_SPREAD)
    return ranges


def _log_uniform(rng, bounds):
    low, high = bounds
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_material(name, rng):
    """A vegetable drawn from its ranges, with the common rule applied again.

    E and G_c are drawn log-uniformly from ``VEGETABLE_RANGES[name]``, because they vary several times over their ranges.
    nu and density are drawn uniformly over the same ranges, and the knife friction uniformly over ``KNIFE_FRICTION_RANGE``:
    the rule research 2 set on 2026-11-02, the same one a tissue kind's draw follows.

    Args:
        name: A key of ``VEGETABLE_RANGES``.
        rng: A numpy random Generator; the same seed gives the same material.

    Returns:
        A SoftSolidMaterial named ``name``.
    """
    ranges = VEGETABLE_RANGES[name]
    return _vegetable(
        name,
        _log_uniform(rng, ranges["youngs_modulus"]),
        rng.uniform(*ranges["poisson_ratio"]),
        rng.uniform(*ranges["density"]),
        _log_uniform(rng, ranges["fracture_energy"]),
        rng.uniform(*KNIFE_FRICTION_RANGE),
    )


def sample_tissue(kind, rng):
    """A tissue kind drawn from its ranges, with the common rule applied again.

    E and G_c are drawn log-uniformly and the other properties uniformly, from ``tissue_range(kind)``. The gel's nu and
    friction are fixed.

    Args:
        kind: A key of ``TISSUE_KINDS``.
        rng: A numpy random Generator; the same seed gives the same material.

    Returns:
        (material, note) as ``tissue_material`` returns them.
    """
    ranges = tissue_range(kind)
    ratio, time = _relaxation(_members(kind))
    youngs = _log_uniform(rng, ranges["youngs_modulus"])
    material = replace(
        _vegetable(
            kind,
            youngs,
            rng.uniform(*ranges["poisson_ratio"]),
            rng.uniform(*ranges["density"]),
            _log_uniform(rng, ranges["fracture_energy"]),
            rng.uniform(*ranges["knife_friction"]),
        ),
        relaxed_modulus=ratio * youngs,
        relaxation_time=time,
    )
    return material, tissue_material(kind)[1]
