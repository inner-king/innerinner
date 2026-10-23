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
"""

import math
from dataclasses import dataclass


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
)

PRESETS = {m.name: m for m in (FIRM_TOFU, SILKEN_TOFU, EXTRA_FIRM_TOFU, PRESSED_TOFU)}
