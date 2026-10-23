# SPDX-License-Identifier: Apache-2.0
"""Material presets for the MPM cutting solver.

The elastic constants of tofu come from mechanical tests on retail tofu. Its fracture
energy does not: no measured J/m^2 value for tofu was found, so the presets use an
order-of-magnitude estimate for soft food gels and say so. Treat ``fracture_energy`` as
the first parameter to calibrate against a real cut.

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
            knife advances where the strain energy density reaches G_c divided by the
            grid spacing (a crack-band model), so the cutting resistance scales with it.
        knife_friction: Coulomb friction between blade and material.
        board_friction: Coulomb friction between material and cutting board.
        contact_friction: Coulomb friction between two cut pieces.
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

    @property
    def shear_modulus(self) -> float:
        """Lame mu [Pa]."""
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))

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
)

PRESETS = {m.name: m for m in (FIRM_TOFU, SILKEN_TOFU, EXTRA_FIRM_TOFU)}
