# SPDX-License-Identifier: Apache-2.0
"""Knife cutting in simulation: soft solids with an MPM solver, rigid objects by runtime cutting.

Soft solids (tofu): a multi-field MPM solver, its coupling to a knife held by a robot, and the fit of the solver's food to
a reconstructed scene; see README.md here.

Runtime cutting: a knife's blade replaces a rigid object by two pieces, cut by the blade's plane where it presses.

- ``geometry``: the cut itself (planes, mesh and collider clipping, the cut face, mass and motion); numpy, scipy,
  shapely; no USD or Isaac Sim.
- ``usd_io``: reads a cuttable asset and writes each piece as a new USD asset; needs ``pxr``.
- ``rule``: the OmniGibson transition rule; import it before the scene is built.
- ``knife_asset``: a procedural chef's knife whose blade the rule recognizes.

This package does not import ``rule`` itself: it imports OmniGibson, and the geometry is used without it.
See repro/README.md, section 15.2.
"""
