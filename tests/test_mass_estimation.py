# SPDX-License-Identifier: Apache-2.0
"""simfoundry.utils.mass_estimation: the mass stage 11 assigns from two VLM answers and the mesh."""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from simfoundry.utils import mass_estimation as me  # noqa: E402

GEO_CLOSED = {"volume_cm3": 100.0, "area_cm2": 200.0, "hull_cm3": 110.0, "fill": 0.9, "extent_cm": [5.0, 5.0, 4.0]}
GEO_THIN = {"volume_cm3": 20.0, "area_cm2": 400.0, "hull_cm3": 500.0, "fill": 0.04, "extent_cm": [10.0, 10.0, 5.0]}


def test_solid_is_density_times_enclosed_volume():
    assert me.material_mass_g({"density": 0.9, "form": "solid"}, GEO_CLOSED) == pytest.approx(90.0)


def test_solid_with_holes_falls_back_to_box_share():
    geo = dict(GEO_CLOSED, volume_cm3=None, fill=None)
    assert me.material_mass_g({"density": 1.0, "form": "solid"}, geo) == pytest.approx(0.65 * 100.0)


def test_thin_shell_counts_one_face_of_the_wall():
    # A cup or plate reconstructed with both faces of its wall: half the mesh area is the wall.
    ans = {"density": 0.8, "form": "shell", "wall_thickness_mm": 0.5}
    assert me.material_mass_g(ans, GEO_THIN) == pytest.approx(0.8 * 0.05 * 200.0)


def test_closed_package_counts_its_surface_and_contents():
    ans = {"density": 7.8, "form": "shell", "wall_thickness_mm": 0.2, "contents_density": 1.0, "contents_fill": 0.9}
    assert me.material_mass_g(ans, GEO_CLOSED) == pytest.approx(7.8 * 0.02 * 200.0 + 0.9 * 100.0)


def test_numbers_given_as_text():
    ans = {"density": "0.9-1.1", "form": "shell", "wall_thickness_mm": "2", "contents_density": 1.0,
           "contents_fill": "95%"}
    assert me.material_mass_g(ans, GEO_CLOSED) == pytest.approx(1.0 * 0.2 * 200.0 + 0.95 * 100.0)


def test_no_density_gives_no_mass():
    assert me.material_mass_g({"form": "solid"}, GEO_CLOSED) is None


@pytest.mark.parametrize("typical, material, expected", [
    (100.0, 150.0, 150.0),   # within a factor 2: the material mass
    (100.0, 700.0, 200.0),   # above: held at twice the typical one
    (100.0, 20.0, 50.0),     # below: held at half of it
    (None, 30.0, 30.0),      # one answer missing: the other
    (40.0, None, 40.0),
    (None, None, None),
])
def test_combine_masses(typical, material, expected):
    assert me.combine_masses(typical, material) == expected


class _VLM:
    def __init__(self, text):
        self.text = text

    def __call__(self, **kwargs):
        return None

    def get_result_text(self, result):
        return self.text


def test_ask_json_unwraps_an_array_and_reports_no_json():
    assert me.ask_json(_VLM('```json\n[{"mass": 0.1, "friction": 0.4}]\n```'), "p", "img") == \
        ({"mass": 0.1, "friction": 0.4}, None)
    answer, err = me.ask_json(_VLM("no json here"), "p", "img")
    assert answer is None and err


def test_enclosed_volume_merges_seam_vertices():
    box = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
    # Every face with vertices of its own, as a textured mesh keeps them along its UV seams.
    tri = box.vertices[box.faces]
    split = trimesh.Trimesh(vertices=tri.reshape(-1, 3), faces=np.arange(len(tri) * 3).reshape(-1, 3), process=False)
    assert not split.is_volume
    assert me.enclosed_volume(split) == pytest.approx(0.1 * 0.2 * 0.3, rel=1e-6)
