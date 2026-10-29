import numpy as np
import pytest

from figo.instances import fibroid_instances
from figo.labels import MYOMA, WALL
from figo.stage3 import FigoRules, _apply_rules, classify_case, contact_areas

from .phantoms import SHAPE, add_fibroid, along_x, uterus

ISO = (1.0, 1.0, 1.0)


def single_type(seg):
    _, rows = classify_case(seg, ISO)
    assert len(rows) == 1, rows
    return rows[0]


def test_contact_area_weights_anisotropic_faces():
    m = np.zeros((6, 6, 6), bool)
    m[1:5, 1:5, 1:5] = True
    cat = np.ones((6, 6, 6), np.uint8)  # all tissue
    areas = contact_areas(m, cat, (1.0, 1.0, 3.0))
    # 2*16 faces of 1x1 (perpendicular to the thick axis) + 4*16 faces of 1x3
    assert areas[1] == pytest.approx(32 * 1 + 64 * 3)
    assert areas[[0, 2, 3]].sum() == 0


def test_intramural_type4():
    seg = add_fibroid(uterus(), [50, 50, 68], 6)
    r = single_type(seg)
    assert r["figo"] == "4", r


def test_intramural_abutting_endometrium_type3():
    # sphere touches the cavity slab (z<=52) at a single tangent point only
    seg = add_fibroid(uterus(), [50, 50, 60], 7)
    r = single_type(seg)
    assert r["figo"] == "3", r


def test_intracavitary_type0():
    seg = add_fibroid(uterus(), [50, 50, 50], 4, cavity_pocket=7)
    r = single_type(seg)
    assert r["figo"] == "0", r


@pytest.mark.parametrize("offset,expected", [(-4, "5"), (3, "6")])
def test_subserosal_5_vs_6(offset, expected):
    seg = add_fibroid(uterus(), along_x(35 + offset), 8)
    r = single_type(seg)
    assert r["figo"] == expected, r


def test_pedunculated_subserosal_type7():
    seg = uterus()
    c = along_x(35 + 7)  # sphere spans x 86-98; uterus surface at x=85
    seg = add_fibroid(seg, c, 6)
    seg[84:87, 49:52, 49:52] = WALL  # thin stalk bridging the 1-voxel gap
    r = single_type(seg)
    assert r["figo"] == "7", r


def test_detached_is_type8():
    seg = uterus()
    seg[2:8, 2:8, 2:8] = MYOMA
    r = single_type(seg)
    assert r["figo"] == "8"


@pytest.mark.parametrize("f_cav,f_tis,f_out,expected", [
    (0.3, 0.7, 0.0, "2"),
    (0.6, 0.4, 0.0, "1"),
    (0.95, 0.05, 0.0, "0"),
    (0.3, 0.4, 0.3, "2-5"),
    (0.0, 0.95, 0.05, "5"),
    (0.0, 0.4, 0.6, "6"),
])
def test_rule_table(f_cav, f_tis, f_out, expected):
    figo, why, _ = _apply_rules(f_cav, f_tis, f_out, 0.0, 5.0 if f_cav == 0 else 0.0, FigoRules())
    assert figo == expected, why


def test_instances_anisotropic_connectivity():
    seg = np.zeros((20, 20, 6), np.uint8)
    seg[2:6, 2:6, 1] = MYOMA
    seg[6:10, 6:10, 2] = MYOMA  # touches the first blob only diagonally across slices
    inst, info = fibroid_instances(seg, (0.5, 0.5, 6.6), min_volume_mm3=0)
    assert len(info) == 2
    seg[5:7, 5:7, 2] = MYOMA  # now face-connected
    _, info = fibroid_instances(seg, (0.5, 0.5, 6.6), min_volume_mm3=0)
    assert len(info) == 1


def test_instances_min_volume():
    seg = np.zeros(SHAPE[:2] + (5,), np.uint8)
    seg[10:12, 10:12, 2] = MYOMA  # 4 voxels * 1 mm^3
    seg[30:50, 30:50, 1:4] = MYOMA
    _, info = fibroid_instances(seg, ISO, min_volume_mm3=10)
    assert len(info) == 1 and info[0]["n_voxels"] == 1200


def test_intramural_cut_is_configurable():
    # 45% of surface outside -> 55% intramural: type 5 at the default cut, type 6 if the cut moves to 0.6
    assert _apply_rules(0.0, 0.55, 0.45, 0.0, 10.0, FigoRules())[0] == "5"
    assert _apply_rules(0.0, 0.55, 0.45, 0.0, 10.0, FigoRules(intramural_cut=0.6))[0] == "6"


def test_sensitivity_counts_changed_calls():
    from figo.figo_eval import sensitivity

    def row(f_cav, f_tis, f_out, dist):
        return {"frac_cavity": f_cav, "frac_myometrium": f_tis, "frac_outside": f_out,
                "frac_cyst": 0.0, "dist_to_cavity_mm": dist}

    rows = [row(0.0, 1.0, 0.0, "inf"),      # type 4, far from everything: never changes
            row(0.0, 0.53, 0.47, "8.0")]     # type 5, 3 points from the 50% cut
    res = {(r["threshold"], r["value"]): r for r in sensitivity(rows, delta=0.05)}
    assert res[("intramural_cut", 0.55)]["n_changed"] == 1
    assert res[("intramural_cut", 0.55)]["n_coarse_changed"] == 0   # 5 -> 6 stays subserosal
    assert res[("intramural_cut", 0.45)]["n_changed"] == 0
    assert "min_volume_mm3" not in {r["threshold"] for r in res.values()}
