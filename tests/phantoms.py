"""Synthetic uterus phantoms with fibroids of known FIGO type (1 mm isotropic)."""

import numpy as np

from figo.labels import CAVITY, MYOMA, WALL

SHAPE = (100, 100, 100)
CENTRE = np.array([50, 50, 50])
UTERUS_R = 35


def _ball(centre, r):
    g = np.ogrid[tuple(slice(0, s) for s in SHAPE)]
    return sum((gi - c) ** 2 for gi, c in zip(g, centre)) <= r**2


def uterus():
    seg = np.zeros(SHAPE, np.uint8)
    seg[_ball(CENTRE, UTERUS_R)] = WALL
    zz = np.arange(SHAPE[2])[None, None, :]
    xy = _ball(CENTRE, 15) & (np.abs(zz - 50) <= 2)
    seg[xy] = CAVITY
    return seg


def add_fibroid(seg, centre, r, cavity_pocket=None):
    if cavity_pocket:  # intracavitary fibroid sitting in a distended cavity
        seg[_ball(centre, cavity_pocket)] = CAVITY
    seg[_ball(centre, r)] = MYOMA
    return seg


def along_x(radius_from_centre):
    """A point on the +x axis at the given distance from the uterus centre."""
    return CENTRE + np.array([radius_from_centre, 0, 0])
