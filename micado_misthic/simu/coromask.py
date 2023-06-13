# -*- coding: utf-8 -*-

import numpy as np

from ..imgproc import get_frame_center
from ..imgproc import get_circle_mask


def get_vortex_phase_mask(nyx, cyx=None, charge=2., offset_rad=0., sign=1.,
                          centering='FFTSTYLE', center_mask_diam = 1):
    """
    Returns a complex phase mask for a vortex coronagraph.

    Parameters
    ----------
    nyx: tuple of int
        in the form (ny,nx), shape of the desired mask.
    cyx: tuple of float, optional
        in the form (cy,cx), center of the desired vortex mask.
    charge: int
         toppological charge of the vortex mask
    offset_rad: float
        phase offset, in radian.
    sign: signed int
        sign of the phase ram, can be +1 or -1.
    centering: string
        centering style when cyx is not given. Can be:
        'FFTSTYLE': center located in the center of a pixel
        'SYMMETRIC': center located between 2 pixels
    center_mask_diam: float
        diameter of an optional occulting mask at the center.

    Returns
    -------
    vortex: 2D-array
        complex 2D-array of shape nyx, with amplitude 1, except if an occulting
        mask diameter is defined to mask out the center, and with a vortex
        phase ramp defined by the input parameters.

    """

    if cyx is None:
        cyx = get_frame_center(nyx, centering=centering)

    gridy, gridx = np.indices(nyx)
    gridx = gridx - cyx[1]
    gridy = gridy - cyx[0]

    if center_mask_diam > 0 :
        center_mask = get_circle_mask(nyx, center_mask_diam/2., centering=centering)
    else :
        center_mask = 0.

    vortex = (1.-center_mask) * np.exp(1.j*(sign*np.arctan2(gridy, gridx)*charge+offset_rad))

    return vortex

def get_fqpm_phase_mask(nyx, cyx = None, centering='FFTSTYLE',
                        center_mask_diam = 0):
    """
    Returns a complex phase mask for a four quadrant phase mask.

    Parameters
    ----------
    nyx: tuple of int
        in the form (ny,nx), shape of the desired mask.
    cyx: tuple of float, optional
        in the form (cy,cx), center of the desired vortex mask.
    centering: string
        centering style when cyx is not given. Can be:
        'FFTSTYLE': center located in the center of a pixel
        'SYMMETRIC': center located between 2 pixels
    center_mask_diam: float
        diameter of an optional occulting mask at the center.

    Returns
    -------
    fqpm: 2D-array
        complex 2D-array of shape nyx, with amplitude defined by quadrant
        to 1 or -1 values (equivalent to a 0 and pi phase), except if an
        occulting mask diameter is defined to mask out the center.
    """
    if cyx is None:
        cyx = get_frame_center(nyx, centering=centering)

    gridy, gridx = np.indices(nyx)
    gridx = gridx - cyx[1]
    gridy = gridy - cyx[0]

    if center_mask_diam > 0 :
        center_mask = get_circle_mask(nyx, center_mask_diam/2., centering=centering)
    else :
        center_mask = 0.

    fqpm = np.zeros(nyx)
    pizone1 = np.where((gridx > 0) & (gridy < 0))
    pizone2 = np.where((gridx < 0) & (gridy > 0))
    nozone1 = np.where((gridx > 0) & (gridy > 0))
    nozone2 = np.where((gridx < 0) & (gridy < 0))
    fqpm[pizone1] = -1.
    fqpm[pizone2] = -1.
    fqpm[nozone1] = 1.
    fqpm[nozone2] = 1.

    fqpm = fqpm * (1. - center_mask)

    return fqpm
