# -*- coding: utf-8 -*-

import numpy as np

def is_empty(t):
    """
    Tests whether input variable is empty or not.

    Parameters
    ----------
    t: input variable

    Returns
    -------
    is_empty: True if t is empty, False if t is not empty.
    """

    if t.any:
        # t is not empty
        return False
    else:
        # t is empty
        return True
    
def change_angle_interval(angle_deg_tab):
    """
    Returns the input angle (in degrees) given in the interval [-180;180] in
    the corresponding angle in the interval [0;360], and
    """
    phasor = np.exp(1j*(angle_deg_tab*np.pi/180.-np.pi))
    phasor_phase = np.arctan2(np.imag(phasor),np.real(phasor))
    angle_tab = 180. + phasor_phase*180./np.pi

    return angle_tab