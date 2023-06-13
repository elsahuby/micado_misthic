# -*- coding: utf-8 -*-

import numpy as np

from ..imgproc import get_circle_mask

def get_dsp(dim, slope=-2., high_freq_cut = None):
    """
    Generates a 2D-Power Spectrum Density of dimension dim*dim and given slope.
    """
    if (dim % 2) == 0 :
        dim2 = int(dim/2.)
    else :
        dim2 = int((dim-1.)/2.)

    x, y = (np.indices((dim, dim)) - dim2) / dim
    r = np.sqrt(x**2+y**2)

    r[dim2,dim2] = 1. # to avoid RuntimeWarning due to the Zero value
    dsp = r ** slope
    dsp[dim2,dim2] = 0.

    # Limiting the frequency range of the DSP
    if high_freq_cut is not None:
        dsp_mask = get_circle_mask((dim,dim), high_freq_cut)
    else :
        dsp_mask = 1.
    dsp = dsp * dsp_mask

    return dsp

def get_opd_screen(dim, opd_rms_nm, slope=-2.,
                    high_freq_cut = None, mask=None, pup_radius=None):
    """
    Draws a random OPD screen based on a given Power Spectrum Density.
    Adapted from IDL routine tirage_dsp.pro by P. Baudoz 2008.

    Parameters
    ----------
    dim: int
        width of the desired screen.
    opd_rms_nm: float
        OPD rms, in nm
    slope: int
        slope of DSP
    high_freq_cut = None
        cut-off frequency of the DSP
    mask: 2D-array with 0 or 1 values, optional
        2D-array defining the region where the rms should be computed.
    pup_radius: int, optional
        if mask is None, this parameter can be used to define the region
        where the rms should be computed, as a disk of this given radius.

    Returns
    -------
    opd_screen_micron: 2D-array
        OPD screen in microns, 2D-array of shape (dim,dim).
    """

    if (dim % 2) == 0 :
        dim2 = int(dim/2.)
    else :
        dim2 = int((dim-1.)/2.)

    x, y = ((np.indices((dim, dim))) - dim2) / dim
    r = np.sqrt(x**2+y**2)

    r[dim2,dim2] = 1. # to avoid RuntimeWarning due to the Zero value
    dsp = r ** slope
    dsp[dim2,dim2] = 0.

    # Limiting the frequency range of the DSP
    if high_freq_cut is not None:
        dsp_mask = get_circle_mask((dim,dim), high_freq_cut)
    else :
        dsp_mask = 1.
    dsp = dsp * dsp_mask

    # Drawing a random phase term
    random_phase = np.random.normal(0.,1.,(dim,dim)) * 2. * np.pi - np.pi
    # Drawing the corresponding random phase
    complex_phase = np.fft.ifft2(np.fft.fftshift(np.sqrt(dsp)*np.exp(1j * random_phase)))
    #opd_screen = np.abs(complex_phase) * np.sign(np.real(complex_phase))
    opd_screen = np.imag(complex_phase)
    opd_screen = np.real(complex_phase)

    # Normalization
    ## computing the RMS level within a given mask (or not)
    if mask is None:
        if pup_radius is not None:
            mask = get_circle_mask((dim,dim), pup_radius)
            opd_screen_rms = np.sqrt(np.mean((opd_screen[np.where(mask == 1)])**2.))
        else :
            opd_screen_rms = np.sqrt(np.mean((opd_screen)**2.))
    else :
        opd_screen_rms = np.sqrt(np.mean((opd_screen[np.where(mask == 1)])**2.))

    opd_screen_micron = opd_screen / opd_screen_rms * (opd_rms_nm*1e-3)

    return opd_screen_micron
