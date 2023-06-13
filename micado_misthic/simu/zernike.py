# -*- coding: utf-8 -*-

import numpy as np
from poppy.zernike import zernike1

from ..imgproc import get_frame_center
from ..imgproc import get_circle_mask

def get_tilted_wavefront(pup_mask, pup_diameter, angle = 0.):
    """
    Returns a tilted wavefront corresponding to a 1 lambda/D shift.
    """
    nyx = pup_mask.shape

    cy, cx = get_frame_center(nyx)

    gridy, gridx = np.indices(nyx)
    gridx = (gridx - cx) / pup_diameter * 2. * np.pi
    gridy = (gridy - cy) / pup_diameter * 2. * np.pi

    tilt_plate = np.cos(angle*np.pi/180.) * gridx + np.sin(angle*np.pi/180.) * gridy

    return tilt_plate

def get_zernike_screen(nyx, zernike_coeff, zernike_nb = None, cyx = None, pup_diameter = None,
                       centering='FFTSTYLE'):
    """
    Returns a linear combination of Zernike polynomials.

    Parameters
    ----------
    nyx: tuple of int
        dimensions of the output 2D array.
    zernike_coeff: list of floats
        coefficients of the Zernike polynomials (micron rms).
    cyx: tuple of int, optional
        coordinates of the center of the output 2D array.
    pup_diameter: float
        pupil diameter in pixels where the Zernike polynomials are defined.

    Returns
    -------
    zernike_screen: 2D-array
        Linear combination of Zernike polynomials (microns),
        2D-array of shape nyx.

    Versions
    --------
    Version 1.1, E.H.: specified pupil diameter can be larger than the output
        array dimensions (nyx). In that case, the polynomials are cropped on
        the edges. Can be usefull when effective pupil is not a circle (e.g.
        ELT pupil).
    Version 1.2, E.H.: Noll Zernike numbers can be specified
    """

    if cyx is None:
        cyx = get_frame_center(nyx, centering=centering)
    if pup_diameter is None:
        pup_diameter = np.min(nyx)

    if centering == 'FFTSTYLE':
        # pup_diameter must be odd: round to the closest larger odd value
        pup_diameter_bis = int((pup_diameter // 2 + (pup_diameter % 2 > 1)) * 2 + 1)
        pup_diam2 = (pup_diameter_bis - 1.)/2.
        shift = 0
    elif centering == 'SYMMETRIC':
        # pup_diameter must be even: round to the closest larger even value
        pup_diameter_bis = int((pup_diameter // 2 + (pup_diameter % 2 > 0)) * 2)
        pup_diam2 = pup_diameter_bis/2.
        shift = 1

    if zernike_nb is None:
        zernike_nb = np.arange(len(zernike_coeff))+1

    z_total = np.zeros((pup_diameter_bis,pup_diameter_bis))

    for i, zcoeff in enumerate(zernike_coeff):
        if zernike_coeff[i] != 0.:
            znb = zernike_nb[i]
            z_total = (z_total +
                       zcoeff * zernike1(znb, npix=pup_diameter_bis, outside = 0.))

    zernike_screen = np.zeros(nyx)
    if pup_diameter_bis < np.min(nyx) :
        y1 = int(cyx[0] - pup_diam2)+shift
        y2 = int(cyx[0] + pup_diam2)+1
        x1 = int(cyx[1] - pup_diam2)+shift
        x2 = int(cyx[1] + pup_diam2)+1
        zernike_screen[y1:y2,x1:x2] = z_total
    else :
#        print('WARNING: (get_zernike_screen) The pupil diameter is larger '+
#              'than or equal to the given grid dimensions.')
        zcyx = pup_diam2
        y1 = int(zcyx - nyx[0]/2.)
        y2 = int(zcyx + nyx[0]/2.)
        x1 = int(zcyx - nyx[1]/2.)
        x2 = int(zcyx + nyx[1]/2.)
        zernike_screen = z_total[y1:y2,x1:x2]

    return zernike_screen

def get_zernike_coeff(wavefront, zernike_nb, pup_diameter=None, cyx=None,
                      centering='FFTSTYLE', res_level = 1e-5):
    """
    
    """


    wf_shape = wavefront.shape

    if len(wf_shape) == 2 :
        wavefront = np.expand_dims(wavefront, 0)
        nyx  = wf_shape
        n_wf = 1
    else :
        nyx  = wf_shape[1:]
        n_wf = wf_shape[0]

    if cyx is None:
        cyx = get_frame_center(nyx, centering=centering)

    if pup_diameter is None:
        pup_diameter = nyx[0]
    elif pup_diameter > nyx[0] :
        print("Warning: pupil diameter pup_diameter is larger than the wavefront grid.")

    pup_mask = get_circle_mask(nyx, pup_diameter/2., cyx=cyx, centering=centering)
    n_pup = np.sum(pup_mask)

    z_coeff = np.zeros((n_wf, len(zernike_nb)))

    for j in range(n_wf) :

        wf = wavefront[j]

        for i, znb in enumerate(zernike_nb):
            # print(znb)
            zernike_i = get_zernike_screen(nyx, [1.], zernike_nb=[znb], 
                                           pup_diameter=pup_diameter, centering=centering)

            coeff = 1.

            while np.abs(coeff) > res_level :

                coeff = np.sum(wf * zernike_i) / n_pup

                wf = wf - coeff * zernike_i

                # print(znb,z_coeff[j,i],coeff)
                
                z_coeff[j,i] += coeff
                
                # wf = wavefront[j] - z_coeff[j,i] * zernike_i
                

    if len(wf_shape) == 2 :
        return {'z_coeff': z_coeff[0],
                'res_wf' : wf}
    else :
        return {'z_coeff': z_coeff,
                'res_wf' : wf}