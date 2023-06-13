# -*- coding: utf-8 -*-

import numpy as np

from ..imgproc import get_frame_center

def get_radial_profile(img, nbin=1, centering='FFTSTYLE', cyx=None, mask=None):
    """
    Computes the mean radial profile of the image.

    Parameters
    ----------
    img:
        2D image.
    nbin:
        width of the annuli in pixels
    centering: string
        centering style when cyx is not given. Can be:
        'FFTSTYLE': center located in the center of a pixel
        'SYMMETRIC': center located between 2 pixels
    cyx: tuple of float, optional
        in the form (cy,cx), center of the desired vortex mask.

    Returns
    -------
    Returns a list L of 2 vectors:
        L[0] is a 1D array containing the radial profile values
        L[1] is a 1D array containing the corresponding separation values
             in pixels.
    """
    ny, nx = img.shape

    if cyx is None:
        cyx = get_frame_center((ny,nx), centering=centering)

    y, x = np.indices((img.shape))
    r = np.sqrt((x - cyx[1])**2 + (y - cyx[0])**2)

    r_max = np.max(r[int(cyx[0]),:])

    npts    = int(np.floor(r_max/nbin))
    O       = np.ones((ny,nx))
    Z       = np.zeros((ny,nx))
    profile = np.zeros(npts)
    xvector = np.zeros(npts)

    if ((mask is not None) and (type(mask) != int)):
        in_mask = np.where(mask > 0, O, Z)
    else :
        in_mask = 1.

    for k in range(npts):
        M = np.where(r>=nbin*k, O, Z) * np.where(r<nbin*(k+1), O, Z) * in_mask
        npix = np.sum(M)
        if npix > 0 :
            profile[k] = np.sum(img*M) / npix
            xvector[k] = np.sum(r*M) / npix
        else :
            profile[k] = np.NaN

    return profile, xvector


def get_rms_contrast(img, nbin=1, centering='FFTSTYLE', cyx=None, mask=None):
    """
    Returns the contrast of the coronagraphic image, by computing the rms of
    the values at constant angular separation from the defined center.

    Parameters
    ----------
    img:
        2D image.
    nbin:
        width of the annuli in pixels
    centering: string
        centering style when cyx is not given. Can be:
        'FFTSTYLE': center located in the center of a pixel
        'SYMMETRIC': center located between 2 pixels
    cyx: tuple of float, optional
        in the form (cy,cx), center of the desired vortex mask.

    Returns
    -------
    Returns a list L of 2 vectors:
        L[0] is a 1D array containing the radial profile values
        L[1] is a 1D array containing the corresponding separation values
             in pixels.
    """
    ny, nx = img.shape

    if cyx is None:
        cyx = get_frame_center((ny,nx), centering=centering)

    y, x = np.indices((img.shape))
    r = np.sqrt((x - cyx[1])**2 + (y - cyx[0])**2)

    r_max = np.max(r[int(cyx[0]),:])

    npts    = int(np.floor(r_max/nbin))
    O       = np.ones((ny,nx))
    Z       = np.zeros((ny,nx))
    contrast = np.zeros(npts)
    xvector  = np.zeros(npts)

    if ((mask is not None) and (type(mask) != int)):
        in_mask = np.where(mask > 0, O, Z)
    else :
        in_mask = 1.

    for k in range(npts):
        M = np.where(r>=nbin*k, O, Z) * np.where(r<nbin*(k+1), O, Z) * in_mask
        ind_M = np.where(M>0)
        npix = np.sum(M)
        if npix > 10 :
            contrast[k] = np.std(img[ind_M])
            xvector[k] = np.sum(r*M) / npix
        else :
            contrast[k] = np.NaN

    return contrast, xvector
