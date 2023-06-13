# -*- coding: utf-8 -*-

import numpy as np

from .imgproc import get_frame_center

def get_circle_mask(nyx, radius, cyx=None, centering='FFTSTYLE'):
    """
    Creates a circular mask of dimensions given in tuple nyx=(ny,nx), and
    defined by a radius and center coordinates.
    If center is not defined, it is set by default in the exact middle.
    Output values are 1 inside the circle, 0 outside.

    Parameters
    ----------
    image : array_like
        Input 2D array.
    radius : float
        Radius of the mask in pixels.
    cyx : tuple of float, optional
        cyx = (cy,cx) gives the position of the image center [pix].
        Can be fractional.
        If not specified, the center is defined at the center of the image.
    centering: string
        centering style when cyx is not given. Can be:
        'FFTSTYLE': center located in the center of a pixel
        'SYMMETRIC': center located between 2 pixels

    Returns
    -------
    circle_mask : array_like
        2D array containing 1 and 0 values only.
    """
    if radius == 0. :
        return np.zeros(nyx)
    
    else :
        
        if cyx is None:
            cy, cx = get_frame_center(nyx, centering=centering)
        else :
            cy, cx = cyx
    
        cx2 = np.round(cx*2.)/2.
        cy2 = np.round(cy*2.)/2.
    
        gridy, gridx = np.indices(nyx)
        gridx = gridx - cx2
        gridy = gridy - cy2
        gridr = np.sqrt(gridx**2. + gridy**2.)
    
        circle_mask = (gridr <= radius).astype(int)
    
        return circle_mask

def get_rect_area(nyx, width_pix, height_pix, cyx=None, centering='FFTSTYLE'):
    """
    Creates a rectangular mask of dimensions given in tuple nyx=(ny,nx), and
    defined by a width and height in pixels.
    If the center coordinates cyx are not defined, it is set by default in the
    exact middle (between pixels or centered on a pixel, depending on the
    centering style).
    Output values are 1 inside the rectangular area, 0 outside.

    Parameters
    ----------
    image : array_like
        Input 2D array.
    radius : float
        Radius of the mask in pixels.
    cyx : tuple of float, optional
        cyx = (cy,cx) gives the position of the image center [pix].
        Can be fractional.
        If not specified, the center is defined at the center of the image.
    centering: string
        centering style when cyx is not given. Can be:
        'FFTSTYLE': center located in the center of a pixel
        'SYMMETRIC': center located between 2 pixels

    Returns
    -------
    circle_mask : array_like
        2D array containing 1 and 0 values only.
    """
    if cyx is None:
        cy, cx = get_frame_center(nyx, centering=centering)
    else :
        cy, cx = cyx

    cx2 = np.round(cx*2.)/2.
    cy2 = np.round(cy*2.)/2.

    gridy, gridx = np.indices(nyx)
    gridx = np.abs(gridx - cx2)
    gridy = np.abs(gridy - cy2)

    rect_mask = ((gridx < width_pix/2.) & (gridy < height_pix/2.)).astype(int)

    return rect_mask