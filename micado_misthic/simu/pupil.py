# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.filters import gaussian_filter

from ..imgproc import get_frame_center
from .zernike import get_zernike_screen


def get_hex_segments(nyx, n_rings, c_to_vertex, seg_gap, cyx = None,
                     angle = 0., darkcenter=False, smooth_edge=0.,
                     z_coeff=None) :
    """
    Function adapted from the IDL PROPER function prop_hex_wavefront
    by John Krist (JPL), July 2007.
    """
    if cyx is None:
        cy, cx = get_frame_center(nyx)

    aperture  = np.zeros(nyx)
    phase     = np.zeros(nyx)

    angle_rad = angle * np.pi / 180
    hexsep    = 2 * c_to_vertex * np.cos(np.pi/6.) + seg_gap

    i_seg = 0

    # The segments are defined row by row, symmetrically
    for iring in range(n_rings+1) :

        x = hexsep * np.cos(30*np.pi/180) * iring
        y = -n_rings * hexsep + iring * hexsep * 0.5

        for iseg in range(2*n_rings-iring+1):
            ### create hex segment on one side
            if ( iring != 0 or not (iseg == n_rings and (darkcenter))):
                xhex = x * np.cos(angle_rad) - y * np.sin(angle_rad) + cx
                yhex = x * np.sin(angle_rad) + y * np.cos(angle_rad) + cy
                segment = get_polygon(nyx, 6, c_to_vertex, cyx=(yhex,xhex), angle=angle)
                # Gaussian smoothing function
                aperture = aperture + gaussian_filter(segment, smooth_edge)
                # Zernike polynomial on individual segments
                if z_coeff is not None:
                    if not np.alltrue(z_coeff[i_seg] == 0) :
                        phase = phase + segment * get_zernike_screen(nyx, z_coeff[i_seg], cyx=(yhex,xhex),pup_diameter=2*c_to_vertex )
                    i_seg = i_seg + 1

            ### create hex segment on opposite side
            if ( iring != 0 ):
                xhex = -x * np.cos(angle_rad) - y * np.sin(angle_rad) + cx
                yhex = -x * np.sin(angle_rad) + y * np.cos(angle_rad) + cy
                # Gaussian smoothing function
                segment = get_polygon(nyx, 6, c_to_vertex, cyx=(yhex,xhex), angle=angle)
                aperture = aperture + gaussian_filter(segment, smooth_edge)
                # Zernike polynomial on individual segments
                if z_coeff is not None:
                    if not np.alltrue(z_coeff[i_seg] == 0) :
                        phase = phase + segment * get_zernike_screen(nyx, z_coeff[i_seg], cyx=(yhex,xhex),pup_diameter=2*c_to_vertex )
                    i_seg = i_seg + 1

            ### Next row
            y = y + hexsep

    return aperture, phase


def get_polygon(nyx, nvert, rad_pix, cyx=None, angle=0., smooth_edge=0.) :
    """
    Function inspired from the IDL PROPER function prop_polygon
    by John Krist (JPL), February 2005.
    """
    if cyx is None:
        cy, cx = get_frame_center(nyx)
    else :
        cy, cx = cyx

    angle_rad = angle * np.pi / 180

    t   = -np.arange(nvert) / nvert * 2 * np.pi	# force clockwise vertex list
    xp0 = np.cos(t)
    yp0 = np.sin(t)

    xp  = xp0 * np.cos(angle_rad) - yp0 * np.sin(angle_rad)
    yp  = xp0 * np.sin(angle_rad) + yp0 * np.cos(angle_rad)

    xp  = xp * rad_pix
    yp  = yp * rad_pix

    gridy, gridx = np.indices(nyx)
    gridy = gridy - cy
    gridx = gridx - cx

    image = np.zeros(nyx)

    for i in range(nvert):
        xa = xp[i]
        if i == nvert-1 :
            xb = xp[0]
        else:
            xb = xp[i+1]

        ya = yp[i]
        if i == nvert-1 :
            yb = yp[0]
        else:
            yb = yp[i+1]

        s = (yb-ya) / (xb-xa)
        b = ya - xa * s

        s2 = (xb-xa) / (ya-yb)
        b2 = -xa - ya * s2

        if np.abs(s) > 1e10:
            if b2 < 0.:
                area = np.where(gridx < -b2, 1, 0)
            else :
                area = np.where(gridx > -b2, 1, 0)
        else :
            if b < 0:
                area = np.where(gridy > s * gridx + b, 1, 0)
            else:
                area = np.where(gridy < s * gridx + b, 1, 0)

        image = image + area

    image = np.float32(np.where(image == nvert, 1, 0))

    if smooth_edge > 0 :
        image = gaussian_filter(image, smooth_edge)

    return image