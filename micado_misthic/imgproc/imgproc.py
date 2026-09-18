# -*- coding: utf-8 -*-

import numpy as np
import cv2
from scipy import interpolate
import sys
import os
from astropy.io import fits

def get_frame_center(nyx, centering='FFTSTYLE'):
    """
    Returns the coordinates (cy,cx) of the frame center, given the input
    frame shape (ny,nx).

    Parameters
    ----------
    nyx: tuple (ny,nx)
        input frame shape
    centering: string
        centering style. Can be:
        'FFTSTYLE': center located in the center of a pixel
        'SYMMETRIC': center located between 2 pixels

    Returns
    -------
    cyx: tuple (cy,cx)
        coordinates of the frame center.
    """

    ny, nx = nyx

    if centering == 'SYMMETRIC':
#        nx = np.floor(nx/2.)*2
#        ny = np.floor(ny/2.)*2
        cx = (nx-1.) / 2.
        cy = (ny-1.) / 2.
    else :
        if centering != 'FFTSTYLE':
            print('No centering style specified. FFTSTYLE chosen by default.')

        nx = np.floor(nx/2.)*2
        ny = np.floor(ny/2.)*2
        cx = nx / 2.
        cy = ny / 2.

    return (cy,cx)

def rotate_frame(img, angle, interpolation = 'lanczos4', cyx=None):
    """
    Rotates the input frame by the given angle (in degrees), around the center
    of the frame by default, or around the given cxy coordinates.

    Parameters
    ----------
    img: 2D-array
        input frame
    angle: float
        angle for the rotation
    interpolation: string
        interpolation method. Can be:
        'cubic', 'linear', 'nearest', 'area', 'lanczos4'
        'lanczos4' is the default as it seems to produce the least artifacts.
    cyx: tuple (cy,cx)
        center of the rotation. If None is given, rotation is performed around
        the center of the frame.

    Returns
    -------
    rotated_img: 2D-array
        Rotated frame, same dimensions as inpute frame.
    """
    ny, nx = img.shape

    if not cyx:
        cy, cx = get_frame_center((ny, nx))
    else :
        cy, cx = cyx

    # interpolation type
    if interpolation == 'cubic':
        intp        = cv2.INTER_CUBIC
    elif interpolation == 'linear':
        intp        = cv2.INTER_LINEAR
    elif interpolation == 'nearest':
        intp        = cv2.INTER_NEAREST
    elif interpolation == 'area':
        intp        = cv2.INTER_AREA
    elif interpolation == 'lanczos4':
        intp        = cv2.INTER_LANCZOS4

    if abs(angle) > 0.1 :
        M           = cv2.getRotationMatrix2D((cx,cy), angle, 1)
        rotated_img = cv2.warpAffine(img.astype(np.float32), M, (nx, ny), flags=intp)
    else:
        rotated_img = img

    return rotated_img

def do_magnify(img, mag_factor, output_shape=None, interp_deg=1,
               centering='FFTSTYLE'):
    """
    Apply a magnification factor to the input image.
    (factor can be > 1 or < 1).

    Parameters
    ----------
    img: 2D array
        input image to magnify
    mag_factor: float
        magnification factor
    output_shape: int tuple
        (ny,nx) output image shape. If None (default), output shape is the same
        as the input image.
    interp_deg: int
        Degrees of the bivariate spline used to interpolate the magnified image.
        Default is 1. This is a keyword of the interpolate.RectBivariateSpline
        function.
    centering: string
        centering style of the frame if needed. Can be:
        'FFTSTYLE': centered on the central pixel
        'SYMMETRIC': centered between 2 pixels
#    cyx: tuple of float
#        center of the magnification effect. If None, the frame center is taken
#        as center, defined by the centering style.

    Returns
    -------
    final_mag_img: 2D array
        the magnified image.

    Note
    ----
    interpolate module from scipy is needed.
    """

    (ny,nx) = img.shape

    (cy, cx) = get_frame_center((ny,nx), centering=centering)

    x = np.linspace(0,nx-1,nx) - cx
    y = np.linspace(0,ny-1,ny) - cy

    # output grid
    output_nx = int(np.round(mag_factor * nx /2.) *2)
    output_ny = int(np.round(mag_factor * ny /2.) *2)

    (output_cy, output_cx) = get_frame_center((output_ny,output_nx),
                                                    centering=centering)
    xx = np.linspace(np.min(x),np.max(x),output_nx)
    xx = xx - xx[int(output_cx)]

    yy = np.linspace(np.min(y),np.max(y),output_ny)
    yy = yy - yy[int(output_cy)]

    interp = interpolate.RectBivariateSpline(x,y,img,
                                             kx=interp_deg,ky=interp_deg)

    mag_img = interp(xx,yy)

    # plt.figure(num=3)
    # plt.clf()
    # plt.imshow(mag_img)

    if output_shape is not None:

        final_mag_img = np.zeros(output_shape)

        if output_shape[0] > output_ny :
            y1 = int(output_shape[0]/2-output_ny/2)
            y2 = int(output_shape[0]/2+output_ny/2)

            if output_shape[1] > output_nx :

                x1 = int(output_shape[1]/2-output_nx/2)
                x2 = int(output_shape[1]/2+output_nx/2)

                final_mag_img[y1:y2,x1:x2] = mag_img
            else :
                x1 = int(output_nx/2-output_shape[1]/2)
                x2 = int(output_nx/2+output_shape[1]/2)

                final_mag_img[y1:y2,:] = mag_img[:,x1:x2]
        else :
            y1 = int(output_ny/2-output_shape[0]/2)
            y2 = int(output_ny/2+output_shape[0]/2)
            if output_shape[1] > output_nx :
                x1 = int(output_shape[1]/2-output_nx/2)
                x2 = int(output_shape[1]/2+output_nx/2)
                final_mag_img[:,x1:x2] = mag_img[y1:y2,:]
            else :
                x1 = int(output_nx/2-output_shape[1]/2)
                x2 = int(output_nx/2+output_shape[1]/2)
                final_mag_img = mag_img[y1:y2,x1:x2]

    else:
        final_mag_img = mag_img

    return final_mag_img


def do_shift(img, shiftxy, interp_deg=1):
    """
    Shifts an image by an amount of pixel that can be fractional.

    Parameters
    ----------
    img: 2D array
        input image
    shiftxy: tuple of float
        the shift amplitudes in x and y.
        Positive shift amplitude shifts the image towards the right or the top
        respectively.
    interp_deg: int
        Degrees of the bivariate spline used to interpolate the magnified image.
        Default is 1. This is a keyword of the interpolate.RectBivariateSpline
        function.

    Returns
    -------
    shift_img: 2D array
        the shifted image.

    Note
    ----
    interpolate module from scipy is needed.
    """
    (ny,nx) = img.shape

    x = np.linspace(0,nx-1,nx)
    y = np.linspace(0,ny-1,ny)

    # output grid
    xx = x - shiftxy[0]
    yy = y - shiftxy[1]

    interp = interpolate.RectBivariateSpline(x,y,img,
                                             kx=interp_deg,ky=interp_deg)

    shift_img = interp(xx,yy)

    return shift_img

def fft_resize(img_0, lbd0, sampling_misthic, pixscale_in_mas, write_dir=0, plotting=True):
    '''
    ===========================================================================
    Resizing image via FFT
    ---------------------------------------------------------------------------
    Inputs:
        img_0            : Initial image to resize (can be single frame or image cube)
        lbd0             : Wavelength at which image was observed (in um)
        sampling_misthic : Sampling in pixels per lambda/d of misthic simulation images; 2 here
        pixscale_in_mas  : Pixel scale in mas/px; either 1.5 or 4 here
        write_dir        : 
        plotting         : When set to True, plots images of each step in the resizing process; for
                           cubes, only first frame is plotted
        write            : When set to True, writes resized image to fits file (currently, directory
                           is hard-coded in, but could change that later)
    ---------------------------------------------------------------------------
    Outputs:
        resized_fft : Image or image cube rezised to micado target dimensions via FFT method
        coeff       : FFT scaling coefficient corresponding to the ratio of the initial and final image
                      dimensions applied to resized image in order to maintain the correct intensity values
    ---------------------------------------------------------------------------
    Authors: H. Baran    
    ===========================================================================
    '''
    if len(np.shape(img_0)) <= 2:
        dim = len(img_0[0,:])
        img = np.zeros((1,dim,dim))
        img[0] = img_0
    else:
        img = img_0
        dim = len(img[0,0,:]) # square
    # print("PSF cube shape: ", np.shape(psf))

    # Size of resized image (even):
    dim1 = round(dim/sampling_misthic*lbd0/38.542/4.85*1000./pixscale_in_mas/2)*2
    # dim1 = int(dim * 2)
    # Check:
    if dim1 % 2 != 0:
        sys.exit("Final dimensions need to be even ! ): {}".format(dim1))    
    
    # Begin FFT process
    frames = img[:,0,0]
    resized_fft = np.zeros((len(frames), dim1, dim1))#,dtype=np.complex128)

    
    for i in range(len(frames)):
    # for i in range(1):
        # Shift image by (-dim/2, -dim/2); puts PSF in four corners
        shift1 = np.roll(img[i], (int(-dim/2), int(-dim/2)), axis=(1, 0))
    
        # FFT of image to get pupil
        fft1 = np.fft.fftn(shift1, axes=(1,0))

        #  Plot pupil
        shift2 = np.roll(fft1, (int(dim/2), int(dim/2)), axis=(1, 0))

        # Add zeroes to pad image to dim1 size - only works for enlarging image
        new_size = np.zeros((dim1,dim1),dtype=np.complex128)
        new_size[0:dim,0:dim] = shift2 + new_size[0:dim,0:dim] 
        
        # Move FT image to corners
        shift3 = np.roll(new_size, (int(-dim/2), int(-dim/2)), axis=(1, 0))
        
        # Inverse FFT of image at corners
        # > note: keep type of fft constant ! fft = ifft; fft2 = ifft2; fftn = ifftn
        inv = np.fft.ifftn(shift3, axes=(1,0))
        
        # Shift image back to center
        shift4 = np.roll(inv, (int(dim1/2), int(dim1/2)), axis=(1, 0))
        resized_fft[i] = shift4
        
        
        # # Checking image vals
        # orig_tot = np.sum(psf[i])
        # resiz_tot = np.sum(resized_fft[i])
        
        # Dimension ration:
        # print(dim1, dim)
        dim_rat = (dim1 / dim) ** 2
        
        # # Scaling ratio:
        # sca_rat = (dim1 / dim) ** 2
        
        # Apply coeff to resized img
        coeff = dim_rat #* sca_rat
        resized_fft[i] *= coeff
        
        # print("Max orig:", np.max(psf[i]))
        # print("Max resized:", np.max(resized_fft[i]))
        # print("Dimensions:", dim, dim1)
        # print("Img totals (orig, resiz):", orig_tot, resiz_tot)
    
    # convert to real values
    # np.real(np.conjugate(resized_fft))
    # type(np.float64(0).item(np.real(resized_fft)))
    print("Cube Type after resizing:", type(resized_fft[0][0][0]))
    
    if write_dir != 0:
        fits.writeto(write_dir+'fft_resized.fits', resized_fft)

    # # Plots
    # if plotting==True:
    #     # Turn plotting off atm; too many frames in cube
    #     fig, ax = plt.subplots(3,3, figsize=(15,15))
        
    #     ax[0,0].imshow(img[0])
    #     ax[0,0].set_title('Original PSF')
        
    #     ax[0,1].imshow(shift1)
    #     ax[0,1].set_title("Shift 1")
        
    #     ax[0,2].imshow(np.abs(np.conjugate(fft1)*fft1))
    #     # ax[0,2].imshow(fft1)
    #     ax[0,2].set_title("FFT 1 (Pupil)")
        
    #     ax[1,0].imshow(np.abs(np.conjugate(shift2)*shift2))
    #     # ax[1,0].imshow(shift2)
    #     ax[1,0].set_title("Shift 2 (Pupil)")
    
    #     ax[1,1].imshow(np.abs(np.conjugate(new_size)*new_size))
    #     # ax[1,1].imshow(new_size)
    #     ax[1,1].set_title("Padding with Zeros (New Dims)")
        
    #     ax[1,2].imshow(np.abs(np.conjugate(shift3)*shift3))
    #     # ax[1,2].imshow(shift3)        
    #     ax[1,2].set_title("Shift 3")
        
    #     ax[2,1].imshow(resized_fft[0])
    #     ax[2,1].set_title("iFFT")
    
    #     fig.delaxes(ax[2,0])
    #     fig.delaxes(ax[2,2])
    #     fig.suptitle("Fourier Process")
    #     plt.subplots_adjust(hspace=0.3)
    #     plt.show()

    return resized_fft, coeff

def bin_images(sci_cube, n_bin_width=None, n_out=None):
    """
    Returns a cube of images averaged by bins of n_bin images.

    Parameters
    ----------
    sci_cube : 2D or 3D array
        input science coronagraphic single image or image cube.
        Dimensions are (ny, nx) for 1 frame or (n_img, ny, nx) for a cube.
    n_bin : integer
        number of images in the returned cube.
        must be comprised between 0 and n_img.
        - if 0: no frame averaging, the returned cube is a copy of the input.
        - if 1: returns the average of the whole cube.
        - if n_bin=integer < n_img: the returned cube is made of n_bin images,
            each being the average of n_img/n_bin images of the input sci_cube.

    Returns
    -------
    sci_cube_binned: 2D or 3D array
        binned image cube.
    """

    if len(sci_cube.shape) == 2:
        sci_cube = np.expand_dims(sci_cube, 0)
        n_sci = 1
        ny, nx = sci_cube.shape
    else :
        n_sci, ny, nx = sci_cube.shape

    if n_out is None and n_bin_width > 0:
        n_out = n_sci // n_bin_width
    if n_bin_width is None and n_out > 0:
        n_bin_width = n_sci // n_out

    if ((n_out == 1 or n_bin_width == 0) and n_sci > 1):
        # all images averaged
        sci_cube_binned = np.expand_dims(np.mean(sci_cube, axis=0),0)
    elif ((n_out == 0 or n_bin_width == 1) and n_sci > 1):
        sci_cube_binned = sci_cube.copy()
    elif n_sci > 1:
        # bin the images
        sci_cube_binned = np.zeros((n_out, ny, nx))
        for i in range(n_out):
            sci_cube_binned[i]= np.mean(sci_cube[n_bin_width*i:n_bin_width*(i+1),:,:], axis=0)
    else :
        sci_cube_binned = sci_cube.copy()

    return sci_cube_binned


def pad_and_roll(a, shiftxy) :
    """
    Returns the input array a rolled along a given axis. Elements that roll
    beyond the last position are not re-introduced, the output array is zero
    padded instead.

    Parameters
    ----------
    a : 2D array
        input array.
    shiftxy : tuple of ints
        the number of places by which elements are shifted, along the x and y
        axes. shiftxy = (shift_x, shift_y)

    Returns
    -------
    c : 2D array
        output array, with the same shape as a.

    v1. EHu 2020-04-18
    """
    shift_x = shiftxy[0]
    shift_y = shiftxy[1]

    ### Zero padding
    if shift_x > 0 :
        b = np.pad(a, ((0, 0), (0, shift_x)), mode='constant')
    elif shift_x < 0 :
        b = np.pad(a, ((0, 0), (-shift_x, 0)), mode='constant')
    else :
        b = a.copy()
    if shift_y > 0 :
        b = np.pad(b, ((0, shift_y), (0, 0)), mode='constant')
    elif shift_y < 0 :
        b = np.pad(b, ((-shift_y, 0), (0, 0)), mode='constant')

    ### Roll
    c = b.copy()
    if (shift_x != 0 or shift_y != 0) :
        c = np.roll(b,(shift_x,shift_y),(1,0))

    if shift_x > 0 :
        c = c[:, :-shift_x]
    elif shift_x < 0 :
        c = c[:, -shift_x:]
    if shift_y > 0 :
        c = c[:-shift_y,:]
    elif shift_y < 0 :
        c = c[-shift_y:, :]

    return c

def micado_adi(cube, cube_psf, par_dir):
    '''
    ===========================================================================
    ADI processing for images (ADI for coro + planet images; sum for PSF)
    ---------------------------------------------------------------------------
    Inputs:
        cube     : Image cube over which to perform ADI calculation (ideally with 
                   photons and noise applied; output of noise function)
        cube_psf : PSF cube over which to sum along time axis (ideally with photons
                   and noise applied; output of noise function)
        par_dir  : Directory of original simulations (not resized); necessary
                   or extracting table of parallactic angle values
    ---------------------------------------------------------------------------
    Outputs:
        adi_sum : Sum total of ADI frames 
        psf_sum : Sum total of PSF frames
    ---------------------------------------------------------------------------
    Authors: H. Baran, P. Baudoz
    ===========================================================================
    '''
    # Open parallactic angle table
    table = [f for f in os.listdir(par_dir) if f.endswith('_parangle_tab.txt')][0]
    #table = [f for f in os.listdir(par_dir) if f.endswith('imagespercube.txt')][0]
    parallactic = np.loadtxt(par_dir+table,skiprows=1)
    
    # Length check for frames
    if len(parallactic) != len(cube[:,0,0]):
        sys.exit("Number of parallactic angles does not equal number of frames ! ):")
    
    # New cube dimensions
    if len(np.shape(cube)) <= 2:
        frames = [1]
    else:
        frames = cube[:,0,0]
    
    # ADI
    # Recalculate mean coronographic img w noise introduced
    mean_coro = np.mean(cube,axis=0)
    # print("Before ADI")
    cube2 = cube.copy()
    for frame in range(len(frames)):
        # Subtract mean
        cube2[frame] -= mean_coro
        # De-rotating (rotation code from misthic_func)
        parang = parallactic[frame]
        cube2[frame] = rotate_frame(cube2[frame], parang)

        
    # Sum PSF cube
    psf_sum = np.sum(cube_psf, axis=0)
    
    # Mean of derotated images
    # adi_stack = np.mean(cube_coronogr,axis=0)
    adi_sum = np.sum(cube2, axis=0)
    
    adi_stdev = np.std(adi_sum)
    print("Standard dev of image cube after noise and ADI:", adi_stdev)

    return adi_sum, psf_sum
