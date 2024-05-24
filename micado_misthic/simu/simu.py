# -*- coding: utf-8 -*-

import numpy as np
import pyfftw
from poppy import matrixDFT

from ..simu.zernike import get_tilted_wavefront
from ..imgproc import get_circle_mask

def do_fftw_crosszp(img, zero_pad_factor = 2, fov = 5, fov_pix = None,
                    planner_effort='FFTW_MEASURE', #'FFTW_PATIENT', #, #'FFTW_ESTIMATE'
                    auto_align_input=False, overwrite_input = True) :
    """
    Same as numpy.fft.fft with 2-step padding (one direction after the other).
    Uses the FFT function of the pyfftw module.
    """

    grid_width = (img.shape)[0]
    output_size = int(np.round(grid_width * zero_pad_factor))
    # print("output_size=", output_size)
    ### Define the field of view in pixels
    if fov_pix is None :
        fov_pix  = int(np.ceil(fov * zero_pad_factor /2.)*2.)
        x1       = int(output_size/2.-fov_pix/2.)
        x2       = int(output_size/2.+fov_pix/2.)
    else :
        x1 = fov_pix[0]
        x2 = fov_pix[1]

    ### padded image
    padded_img = np.zeros((output_size,grid_width), dtype=type(img[0,0]))
    zp1 = int((output_size-grid_width)/2.)
    zp2 = int((output_size+grid_width)/2.)
    padded_img[zp1:zp2,] = img
    
    # ################ DEBUG ##########################
    # print("padded_img.shape", padded_img.shape)
    # ################ END DEBUG ######################
    
    pyfftw.interfaces.cache.enable() ## Elsa 2024.03.19

    img_fft_tmp = pyfftw.interfaces.numpy_fft.fft(np.fft.fftshift(padded_img, axes=0), axis=0,
                                                planner_effort = planner_effort,
                                                auto_align_input = auto_align_input,
                                                threads=8, overwrite_input=overwrite_input )#,
                                                #norm='backward') #/np.sqrt(output_size*grid_width)
    ### Elsa 2024.03.19 added the normalization factor
    # img_fft_tmp = np.fft.fft(np.fft.fftshift(padded_img, axes=0), axis=0, norm='backward')
    # print("img_fft_tmp.shape", img_fft_tmp.shape)
    final_n = x2-x1
    c = int(final_n/2.)
    # print("final_n=", final_n)
    img_fft = np.zeros((final_n, output_size), dtype=type(img_fft_tmp[0,0]))
    # print("img_fft.shape", img_fft.shape)
    # print("c, zp1, zp2", c, zp1, zp2)
    img_fft[:c,zp1:zp2] = img_fft_tmp[-c:,]
    img_fft[c:,zp1:zp2] = img_fft_tmp[:c,]

    img_fft_tmp = pyfftw.interfaces.numpy_fft.fft(np.fft.fftshift(img_fft, axes=1), axis=1,
                                                planner_effort = planner_effort,
                                                auto_align_input = auto_align_input,
                                                threads=8, overwrite_input=overwrite_input)#,
                                                #norm='backward') #/np.sqrt(final_n*output_size)
    ### Elsa 2024.03.19 added the normalization factor
    # img_fft_tmp = np.fft.fft(np.fft.fftshift(img_fft, axes=1), axis=1, norm='forward')
    
    ### Elsa 2024.03.19: note that the simple numpy fft goes faster for a single computation,
    ### but if the FFT computation is repeated (with the same parameters), it is worth planning
    ### the computation with pyFFT (that is why it takes much longer for the first iteration)

    img_fft = np.zeros((final_n, final_n), dtype=type(img_fft_tmp[0,0]))

    img_fft[:,:c] = img_fft_tmp[:,-c:]
    img_fft[:,c:] = img_fft_tmp[:,:c]

    return img_fft

def propagate_mono_lyot(input_wavefront, lyot_mask, occulter_fov = 1.,
                        mft_sampling = 10., occ_rad_pix = 2.,
                        det_fov_pix = None, det_sampling = 2.,
                        centering='FFTSTYLE', lbd=1., lbd_ref=None,
                        lyot_output=False) :
    """
    Propagates a monochromatic input wavefront (entrance pupil) to the detector
    in the case of Lyot coronagraph with a focal plane occulting mask.
    Intermediate planes are a focal plane with occulting mask, and a Lyot stop
    in the re-imaged pupil plane.
    The method of the semi-analytic computation of the Lyot coronagraph is
    used: only the field occulted by the focal plane mask is computed and
    and propagated to the Lyot plane. The field after the Lyot stop is then
    computed as the subtraction of the input field and the field rejected
    by the focal plane mask.

    Pupil plane -> Focal plane
    Matrix Fourier Transform (MFT, based on Soummer 2017) is used to compute
    the field in the focal plane with high sampling rate (mft_sampling)
    on a reduced field of view (occulter_fov).
    The function used is from the poppy.matrixDFT module.
    https://github.com/mperrin/poppy

    Focal plane -> Lyot plane
    Inverse MFT is used.

    Lyot plane -> detector plane
    FFTW is used, with zero padding (do_fftw_crosszp).

    Parameters
    ----------
    input_wavefront :
        entrance pupil plane field (can be real, or complex with amplitude and phase)
    lyot_mask :
        Lyot mask (can be real for a Lyot stop, or complex for a phase apodized
        mask like Lyot-Plane Phase Mask)
    occulter_fov = 1. [lambda/D]
        focal plane field of view for computing the field. Should be reduced
        as much as possible to minimize computation time with the MFT, depending
        on the occulting mask radius.
    mft_sampling = 10. [pixels/(l/D)]
        sampling factor of the computed focal plane field with the MFT
        (corresponds to the number of pixels per lambda/D)
    occ_rad_pix = 2. [pixels]
        radius of the occulting mask in pixels.
    det_fov_pix = None [tuple (x1,x2)]
        boundary of the field of view on the detector (given in pixels).
    det_sampling = 2. [pixels/(l/D)]
        sampling of the image on the detector.
    centering='FFTSTYLE'
        centering style of the focal plane image. Can be:
        'FFTSTYLE': centered on the central pixel
        'SYMMETRIC': centered between 2 pixels
    lbd = 1.
        wavelength of the simulated image
    lbd_ref = None
        wavelength used as reference to define the sampling. If given, the
        detector image sampling will be scaled by lbd/lbd_ref.

    Returns
    -------
    detector_img: 2D-array
        final detector image intensity.
        Dimensions are defined by det_fov_pix[1] - det_fov_pix[0].
    """


    pup_shape = input_wavefront.shape

    if lbd_ref is None:
        lbd_ref = lbd

    lbd_coeff = lbd/lbd_ref

    if centering == 'SYMMETRIC':
        # phase ramp to offset the image by half a pixel
        demipix_fp  = 1. / (np.sqrt(2.)*np.float64(mft_sampling))
        demipix_det = 1. / (np.sqrt(2.)*np.float64(det_sampling*lbd_coeff))
        tilted_wf = get_tilted_wavefront(input_wavefront, pup_shape[0],angle = 45.)
        tiptilt_form_fp = np.exp( -1j * tilted_wf * demipix_fp)
        tiptilt_form_det = np.exp(-1j * tilted_wf * demipix_det)
    else:
        tiptilt_form_fp = 1.
        tiptilt_form_det = 1.

    npix         = (int(np.round(occulter_fov * mft_sampling)) // 2 ) *2 # force even dimension
    occulter_fov = float(npix) / float(mft_sampling)

    focal_plane = matrixDFT.matrix_dft(input_wavefront*tiptilt_form_fp,
                                       occulter_fov, npix, centering='FFTSTYLE')

    if occ_rad_pix > 0:
        occulter_area = get_circle_mask((npix,npix), occ_rad_pix, centering=centering)
        lyot_plane_rejected = matrixDFT.matrix_dft(focal_plane*occulter_area,
                                                   occulter_fov, pup_shape, inverse=True,
                                                   centering='FFTSTYLE')

        ### save rejected light ###
#        TF_mask = matrixDFT.matrix_dft(occulter_area, occulter_fov, pup_shape, inverse=True,
#                                                   centering='FFTSTYLE')
#        print('Write fits: TF[M] and P*TF[M]')
#        write_fits('P_conv_TF-M.fits', np.abs(lyot_plane_rejected)**2, overwrite=True)
#        write_fits('TF-M_real.fits', np.real(TF_mask), overwrite=True)
#        write_fits('TF-M_imag.fits', np.imag(TF_mask), overwrite=True)
#        write_fits('TF-P_times_M.fits', np.abs(focal_plane*occulter_area)**2, overwrite=True)
#        write_fits('TF-P_times_1-M.fits', np.abs(focal_plane*(1.-occulter_area))**2, overwrite=True)
#        write_fits('TF-P.fits', np.abs(focal_plane)**2, overwrite=True)
    else:
        lyot_plane_rejected = 0.

    before_lyot_stop = (input_wavefront*tiptilt_form_fp - lyot_plane_rejected)



    after_lyot_stop = before_lyot_stop * lyot_mask
    
    detector_img = np.abs(do_fftw_crosszp(after_lyot_stop*np.conj(tiptilt_form_fp)*tiptilt_form_det,
                                          zero_pad_factor = det_sampling*lbd_coeff,
                                          fov_pix = det_fov_pix))**2

    if lyot_output is True:
        final_output = {'img_detector': detector_img,
                        'img_before_lyot': np.abs(before_lyot_stop)**2,
                        'img_after_lyot': np.abs(after_lyot_stop)**2}
    else:
        final_output = {'img_detector': detector_img}

    return final_output

def propagate_mono_vortex(input_wavefront, vortex_mask, lyot_mask,
                          fp_fov = 8, fp_sampling = 2,
                          det_fov_pix = None, det_sampling = 2.,
                          centering='SYMMETRIC', lbd=1., lbd_ref=None,
                          fft_method='MFT',
                          lyot_output=False) :
    """
    Propagates a monochromatic input wavefront (entrance pupil) to the detector
    in the case of a focal plane phase mask.
    Intermediate planes are a focal plane with phase mask, and a Lyot stop
    in the re-imaged pupil plane.

    Pupil plane -> focal plane -> Lyot plane
    For the FFT computation from entrance pupil to focal plane, it is possible
    to use either the MFT or the FFTW (see below).
    In general, MFT should be preferred for small FoV and/or very high sampling.
    For focal plane to Lyot plane propagation, the inverse of the above choosen
    method is used.

    Matrix Fourier Transform (MFT, based on Soummer 2017) is used to compute
    the field in the focal plane with the given sampling rate (fp_sampling)
    on a reduced field of view (fp_fov).
    The function used is from the poppy.matrixDFT module.
    https://github.com/mperrin/poppy

    FFTW is performed with zero-padding in 2 steps.

    Lyot plane -> detector plane
    FFTW is used, with 2-step zero padding (do_fftw_crosszp).

    Parameters
    ----------
    input_wavefront :
        entrance pupil plane field (can be real, or complex with amplitude and phase)
    vortex_mask :
        focal plane phase mask (complex)
    lyot_mask :
        Lyot mask (can be real for a Lyot stop, or complex for a phase apodized
        mask like Lyot-Plane Phase Mask)
    fp_fov = 8.
        focal plane field of view for computing the field.
    fp_sampling = 2. [pixels/(l/D)]
        sampling factor of the computed focal plane field with the MFT
        (corresponds to the number of pixels per lambda/D)
    det_fov_pix = None [tuple (x1,x2)]
        boundary of the field of view on the detector (given in pixels).
    det_sampling = 2. [pixels/(l/D)]
        sampling of the image on the detector.
    centering='SYMMETRIC'
        centering style of the focal plane image. Can be:
        'FFTSTYLE': centered on the central pixel
        'SYMMETRIC': centered between 2 pixels
    lbd = 1.
        wavelength of the simulated image
    lbd_ref = None
        wavelength used as reference to define the sampling. If given, the
        detector image sampling will be scaled by lbd/lbd_ref.
    fft_method = 'FFTW'
        FFT method for propagation from pupil to focal plane to Lyot plane.
        Can be:
        'FFTW'
        'MFT' (default)

    Returns
    -------
    detector_img: 2D-array
        final detector image intensity.
        Dimensions are defined by det_fov_pix[1] - det_fov_pix[0].
    """

    pup_shape = input_wavefront.shape

    if lbd_ref is None:
        lbd_ref = lbd

    lbd_coeff = lbd/lbd_ref
    # print("lbd_coeff=",lbd_coeff)

    if centering == 'SYMMETRIC':
        demipix_fp  = 1. / (np.sqrt(2.)*np.float64(fp_sampling))
        demipix_det = 1. / (np.sqrt(2.)*np.float64(det_sampling*lbd_coeff))
        tilted_wf   = get_tilted_wavefront(input_wavefront, pup_shape[0], angle = 45.)
        tiptilt_form_fp = np.exp(-1j*tilted_wf*demipix_fp)
        tiptilt_form_det = np.exp(-1j*tilted_wf*demipix_det)
    else:
        tiptilt_form_fp = 1.
        tiptilt_form_det = 1.

    if isinstance(vortex_mask, float) :
        # NO MASK
        before_lyot_stop = input_wavefront*tiptilt_form_fp
    else:
        if fft_method =='MFT':
            focal_plane = matrixDFT.matrix_dft(input_wavefront*tiptilt_form_fp, fp_fov, fp_sampling*fp_fov)
            before_lyot_stop = matrixDFT.matrix_dft(focal_plane*vortex_mask, fp_fov, pup_shape, inverse=True)
    #                                                centering='FFTSTYLE')
        else :
            focal_plane = do_fftw_crosszp(input_wavefront*tiptilt_form_fp, 
                                          zero_pad_factor = fp_sampling, fov = fp_fov)
            before_lyot_stop = do_fftw_crosszp(focal_plane*vortex_mask,
                                               zero_pad_factor = pup_shape[0]/fp_sampling,
                                               fov = fp_fov)

    after_lyot_stop = before_lyot_stop * lyot_mask
    after_lyot_stop2 = after_lyot_stop*np.conj(tiptilt_form_fp)*tiptilt_form_det

    # ################ DEBUG ##########################
    # print("lbd_coeff", lbd_coeff)
    # print("det_sampling", det_sampling)
    # ################ END DEBUG ######################
    # print("zero_pad_factor=", det_sampling*lbd_coeff)
    detector_img = np.abs(do_fftw_crosszp(after_lyot_stop2,
                                          zero_pad_factor = det_sampling*lbd_coeff,
                                          fov_pix = det_fov_pix))**2

#    detector_img = np.abs(matrixDFT.matrix_dft(after_lyot_stop, (det_fov_pix[1]-det_fov_pix[0])/det_sampling, (det_fov_pix[1]-det_fov_pix[0])))**2

    if lyot_output is True:
        final_output = {'img_detector': detector_img,
                        'img_before_lyot': np.abs(before_lyot_stop)**2,
                        'img_after_lyot': np.abs(after_lyot_stop)**2}
    else:
        final_output = {'img_detector': detector_img}

    return final_output


def propagate_mono_perfect(input_wavefront, pupil_mask, lyot_mask, nocoro=False,
                          det_fov_pix = None, det_sampling = 2.,
                          centering='SYMMETRIC', lbd=1., lbd_ref=None,
                          fft_method='MFT',
                          lyot_output=False) :
    """
    Propagates a monochromatic input wavefront (entrance pupil) to the detector
    in the case of a focal plane phase mask.
    Intermediate planes are a focal plane with phase mask, and a Lyot stop
    in the re-imaged pupil plane.

    Pupil plane -> focal plane -> Lyot plane
    For the FFT computation from entrance pupil to focal plane, it is possible
    to use either the MFT or the FFTW (see below).
    In general, MFT should be preferred for small FoV and/or very high sampling.
    For focal plane to Lyot plane propagation, the inverse of the above choosen
    method is used.

    Matrix Fourier Transform (MFT, based on Soummer 2017) is used to compute
    the field in the focal plane with the given sampling rate (fp_sampling)
    on a reduced field of view (fp_fov).
    The function used is from the poppy.matrixDFT module.
    https://github.com/mperrin/poppy

    FFTW is performed with zero-padding in 2 steps.

    Lyot plane -> detector plane
    FFTW is used, with 2-step zero padding (do_fftw_crosszp).

    Parameters
    ----------
    input_wavefront :
        entrance pupil plane field (can be real, or complex with amplitude and phase)
    pupil_mask :
        entrance pupil mask (contains 0 and 1)
    lyot_mask :
        Lyot mask (can be real for a Lyot stop, or complex for a phase apodized
        mask like Lyot-Plane Phase Mask)
    det_fov_pix = None [tuple (x1,x2)]
        boundary of the field of view on the detector (given in pixels).
    det_sampling = 2. [pixels/(l/D)]
        sampling of the image on the detector.
    centering='SYMMETRIC'
        centering style of the focal plane image. Can be:
        'FFTSTYLE': centered on the central pixel
        'SYMMETRIC': centered between 2 pixels
    lbd = 1.
        wavelength of the simulated image
    lbd_ref = None
        wavelength used as reference to define the sampling. If given, the
        detector image sampling will be scaled by lbd/lbd_ref.
    fft_method = 'FFTW'
        FFT method for propagation from pupil to focal plane to Lyot plane.
        Can be:
        'FFTW' (default)
        'MFT'

    Returns
    -------
    detector_img: 2D-array
        final detector image intensity.
        Dimensions are defined by det_fov_pix[1] - det_fov_pix[0].
    """

    pup_shape = input_wavefront.shape

    if lbd_ref is None:
        lbd_ref = lbd

    lbd_coeff = lbd/lbd_ref

    if centering == 'SYMMETRIC':
        demipix_det = 1. / (np.sqrt(2.)*np.float64(det_sampling*lbd_coeff))
        tilted_wf   = get_tilted_wavefront(input_wavefront, pup_shape[0], angle = 45.)
        tiptilt_form_det = np.exp(1j * tilted_wf*demipix_det)
    else:
        tiptilt_form_det = 1.

    if nocoro is not True:     ### perfect coronagraph: remove the coherent energy
        # option 1: direct computation of the mean
        e_c = np.sum(pupil_mask * input_wavefront) / np.sum(pupil_mask)

    #    # option 2: computation of the spatial variance of the phase
    #    input_phase = np.arctan2(np.imag(input_wavefront), np.real(input_wavefront))
    #    input_phase_mean = np.sum(input_phase*pupil_mask) / np.sum(pupil_mask)
    #    input_phase = input_phase - input_phase_mean
    #    pupil_index = np.where(pupil_mask>0)
    #    var = np.std(input_phase[pupil_index])**2
    #    e_c = np.exp(-var/2.)
    else :
        e_c = 0.

    before_lyot_stop = (input_wavefront - e_c)
    after_lyot_stop  = before_lyot_stop * lyot_mask
    after_lyot_stop2 = after_lyot_stop * tiptilt_form_det

    detector_img = np.abs(do_fftw_crosszp(after_lyot_stop2,
                                          zero_pad_factor = det_sampling*lbd_coeff,
                                          fov_pix = det_fov_pix))**2

    if lyot_output is True:
        final_output = {'img_detector': detector_img,
                        'img_before_lyot': np.abs(before_lyot_stop)**2,
                        'img_after_lyot': np.abs(after_lyot_stop)**2}
    else:
        final_output = {'img_detector': detector_img}

    return final_output

def choose_fft_function(dim, fov, sampling):
    """
    Attempt for a choosing the fastest FFT function depending on the grid,
    FoV and sampling...
    """

    total_dim = dim * sampling

    if fov <= 128 :
        fft_function = matrixDFT.matrix_dft
    elif total_dim <= 4096 :
        fft_function = pyfftw.interfaces.numpy_fft.fft2
    elif fov < 512 and total_dim <= 1024*11 :
        fft_function = do_fftw_crosszp
    elif fov >= 512 and sampling < 10 :
        fft_function = pyfftw.interfaces.numpy_fft.fft2
    else:
        fft_function = 'unreasonable computation grids'

    return fft_function