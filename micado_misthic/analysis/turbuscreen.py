# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from poppy import matrixDFT

from ..imgproc import get_frame_center
from ..imgproc import do_magnify
from ..simu import do_fftw_crosszp

def magnification_factor(pup1, pup2, mag_range, n_mag, 
                         centering='FFTSTYLE', display=False):
    shape1 = pup1.shape
    (cy1,cx1) = get_frame_center(shape1)
    
    shape2 = pup2.shape
    (cy2,cx2) = get_frame_center(shape2)
    
    # Determining the scaling factor
    
    mag_tab = np.linspace(mag_range[0], mag_range[1], n_mag)
    sub_tab = np.zeros(n_mag)
    sub_tab_abs = np.zeros(n_mag)
    area2_tab = np.zeros(n_mag)
    
    for i in range(n_mag) :
        newpup2    = do_magnify(pup2, mag_tab[i], output_shape=shape1, 
                             interp_deg=1, centering=centering)
        # keep only 0/1 values:
        newpup2b   = np.where(newpup2 > 0.5, 1, 0)
        
        # metric 1: measures the correlation
        sub_tab_abs[i] = np.sum(np.abs(newpup2b-pup1))
        sub_tab[i]     = np.sum(newpup2b-pup1)
        
        # metric 2: measures the total area and then compares it with pup1 area
        area2_tab[i] = np.sum(newpup2b)
    
    area1       = np.sum(pup1)
    area2_tab   = np.abs(area2_tab - area1)
    
    # metric 1
    ind_optimal1 = np.where(area2_tab == np.min(area2_tab))
    final_mag1   = mag_tab[ind_optimal1]
    
    if len(final_mag1) > 1:
        mag1 = np.mean(final_mag1)
    else :
        mag1 = final_mag1[0]
    
    # metric 2
    ind_optimal2 = np.where(sub_tab == np.min(sub_tab))
    final_mag2 = mag_tab[ind_optimal2]
    if len(final_mag2) > 1:
        mag2 = np.mean(final_mag2)
    else :
        mag2 = final_mag2[0]
    
    # metric 3: combined
    ind_opt_bis = np.where(sub_tab > 0.)
    mag_tab_bis = mag_tab[ind_opt_bis]
    sub_tab_abs_bis = sub_tab_abs[ind_opt_bis]
    ind_opt = np.where(sub_tab_abs_bis == np.min(sub_tab_abs_bis))
    final_mag3   = mag_tab_bis[ind_opt]
    
    if len(final_mag3) > 1:
        mag3 = np.mean(final_mag3)
    else :
        mag3 = final_mag3[0]
    
    if display is True :        
        print('\nMETRIC 1: [pupil area] minimizes abs(sum(pup1)-sum(pup2))')
        print(final_mag1, ind_optimal1, np.min(area2_tab))
        print('final mag 1 = ', mag1)
        
        print('\nMETRIC 2: [correlation] minimizes sum(abs(pup1-pup2))')
        print(final_mag2, ind_optimal2, np.min(sub_tab))
        print('final mag 2 = ', mag2)
        
        print(final_mag3)
        print('\nfinal mag 3 = ', mag3)
        
        plt.figure(num=1, figsize=(5,5))
        plt.clf()
        plt.plot(mag_tab, area2_tab, label='m1 = abs(sum(pup2)-sum(pup1))')
        plt.plot(mag_tab, sub_tab_abs, label='m2 = sum(abs(pup2-pup1))')
        plt.plot(mag_tab, sub_tab, label='m2b = sum(pup2-pup1)')
        plt.plot(mag_tab_bis[ind_opt], sub_tab_abs_bis[ind_opt], 'o', label='optimal')
        plt.legend()
        
        newpup2  = do_magnify(pup2, mag3, output_shape=shape1)
        newpup2b = np.where(newpup2 > 0.5, 1, 0)
        
        plt.figure(num=2, figsize=(5,5))
        plt.clf()
        plt.imshow(newpup2b-pup1)
        plt.title('Final pupil subtraction pup2-pup1')
        
    final_mag = mag3
    
    return final_mag


def estimate_strehl_ratio(opd_screen, lbd, pup, full_output=False,
                          fov=2, npix=10):
    
    s_opd = opd_screen.shape
    if len(s_opd) == 3:
        n_opd = s_opd[0]
    else:
        n_opd = 1
        opd_screen = np.expand_dims(opd_screen, 0)
    
    if full_output is False:
        strehl_tab    = np.empty((n_opd))
    else : 
        strehl_tab    = np.empty((n_opd, 2))
        pup_area      = np.where(pup > .5)

    PSF_nocoro = np.abs(matrixDFT.matrix_dft(pup,fov,npix,centering='FFTSTYLE'))**2
    max_nocoro = np.max(PSF_nocoro)

    for i in range(n_opd):
        
        if ((i/100.-i//100) == 0.) :
            print(i, '/', n_opd)
            
        phase = opd_screen[i] * 2*np.pi / lbd
        PSF_coro = np.abs(matrixDFT.matrix_dft(np.exp(1j * phase) * pup,
                                               fov, npix, 
                                               centering='FFTSTYLE'))**2
    
        if full_output is False :
            strehl_tab[i] = np.max(PSF_coro) / max_nocoro
        else :
            strehl_tab[i,0] = np.max(PSF_coro) / max_nocoro
            strehl_tab[i,1] = np.std((phase)[pup_area])

    return strehl_tab


def estimate_psd(opd_screen, lbd):
    
    s_opd = opd_screen.shape
    if len(s_opd) == 3:
        n_opd = s_opd[0]
    else:
        n_opd = 1
        opd_screen = np.expand_dims(opd_screen, 0)
    
    mean_psd = np.zeros_like(opd_screen[0])
    
    for i, opd_i in enumerate(opd_screen):
        
        mean_psd = mean_psd + np.fft.fftshift((np.abs(np.fft.fft2(opd_i))**2)) / n_opd
    
    return mean_psd
        


def compute_psf_mft(opd_screen, lbd, pup, 
                fov = 5, npix=25):
    
    phase = opd_screen * 2*np.pi / lbd
    
    PSF = np.abs(matrixDFT.matrix_dft(np.exp(1j * phase) * pup,
                                               fov, npix, 
                                               centering='FFTSTYLE'))**2
    
    return PSF

def compute_psf_fft(opd_screen, lbd, pup, 
                    sampling=5, fov=None):
    
    phase = opd_screen * 2*np.pi / lbd
    
    if fov is None:
        fov = 512
    
    PSF = np.abs(do_fftw_crosszp(np.exp(1j * phase) * pup,
                               zero_pad_factor = sampling, fov=fov))**2

    return PSF