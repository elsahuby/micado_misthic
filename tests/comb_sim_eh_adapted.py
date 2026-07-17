#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:59:28 2022

@author: hbaran
Script adapted by EHuby
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import interpolate
import os
from astropy import units as u
import glob
from astropy.table import Table

from micado_misthic.utils.obsparams import * 
from micado_misthic.imgproc import get_circle_mask
from micado_misthic.imgproc import fft_resize
from micado_misthic.imgproc import micado_adi

from micado_misthic.analysis import get_rms_contrast
from scipy.signal import fftconvolve

#plt.style.use('seaborn-muted')
#plt.style.use('viridis')


### Simulation Parameters
tel_diam        = 38.542 #m

###-- Wavelength table
#lbd             = [1.190, 1.270, 1.582, 1.693, 2.100, 2.235] #, 1.245, 1.635, 2.145]#, 1.635, 1.693] # in um
#lbd             = [1.600] #, 1.245, 1.635, 2.145]#, 1.635, 1.693] # in um
#lbd             = [1.245, 1.582, 2.100, 2.145]
lbd = [1.245, 1.635, 2.145] # central wavelength of J, H, K bands in um


###-- Observation params
coronagraph     = 'CLC1' # 'CLC0' or 'CLC1'
quartile        = 'MED'#['Q1', 'Q4', 'MED']
frame_exp_time  = 10. ## exposure time per frame in seconds ### NEW
# obs_time = 1800 # total obs time in seconds
ron             = 15 # electrons; from FDR paper
pixscale_in_mas = 1.5 # 1.5 or 4 mas
sampling_misthic = 7

# Switches
save            = False
coro_on         = True
planet_on       = False

###-- Planet Parameters
# CLC0 = [15, 30, 50, 75, 100, 125, 150, 200, 250, 300]
# CLC0 = [50, 75, 100, 200, 300]

# CLC1_J = [30, 50, 75, 100, 150, 200, 250, 500]
#CLC1_H = [30, 50, 75, 100, 150, 200, 250, 500, 700, 1000]
# CLC1_K = [30, 50, 75, 100, 150, 200, 250, 500]
# CLC1_K = [150, 200, 250, 500]
#pl_dist = [30, 100, 200] #CLC1_H #CLC0 # in mas
#pl_dist=[15, 30, 50 75]#, 100, 125, 150, 300]
# pl_dist = [50] #, 100, 200, 500]
# pl_dist = [6.600]
pl_dist = [50.000]

###-- Simulation duration
sim_duration = [30.0] # in minutes

###-- Star Parameters

## HR8799 ###################################################################
star_mag = [5.383, 5.28, 5.240] # magnitudes respectively in J, H, K # formerly 5.24 in K (!!! catalogue says fluxes; check that !!!!!)
star_name = 'HR8799_'
zenith_distance = np.abs(-24.5 - (21.1)) # deg
dist_pc = 39.4 # distance in parsecs
airmass = 1./np.cos(zenith_distance*np.pi/180.)
r_pl    = 1 # jupiter masses
temp    = 400 # Kelvin
logg    = 4.
met     = .32
CO      = 0.1

### Eps Eridani ##############################################################
# star_mag = [ 2.23, 1.75, 1.67] #formerly 1.75 in H
# star_name = 'EpsEri'
# zenith_distance = np.abs(-24.5 - (-9-27/60.))
# dist_pc = 3.2 # distance in parsecs
# airmass = 1. #/np.cos(zenith_distance*np.pi/180.)
# r_pl    = 1 # jupiter masses
# temp    = 1500 # Kelvin
# logg    = 4.5
# met     = 1.00
# CO      = 0.5

### Beta Pictoris ############################################################
# star_mag = [3.57, 3.51, 3.48] #formerly 2.41 
# star_name = 'Beta Pic'
# zenith_distance = np.abs(-24.5 - (21.1)) # deg
# dist_pc = 19.3 # distance in parsecs
# airmass = 1./np.cos(zenith_distance*np.pi/180.)
# r_pl    = 1 # jupiter masses
# temp    = 400 # Kelvin
# logg    = 4.
# met     = .32
# CO      = 0.1

# Planet magnitude: 
# planet_mag = 23


###-- Directories
### directory with flux & transmission data for MICADO
flux_dir                    = 'C:/Users/Red Slottje/Documents/Misthic_inputs/Flux_input/'

### directory with simulated spectrum data
spec_dir                    = 'C:/Users/Red Slottje/Documents/Misthic_inputs/R500_cloudless_2025/'
### directories with the coronagraphic simulated images
directory_prefix                 = f'D:/FROM_RED/'

coro_sim_dir_prefix         = directory_prefix+f'coro_sims/{coronagraph}_{quartile}_NoNCPA/'
planet_sim_dir_prefix       = directory_prefix+f'planet_sims/{coronagraph}_{quartile}_NoNCPA/'
coro_resized_dir_prefix     = directory_prefix+f'resized_fft_coro/{coronagraph}_{quartile}_noNCPA/'
#coro_resized_dir_prefix     = f'D:/FROM_NELLIE/resized_fft/{coronagraph}_{quartile}_noNCPA/'
planet_resized_dir_prefix   = directory_prefix+f'resized_fft_planet/{coronagraph}_{quartile}_noNCPA/'
### directories where the resulting ADI images are saved
adi_img_dir_prefix          = directory_prefix+f'coro_adi_new/{star_name}/{coronagraph}_{quartile}_noNCPA/'
pladi_img_dir_prefix        = directory_prefix+f'planet_adi_new/{star_name}/{coronagraph}_{quartile}_noNCPA/'

### telescope pupil file, needed to normalize the input flux
telescope_pupil_file        = f'D:/FROM_NELLIE/INPUT_fromNellie/PUPIL/Pupil_ELT_v03.fits'

all_pl_max = []

### number of images in the cube
#n_images = 4

### telescope latitude, needed to calculate the parallactic angle
tel_latitude = -24.5

# Calculating parallactic angle table, one angle for each image in the cube
'''ha_hours      = (np.arange(n_images) - (n_images-1.)/2. ) * frame_exp_time / 3600.
dec_deg       = (tel_latitude - zenith_distance)
parangle_tab  = get_parallactic_angle(ha_hours, dec_deg, tel_latitude)

if not os.path.exists(parallactic_angle_dir+f'parangle_tab_{n_images}_imagespercube.txt'):
    t = Table([parangle_tab],names=('a'))
    t.write(parallactic_angle_dir+f'parangle_tab_{n_images}_imagespercube.txt', format='ascii')'''

#else :
#    parallactic_angle_dir = parallactic_angle_dir+f'parangle_tab_{n_images}_imagespercube.txt'
# {pl_dist[i]:5.3f}
for i in range(len(lbd)):
    #coro_sim_dir   = coro_sim_dir_prefix + f'{coronagraph}_25Hz_NEW_Lambda={lbd[i]:5.3f}/'
    coro_sim_dir = coro_sim_dir_prefix +f'lambda={lbd[i]:5.3f}/p_dist_mas={pl_dist[0]:5.3f}/TURBU_ABERR/duree={sim_duration[0]}min/'
    planet_sim_dir = planet_sim_dir_prefix+ f'lambda={lbd[i]:5.3f}/p_dist_mas={pl_dist[0]:5.3f}/TURBU_ABERR/duree={sim_duration[0]}min/'
    #parallactic_angle_dir       = coro_sim_dir_prefix+f'lambda={lbd[i]:5.3f}/p_dist_mas={pl_dist[i]:5.3f}/' #lbd[i]:5.3f
    parallactic_angle_dir        = coro_sim_dir
    print ('planet_sim_dir =', planet_sim_dir)
    print('parallactic_angle_dir=', parallactic_angle_dir)
    
    coro_resized_dir = coro_resized_dir_prefix + f'lbd={lbd[i]:5.3f}_pxscale={pixscale_in_mas}/'
    planet_resized_dir = planet_resized_dir_prefix + f'lbd={lbd[i]:5.3f}_pxscale={pixscale_in_mas}/'

    # =============================================================================
    # CORO IMAGE PROCESSING
    # =============================================================================

    if coro_on == True:
    #     # -----------------------------------------------------------------------------
    #     # If desired resizing directory doesn't exist, create it. If it's empty, resize the files.
    #     if not os.path.exists(coro_resized_dir):
    #         os.makedirs(coro_resized_dir)
    #     file_list_coro = os.listdir(coro_resized_dir)
        
    #     if len(file_list_coro) == 0:
    #         print('-- Coro Img: Resizing')
    #         # If directory is empty, do coro resizing
    #         perf_file = [f for f in os.listdir(coro_sim_dir) if f.endswith('_perfect_psf.fits')][0]
    #         HDU_perf = fits.open(coro_sim_dir+perf_file)
    #         perf_psf = HDU_perf[0].data
            
    #         psf_file = [f for f in os.listdir(coro_sim_dir) if f.endswith('_psf_cube.fits')][0]
    #         HDU_psf = fits.open(coro_sim_dir+psf_file)
    #         cube_psf = HDU_psf[0].data
            
    #         coro_file = [f for f in os.listdir(coro_sim_dir) if f.endswith('_image_cube.fits')][0]
    #         HDU_coro = fits.open(coro_sim_dir+coro_file)
    #         cube_coro = HDU_coro[0].data
            
    #         coro_cube, coro_scale_coeff     = fft_resize(cube_coro, lbd[i], sampling_misthic, pixscale_in_mas, write_dir=coro_resized_dir+'image_cube_',  plotting=False)
    #         psf_cube,  psf_scale_coeff      = fft_resize(cube_psf,  lbd[i], sampling_misthic, pixscale_in_mas, write_dir=coro_resized_dir+'psf_cube_',    plotting=False)
    #         perf_psf,  perf_psf_scale_coeff = fft_resize(perf_psf,  lbd[i], sampling_misthic, pixscale_in_mas, write_dir=coro_resized_dir+'perfect_psf_', plotting=False)
    #     else:
    #         # Opening resized files
    #         perf_psf    = fits.getdata(coro_resized_dir+'perfect_psf_fft_resized.fits')            
    #         psf_cube    = fits.getdata(coro_resized_dir+'psf_cube_fft_resized.fits')            
    #         coro_cube   = fits.getdata(coro_resized_dir+'image_cube_fft_resized.fits')
        
        # perf_psf    = fits.getdata(coro_resized_dir+'perfect_psf_fft_resized.fits')            
        # psf_cube    = fits.getdata(coro_resized_dir+'psf_cube_fft_resized.fits')            
        # coro_cube   = fits.getdata(coro_resized_dir+'image_cube_fft_resized.fits')

        open_perf_psf = [f for f in os.listdir(coro_sim_dir) if f.endswith('_perfect_psf.fits')][0] 
        perf_psf    = fits.getdata(coro_sim_dir+open_perf_psf)

        open_image_cube   = [f for f in os.listdir(coro_sim_dir) if f.endswith('_image_cube.fits')][0]            
        psf_cube    = fits.getdata(coro_sim_dir+open_image_cube)         

        open_psf   = [f for f in os.listdir(coro_sim_dir) if f.endswith('_psf_cube.fits')][0]
        coro_cube   = fits.getdata(coro_sim_dir+open_psf)

        # Check if the ADI coro and psf images exist
        adi_img_dir = adi_img_dir_prefix + f'lbd={lbd[i]:5.3f}_pxscale={pixscale_in_mas}/duree={sim_duration[0]}/'
        coro_adi_file = adi_img_dir+f'{star_name}_coro_adi_{star_mag[i]}mag_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec.fits'
        psf_adi_file = adi_img_dir+f'{star_name}_psf_adi_{star_mag[i]}mag_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec.fits'
        coro_adi_conv_file = adi_img_dir+f'{star_name}_coro_adi_conv_{star_mag[i]}mag_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec.fits'
        psf_adi_conv_file = adi_img_dir+f'{star_name}_psf_adi_conv_{star_mag[i]}mag_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec.fits'
        
        if len(glob.glob(coro_adi_file))==1 :
            print('-- Coro Img: LOAD ADI images for CORO and PSF')
            print(coro_adi_conv_file)
            print(psf_adi_conv_file)
            adi_sum_coro = fits.getdata(coro_adi_file)
            psf_sum_coro = fits.getdata(psf_adi_file)
            conv_adi_coro = fits.getdata(coro_adi_conv_file)
            conv_psf_coro = fits.getdata(psf_adi_conv_file)
        else :
            
            # -----------------------------------------------------------------------------
            # # Flux and ADI
            # print("\n")
            # print("====================================================================")
            # print("CORO {}, Mag {}, Lambda = {:5.3f}um, Seeing {}".format(coronagraph,star_mag,lbd[i],quartile))
            # print("--------------------------------------------------------------------")
          
            # # Calculating coro flux in PSF
            # flux_psf, emission_tot, trans_out = simp_micado_flux(flux_dir, star_mag, lbd[i], 
            #                                                       filter_type='{:5.3f}'.format(lbd[i]), 
            #                                                       obs_time=frame_exp_time, airmass=1., 
            #                                                       pixscale=pixscale_in_mas,planet=False)
            # print("Photons reaching detector in perf system (coro)  : {:e}".format(flux_psf))
           
            ### NEW ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
            # Star Spectrum
            _, star_flux = get_star_spectrum(flux_dir, star_mag[i], lbd[i])
            
            # Aperture Surface
            telescope_surface = get_aperture_surface(telescope_pupil_file)
            print(f'lambda = {lbd[0]}')
            photon_flux, emission_per_pix, global_transmission = get_micado_flux(flux_dir, 
                                                                                 star_flux, f'{lbd[0]:5.3f}', 
                                                                                 frame_exp_time, telescope_surface, 
                                                                                 airmass=airmass, pixel_scale=pixscale_in_mas)
            print("#NEW# Photons reaching detector in perf system (coro)  : {:e}".format(photon_flux))
            ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

            perf_psf_sum = np.sum(perf_psf)
            
            ### NEW ###
            psf_cube_noisy, perf_psf_noisy, flux_per_frame = scale_to_photons(psf_cube, perf_psf, 
                                                                                photon_flux, emission_per_pix, 
                                                                                frame_exp_time=frame_exp_time, sig_ron = 15., 
                                                                                no_noise=False)
            
            coro_cube_noisy, perf_psf_noisy, flux_per_frame = scale_to_photons(coro_cube, perf_psf, 
                                                                                photon_flux, emission_per_pix, 
                                                                                frame_exp_time=frame_exp_time, sig_ron = 15., 
                                                                                no_noise=False)
          
            # plt.figure(num=22)
            # plt.clf()
            # fig22, ax22 = plt.subplots(nrow=1, ncols=2, sharey=True, sharex=True, num=22)
            # ax22[0].imshow(psf_cube_noisy[0])
            # ax22[1].imshow(coro_cube_noisy[0])

            print('psf_cube_noisy.shape=', psf_cube_noisy.shape)
            print('perf_psf_noisy.shape=', perf_psf_noisy.shape)
            print('psf_cube.shape=', psf_cube.shape)
            print('perf_psf.shape =', perf_psf.shape)
            print('coro_cube.shape=', coro_cube.shape)
           
            ### ADI
            print('----------')
            print("Beginning coro ADI processing")
            # adi_sum_coro, psf_sum_coro = micado_adi(img_noise_coro, psf_noise_coro, coro_sim_dir)
            adi_sum_coro, psf_sum_coro = micado_adi(coro_cube_noisy, psf_cube_noisy, parallactic_angle_dir)
            #adi_sum_coro, psf_sum_coro = micado_adi(coro_cube_noisy, psf_cube_noisy, coro_sim_dir)
        
            ### Convolution
            print('----------')
            print("Beginning coro image convolution")
            # dim = len(adi_sum_coro[0,:])
            #   = cheese_func(dim, lbd[i], r=0.5, px_scale=1.5)
            
            l_over_d_pix = lbd[i]*1e-6/tel_diam * 180./np.pi * 3600*1e3 / (pixscale_in_mas)
            mask_radius = l_over_d_pix * .5
            mask = get_circle_mask(adi_sum_coro.shape, mask_radius)

            print('mask coro.shape=', mask.shape)
            
            # conv_adi_coro = convolution(adi_sum_coro, mask)
            # conv_psf_coro = convolution(psf_sum_coro, mask)
            
            y_dim, x_dim = adi_sum_coro.shape
            conv_adi_coro = fftconvolve(adi_sum_coro, mask, mode='full')
            conv_psf_coro = fftconvolve(psf_sum_coro, mask, mode='full')
    
            y_dim2, x_dim2 = conv_adi_coro.shape
            y1 = int((y_dim2-y_dim+1)/2)
            y2 = y1 + y_dim
            conv_adi_coro = conv_adi_coro[y1:y2,y1:y2]
            conv_psf_coro = conv_psf_coro[y1:y2,y1:y2]
        
            ### SAVE #######################################################
            if not os.path.exists(adi_img_dir):
                os.makedirs(adi_img_dir)
            fits.writeto(coro_adi_file, adi_sum_coro)
            fits.writeto(psf_adi_file, psf_sum_coro)
            fits.writeto(coro_adi_conv_file, conv_adi_coro)
            fits.writeto(psf_adi_conv_file, conv_psf_coro)
        
        max_psf_coro = np.max(conv_psf_coro)
        norm_adi_coro = conv_adi_coro / max_psf_coro    
    
        rms_contrast, x = get_rms_contrast(conv_adi_coro)
        rms_contrast = 5.*rms_contrast/max_psf_coro
        x *= pixscale_in_mas

        plt.plot(x, rms_contrast)#, label="Lambda={:5.3f} um".format(lbd[i]))
        
        plt.suptitle(f"{star_name}: {coronagraph}, {quartile} seeing, lambda={lbd[i]:5.3f}um, pxscale={pixscale_in_mas}mas, Mag={star_mag[i]}",fontsize=12)
        #plt.title(r'Planet: {} $R_J$, {} pc, {}K, logg={}, met={:.2f}, CO ratio={:3.2f}'.format(r_pl, dist_pc, temp, logg, met, CO),fontsize=10)
        plt.xlabel("Distance from center (mas)")
        plt.ylabel("Contrast, 5-sigma")
        plt.yscale('log')
        plt.grid(color='.9')
        plt.xlim(0,)
    #{temp}K_logg{logg}_met{met:.2f}_CO{CO:3.2f}_sep{pl_dist}mas_
        save_file = f'{star_name}_no_planet_adi_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec_lbd{lbd[i]:5.3f}'
        plt.savefig(adi_img_dir+save_file+'_contrast.png', dpi=120, bbox_inches='tight')
        plt.show()
    
    # =============================================================================
    # PLANET IMAGE PROCESSING
    # =============================================================================
    
    if planet_on == True:

        # -----------------------------------------------------------------------------
        # ### Applying EXOREM Spectra
        # ### (only if spectra is the input and not magnitude)
        # spec_file = np.loadtxt(spec_dir+'spectra_YGP_{}K_logg{}_met{:.2f}_CO{:3.2f}.dat'.format(temp,logg,met,CO), unpack=True)
        
        # HDU_wave_tr = fits.open(flux_dir+'Wavelength_for_emission_transmission_in_micrometer.fits')
        # wave_tr     = HDU_wave_tr[0].data
        # step = np.round(wave_tr[1]-wave_tr[0], 3) * u.micron
        
        # spec_flux = spec_file[1,:] # W*m^-2 / cm^-1
        # wave_num  = spec_file[0,:] # cm^-1
        # k_step = wave_num[1]-wave_num[0]
        
        # wavelength = 10000 / wave_num
        # lam_spec = spec_flux * wave_num**2 / 10000
          
        # f = interpolate.interp1d(wavelength,lam_spec, kind='linear',fill_value="extrapolate")
        # spec_interp = f(wave_tr)
        # spec_interp[spec_interp < 0] = 0
    
        # -----------------------------------------------------------------------------


        ### NEW ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
        # Planet Spectrum
        planet_flux, spec_wave = get_planet_spectrum(spec_dir, temp, logg, met, CO, dist_pc, r_pl, flux_dir=flux_dir)
        # Aperture Surface
        telescope_surface = get_aperture_surface(telescope_pupil_file)
        planet_photon_flux, planet_emission_per_pix, planet_global_transmission = get_micado_flux(flux_dir, 
                                                                             planet_flux, f'{lbd[0]:5.3f}', 
                                                                             frame_exp_time, telescope_surface, 
                                                                             airmass=airmass, pixel_scale=pixscale_in_mas)
        print("#NEW# Photons reaching detector in perf system (planet)  : {:e}".format(planet_photon_flux))
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
        
        max_pl_list = np.zeros(len(pl_dist))
        log_max_pl_list = np.zeros(len(pl_dist))
        for j in range(len(pl_dist)):
            print("\n")
            print("====================================================================")
            print("PLANET (Lam={:5.3f}, Distance from Star, mas: {})".format(lbd[i], pl_dist[j]))
            print("--------------------------------------------------------------------")
            
            ### check if the ADI cube exists
            pladi_img_dir = pladi_img_dir_prefix + f'lbd={lbd[i]:5.3f}_pxscale={pixscale_in_mas}/duree={sim_duration[0]}/'
            pl_adi_file = pladi_img_dir+f'{star_name}_planet_adi_{temp}K_logg{logg}_met{met:.2f}_CO{CO:3.2f}_sep{pl_dist[j]}mas_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec.fits'
            plpsf_adi_file = pladi_img_dir+f'{star_name}_planet-psf_adi_{temp}K_logg{logg}_met{met:.2f}_CO{CO:3.2f}_sep{pl_dist[j]}mas_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec.fits'
            pl_adi_conv_file = pladi_img_dir+f'{star_name}_coro_adi_conv_{temp}K_logg{logg}_met{met:.2f}_CO{CO:3.2f}_sep{pl_dist[j]}mas_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec.fits'
            plpsf_adi_conv_file = pladi_img_dir+f'{star_name}_psf_adi_conv_{temp}K_logg{logg}_met{met:.2f}_CO{CO:3.2f}_sep{pl_dist[j]}mas_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec.fits'

            if len(glob.glob(pl_adi_file))==1 :
                print('-- Coro Img: LOAD ADI images for PLANET')
                adi_sum_pl = fits.getdata(pl_adi_file)
                psf_sum_pl = fits.getdata(plpsf_adi_file)
                conv_adi_pl = fits.getdata(pl_adi_conv_file)
                conv_psf_pl = fits.getdata(plpsf_adi_conv_file)

            else :
        
                # Simulation directories:
                # pl_sim_dir = planet_sim_dir #+'p_dist_mas={}/'.format(pl_dist[j])
                # print('pl_sim_dir=', pl_sim_dir)
                # pl_resized_dir = planet_resized_dir+'p_dist_mas={}/'.format(pl_dist[j])
                # # If desired resizing directory doesn't exist, create it. If it's empty, resize the files.
                # if not os.path.exists(pl_resized_dir):
                #     os.makedirs(pl_resized_dir)
                # file_list_planet = os.listdir(pl_resized_dir)
                
                # if len(file_list_planet) == 0:
                # # If directory is empty, do planet resizing
                #     perf_file_pl = glob.glob(pl_sim_dir+'*_perfect_psf.fits')
                #     perf_psf_pl = fits.getdata(perf_file_pl[0])
                    
                #     psf_file_pl = glob.glob(pl_sim_dir+'*_psf_cube.fits')
                #     cube_psf_pl = fits.getdata(psf_file_pl[0])
                    
                #     planet_file = glob.glob(pl_sim_dir+'*_image_cube.fits')
                #     planet_cube = fits.getdata(planet_file[0])
                    
                #     planet_cube, planet_scale_coeff_pl   = fft_resize(planet_cube, lbd[i], sampling_misthic, pixscale_in_mas, write_dir=pl_resized_dir+'image_cube_',  plotting=False)
                #     cube_psf_pl, psf_scale_coeff_pl      = fft_resize(cube_psf_pl, lbd[i], sampling_misthic, pixscale_in_mas, write_dir=pl_resized_dir+'psf_cube_',    plotting=False)
                #     perf_psf_pl, perf_psf_scale_coeff_pl = fft_resize(perf_psf_pl, lbd[i], sampling_misthic, pixscale_in_mas, write_dir=pl_resized_dir+'perfect_psf_', plotting=False)
                # else:
                #     # Opening resized files
                #     perf_psf_pl = fits.getdata(pl_resized_dir+'perfect_psf_fft_resized.fits')
                #     cube_psf_pl = fits.getdata(pl_resized_dir+'psf_cube_fft_resized.fits')
                #     planet_cube = fits.getdata(pl_resized_dir+'image_cube_fft_resized.fits')
                
                # perf_psf_pl    = fits.getdata(planet_sim_dir+'_occulter_perfect_psf.fits')            
                # cube_psf_pl    = fits.getdata(planet_sim_dir+'_occulter_psf_cube.fits')            
                # planet_cube   = fits.getdata(planet_sim_dir+'_occulter_image_cube.fits')

                pl_sim_dir = planet_sim_dir
                open_perf_psf_pl = [f for f in os.listdir(planet_sim_dir) if f.endswith('_perfect_psf.fits')][0] 
                perf_psf_pl    = fits.getdata(planet_sim_dir+open_perf_psf_pl)

                open_planet_cube   = [f for f in os.listdir(planet_sim_dir) if f.endswith('_image_cube.fits')][0]            
                planet_cube    = fits.getdata(planet_sim_dir+open_planet_cube)         

                open_psf_pl   = [f for f in os.listdir(planet_sim_dir) if f.endswith('_psf_cube.fits')][0]
                cube_psf_pl   = fits.getdata(planet_sim_dir+open_psf_pl)
                
                # # Calculating planet flux in PSF
                # # print("Filter before flux: {:5.3f}".format(lbd[i]))
                # flux_psf_pl, emission_tot_pl, trans_out_pl = simp_micado_flux(flux_dir, star_mag, lbd[i], 
                #                                                               filter_type='{:5.3f}'.format(lbd[i]), 
                #                                                               obs_time=10., planet_spec=spec_interp,
                #                                                               airmass=airmass, pixscale=pixscale_in_mas,
                #                                                               r_planet=r_pl,dist_pc=dist_pc,planet=True)

                # perf_psf_sum_pl = np.sum(perf_psf_pl)
                
                ### NEW ###
                planet_cube_noisy, perf_psf_noisy, flux_per_frame = scale_to_photons(planet_cube, perf_psf_pl, 
                                                                                    planet_photon_flux, planet_emission_per_pix, 
                                                                                    frame_exp_time=frame_exp_time, sig_ron = ron, 
                                                                                    no_noise=False)
                plpsf_cube_noisy, perf_psf_noisy, flux_per_frame = scale_to_photons(cube_psf_pl, perf_psf_pl, 
                                                                                    planet_photon_flux, planet_emission_per_pix, 
                                                                                    frame_exp_time=frame_exp_time, sig_ron = ron, 
                                                                                    no_noise=False)

                print('planet_cube_noisy.shape=', planet_cube_noisy.shape)
                print('perf_psf_noisy.shape=', perf_psf_noisy.shape)
                print('planet_cube.shape=', planet_cube.shape)
                print('perf_psf_pl.shape =', perf_psf_pl.shape)

                # ADI
                print('----------')
                print("Beginning planet ADI processing")
                adi_sum_pl, psf_sum_pl = micado_adi(planet_cube_noisy, plpsf_cube_noisy, pl_sim_dir)
                
                # Convolution
                print('----------')
                print("Beginning planet image convolution")
                    
                l_over_d_pix = lbd[i]*1e-6/tel_diam * 180./np.pi * 3600*1e3 / (pixscale_in_mas)
                mask_radius = l_over_d_pix * .5
                mask = get_circle_mask(adi_sum_pl.shape, mask_radius)
                
                print('mask planet.shape=', mask.shape)
                y_dim, x_dim = adi_sum_pl.shape
                
                conv_adi_pl = fftconvolve(adi_sum_pl, mask, mode='full')
                conv_psf_pl = fftconvolve(psf_sum_pl, mask, mode='full')
    
                
                y_dim2, x_dim2 = conv_adi_pl.shape
                y1 = int((y_dim2-y_dim+1)/2)
                y2 = y1 + y_dim
                
                conv_adi_pl = conv_adi_pl[y1:y2,y1:y2]
                conv_psf_pl = conv_psf_pl[y1:y2,y1:y2]       
                
                ### SAVE
                if not os.path.exists(pladi_img_dir):
                    os.makedirs(pladi_img_dir)
                fits.writeto(pl_adi_file, adi_sum_pl)
                fits.writeto(plpsf_adi_file, psf_sum_pl)
                fits.writeto(pl_adi_conv_file, conv_adi_pl)
                fits.writeto(plpsf_adi_conv_file, conv_psf_pl)
            
            norm_adi_pl = conv_adi_pl / max_psf_coro 
            # norm_adi_pl = conv_adi_pl0 / max_psf_coro 

            
            max_psf_pl = np.max(conv_psf_pl)

            print("ADI planet max:", np.max(adi_sum_pl))
            print("Total flux in ADI img:", np.sum(adi_sum_pl))
            
            
            # Max val (planet)
            max_planet = np.max(conv_adi_pl) / max_psf_coro
            print("----------")
            print("Max of planet PSF:", max_psf_pl)
            print("Max of planet ADI img (not normalized):", np.max(conv_adi_pl))
            print("Max of planet ADI image (normalized by coro PSF): ", max_planet)
            print("Total flux in planet ADI img:", np.sum(adi_sum_pl))
               
            
            max_pl_list[j] = max_planet
            log_max_pl_list[j] = np.log10(max_planet)
        
            
            plt.figure(num=30+i)
            plt.clf()
            pixscale = pixscale_in_mas/1000.
            ny, nx = conv_adi_coro.shape
            e = [-ny/2*pixscale, nx/2*pixscale, -ny/2*pixscale, nx/2*pixscale]
            plt.imshow(conv_adi_coro+conv_adi_pl, cmap="afmhot", origin='lower', 
                       extent = e, vmin=conv_adi_coro.min()/5, vmax=conv_adi_coro.max()/5.)
            plt.suptitle(f"{star_name}: {coronagraph} {quartile} ADI image at lambda={lbd[i]:5.3f}, pxscale={pixscale_in_mas}, Mag {star_mag[i]}",fontsize=12)
            plt.title('Planet: {} R_Jup, {} pc, {}K, logg={}, met={:.2f}, CO ratio={:3.2f}'.format(r_pl, dist_pc, temp, logg, met, CO),fontsize=10)
            plt.xlabel('[arcsec]')
            plt.ylabel('[arcsec]')
            plt.xlim(-.7,.7)
            plt.ylim(-.7,.7)
            
            save_file = f'{star_name}_planet_adi_{temp}K_logg{logg}_met{met:.2f}_CO{CO:3.2f}_sep{pl_dist[j]}mas_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec'
            plt.savefig(adi_img_dir+save_file+'_ADI_img.png', dpi=120, bbox_inches='tight')
            
            plt.figure(num=40+i)
            plt.clf()
            plt.plot(x, rms_contrast)#, label="Lambda={:5.3f} um".format(lbd[i]))
            plt.plot(pl_dist[j], max_planet, 'o')
            
            plt.suptitle(f"{star_name}: {coronagraph}, {quartile} seeing, lambda={lbd[i]:5.3f}um, pxscale={pixscale_in_mas}mas, Mag={star_mag[i]}",fontsize=12)
            plt.title(r'Planet: {} $R_J$, {} pc, {}K, logg={}, met={:.2f}, CO ratio={:3.2f}'.format(r_pl, dist_pc, temp, logg, met, CO),fontsize=10)
            plt.xlabel("Distance from center (mas)")
            plt.ylabel("Contrast, 5-sigma")
            plt.yscale('log')
            plt.grid(color='.9')
            plt.xlim(0,)
            
            save_file = f'{star_name}_planet_adi_{temp}K_logg{logg}_met{met:.2f}_CO{CO:3.2f}_sep{pl_dist[j]}mas_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec'
            plt.savefig(adi_img_dir+save_file+'_contrast.png', dpi=120, bbox_inches='tight')
    
    if coro_on == True:
        # plt.legend()
        plt.show()
        
        if save == True:
            # Save figure and coro adi images
            # fig_dir = "/media/hbaran/COROMICADO/OUTPUT/coro_rad_profs_new/{}_{}_noNCPA/lbd={:5.3f}_pxscale={}/".format(coronagraph,quartile,lbd,pixscale_in_mas)
            #fig_dir = "./"
            fig_dir = adi_img_dir
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)  
            plt.savefig(fig_dir+"contrast_plot_rad{}Mjup_{}K_logg{}_met{:.2f}_CO{:3.2f}.png".format(r_pl,temp,logg,met,CO))#,overwrite=True)
        
            # coro_adi_dir = "/media/hbaran/COROMICADO/OUTPUT/coro_adi_new/{}_{}_noNCPA/lbd={:5.3f}_pxscale={}/".format(coronagraph,quartile,lbd,pixscale_in_mas)
            coro_adi_dir = adi_img_dir
            if not os.path.exists(coro_adi_dir):
                os.makedirs(coro_adi_dir)
                
            hdu_coro = fits.PrimaryHDU(norm_adi_coro)
            hdu_coro.writeto(coro_adi_dir + 'coro_adi_mag={}.fits'.format(star_mag[i]),overwrite=True)

#     max_pl_list = np.array(max_pl_list)
#     all_pl_max.append(max_pl_list)
    

# all_pl_max = np.array(all_pl_max)

#%% 

# for i in range(len(lbd)):
    
#     plt.figure(num=30+i)
#     plt.clf()
#     pixscale = pixscale_in_mas/1000.
#     ny, nx = conv_adi_coro.shape
#     e = [-ny/2*pixscale, nx/2*pixscale, -ny/2*pixscale, nx/2*pixscale]
#     plt.imshow(conv_adi_coro+conv_adi_pl, cmap="Blues_r", origin='lower', 
#                extent = e) #, vmin=-1e4, vmax=1e6)
#     plt.suptitle(f"{star_name}: {coronagraph} {quartile} ADI image at $\lambda$={lbd[i]:5.3f}, pxscale={pixscale_in_mas}, Mag {star_mag}",fontsize=12)
#     plt.title('Planet: {} R_Jup, {} pc, {}K, logg={}, met={:.2f}, CO ratio={:3.2f}'.format(r_pl, dist_pc, temp, logg, met, CO),fontsize=10)
#     plt.xlabel('[arcsec]')
#     plt.ylabel('[arcsec]')

#%%
    
# plt.figure()
# for i in range(len(lbd)):
#     # Radial profiles
      # noise_map, nomp_mean, std_prof = make_noise_map_no_mask(conv_adi_coro, max_psf_coro, delta_radii=0.5)
#     # noise_map_psf, nomp_mean_psf, std_prof_psf = make_noise_map_no_mask(conv_psf_coro, max_psf_coro, delta_radii=0.5)
        
#     # max_coro = np.max(conv_adi_coro) / max_psf_coro
#     # print("----------")
#     # print("Max of coro PSF:", max_psf_coro)
#     # print("Max of coro ADI img (not normalized):", np.max(conv_adi_coro))
#     # print("Max of coro ADI image (normalized by coro PSF): ", max_coro)
#     # print("Total flux in coro ADI img:", np.sum(adi_sum_coro))
            
#     # # Plotting radial profile
#    # x = (np.arange(len(std_prof))-2) * pixscale_in_mas     
#    # rms_contrast = 5*std_prof
    
#     rms_contrast, x = get_rms_contrast(conv_adi_coro)
#     rms_contrast = 5.*rms_contrast/max_psf_coro
#     x *= pixscale_in_mas
    

# #     plt.clf()
#     plt.plot(x, rms_contrast, label="Lambda={:5.3f} um".format(lbd[i]))
# plt.show()
    
#     plt.suptitle(f"{star_name}: {coronagraph} {quartile} Contrast Profile of $\lambda$={lbd[i]:5.3f}, pxscale={pixscale_in_mas}, Mag {star_mag}",fontsize=12)
#     plt.title('Planet: {} R_Jup, {} pc, {}K, logg={}, met={:.2f}, CO ratio={:3.2f}'.format(r_pl, dist_pc, temp, logg, met, CO),fontsize=10)
#     plt.xlabel("Distance from center (mas)")
#     plt.ylabel("Contrast, 5-sigma")
#     plt.yscale('log')
#     plt.grid(color='.9')

#     for j,d in enumerate(pl_dist):
#         plt.plot(d, all_pl_max[i,j], 'o')


