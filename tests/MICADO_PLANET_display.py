#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 09:26:32 2022

@author: ehuby
"""


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import interpolate
import os
from astropy import units as u
import glob


# from sim_funcs import fft_resize, micado_noise, micado_adi, simp_micado_flux, convolution, cheese_func, make_noise_map_no_mask

from micado_misthic.utils.obsparams import *
from micado_misthic.imgproc import get_circle_mask
from micado_misthic.analysis import get_rms_contrast
from scipy.signal import fftconvolve
from micado_misthic.imgproc.imgproc import rotate_frame


#plt.style.use('seaborn-muted')
plt.style.use('viridis')

save_fig = True

tel_diam        = 38.542 #m
lbd             = [1.19, 1.270, 1.582, 1.693, 2.100, 2.235, 1.245, 1.635, 2.145]#, 1.635, 1.693] # in um
lbd             = [2.1] #, 1.270, 1.582, 1.693, 2.100, 2.235, 1.245, 1.635, 2.145]#, 1.635, 1.693] # in um

coronograph = 'CLC1' # 'CLC0' or 'CLC1'
quartile_tab = ['Q4', 'MED', 'Q1']
pixscale_in_mas = 1.5
ron = 15
frame_exp_time=10.

star_mag = 5.24 # in K 
star_name = 'HR8799_'
zenith_distance = 24.5 + 21.1 # deg
dist_pc = 39.4 # distance in parsecs
airmass = 1./np.cos(zenith_distance*np.pi/180.)


# star_mag = 1.75 # in H
# star_name = 'EpsEri'
# dist_pc = 3.2 # distance in parsecs
# # zenith_distance = 24.5 - (9+27/60.)



# airmass = 1./np.cos(zenith_distance*np.pi/180.)
r_pl    = 1 # jupiter masses
temp_tab= [800, 1200] # Kelvin
logg    = 4.
met     = .32
CO      = 0.1

#%%

flux_dir = '/home/ehuby/WORK/SIMU/SIMU_MICADO/INPUT/FLUX/'
spec_dir = '/home/ehuby/WORK/SIMU/SIMU_MICADO/INPUT/Grid_BCharnay/cloud_R500/'
temp=temp_tab[0]
spec_file = np.loadtxt(spec_dir+'spectra_YGP_{}K_logg{}_met{:.2f}_CO{:3.2f}.dat'.format(temp,logg,met,CO), unpack=True)
planet_flux, spec_wave = get_planet_spectrum(spec_dir, temp, logg, met, CO, dist_pc, r_pl, flux_dir=flux_dir)
planet_flux0, spec_wave0 = get_planet_spectrum(spec_dir, temp, logg, met, CO, dist_pc, r_pl)

plt.figure(num=21)
plt.clf()
plt.plot(spec_wave0, planet_flux0, '-', label='Raw EXOREM data')
plt.plot(spec_wave, planet_flux, '.', label='Interpolated spectrum')
plt.xlim(.5, 2.5)
plt.xlabel('Wavelength [micron]')
plt.ylabel('Planet Spectrum [au]')
plt.legend()
plt.title(r'Planet: {} $R_J$, {} pc, {}K, logg={}, met={:.2f}, CO ratio={:3.2f}'.format(r_pl, dist_pc, temp, logg, met, CO),fontsize=10)

planet_params = []
planet_params.append([1200, 4.0, 0.32, 0.10, 39.4, 1])
# planet_params.append([1000, 4.0, 0.32, 0.50, 39.4, 1])
planet_params.append([800, 4.0, 0.32, 0.10, 39.4, 1])
# planet_params.append([600, 4.0, 0.32, 0.50, 39.4, 1])
# planet_params.append([500, 4.0, 0.32, 0.50, 39.4, 1])
# planet_params.append([500, 4.5, 1., 0.5, 3.2, 1])
# planet_params.append([500, 4., .32, 0.5, 39.4, 1])

plt.figure(num=22, figsize=(6,7))
plt.clf()

telescope_surface = get_aperture_surface('/home/ehuby/WORK/SIMU/SIMU_MICADO/INPUT/PUPIL/Pupil_ELT_v02_5missing_segments_v1.fits')

ls_tab=['-','-']
a_tab =[.9,.5]
c_tab = ['mediumseagreen', 'coral']
for pi,pp in enumerate(planet_params):
    planet_flux, spec_wave = get_planet_spectrum(spec_dir, *pp, flux_dir=flux_dir)
    planet_photon_flux, planet_emission_per_pix, planet_global_transmission = get_micado_flux(flux_dir, 
                                                                     planet_flux, '0', 
                                                                     frame_exp_time, telescope_surface, 
                                                                     airmass=airmass, pixel_scale=pixscale_in_mas)
    plt.plot(spec_wave, planet_flux*planet_global_transmission, linestyle=ls_tab[pi], 
             color=c_tab[pi], alpha=a_tab[pi], label=f'{pp[0]}K', linewidth=2)

plt.ylim(1e3, 1e7)
plt.xlim(1., 2.5)
plt.legend(loc=4, title='Planet temp.', fontsize=16, title_fontsize=16)
plt.xlabel('Wavelength [micron]', fontsize=16)
plt.ylabel(r'Planet Spectrum [photons.$\mu$m-1]', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.suptitle(r'Planet: R={}$R_J$, logg={}, met={:.2f}, CO ratio={:3.2f}'.format(r_pl, logg, met, CO),fontsize=16)
# plt.title("Effect of temperature")
plt.yscale('log')
plt.grid(color='.9')
# plt.savefig(f'/home/ehuby/WORK/SIMU/SIMU_MICADO/ANALYSIS/PLANETS/Planet_spectra_hr8799_1200K-vs-800K.png', dpi=300, bbox_inches='tight')




#%%

color_tab = ['mediumaquamarine', 'mediumorchid','coral']
linestyle_tab = [':','--','-']
planet_color = '.2'
planet_marker = ['o', 's', '^']

n_lbd = len(lbd)

for i in range(n_lbd):
    
    plt.figure(num=55, figsize=(9,7))
    plt.clf()
    fig55, ax55 = plt.subplots(num=55)
    
    line_contrast=[]
    marker_planet = []
    temp_tab_ = []
    
    for q,quartile in enumerate(quartile_tab) :
        adi_img_dir = f'/home/ehuby/WORK/SIMU/SIMU_MICADO/ANALYSIS/PLANETS/{star_name}/{coronograph}_{quartile}_noNCPA/lbd={lbd[i]:5.3f}_pxscale={pixscale_in_mas}/'
        
        coro_adi_conv_file = adi_img_dir+f'{star_name}_coro_adi_conv_{star_mag}mag_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec.fits'
        conv_adi_coro = fits.getdata(coro_adi_conv_file)
        
        psf_adi_conv_file = adi_img_dir+f'{star_name}_psf_adi_conv_{star_mag}mag_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec.fits'
        conv_psf_coro = fits.getdata(psf_adi_conv_file)

        max_psf_coro = np.max(conv_psf_coro)
        # norm_adi_coro = conv_adi_coro / max_psf_coro 

        rms_contrast, x = get_rms_contrast(conv_adi_coro)
        rms_contrast = 5.*rms_contrast/max_psf_coro
        x *= pixscale_in_mas
        
        plt.figure(num=55)
        line_cont, = ax55.plot(x, rms_contrast, label=f'Seeing: {quartile}', color=color_tab[q], linestyle=linestyle_tab[q], linewidth=3, alpha=.8)
        line_contrast.append(line_cont)
        
        for t,temp in enumerate(temp_tab):
            pl_adi_conv_file = adi_img_dir+f'{star_name}_coro_adi_conv_{temp}K_logg{logg}_met{met:.2f}_CO{CO:3.2f}_sep*mas_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec.fits'
            # plpsf_adi_conv_file = adi_img_dir+f'{star_name}_psf_adi_conv_{temp}K_logg{logg}_met{met:.2f}_CO{CO:3.2f}_sep*mas_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec.fits'
        
            file_list = glob.glob(pl_adi_conv_file)
            n_file = len(file_list)
            if n_file > 0 :
                sep_max = np.zeros((n_file, 2))
                all_planet_adi = []
                for j,f in enumerate(file_list):
                    print(f)
                    sep = int(f[f.find('sep')+3:f.find('mas')])
                    
                    conv_adi_pl = fits.getdata(f)
                    all_planet_adi.append(conv_adi_pl)
                    
                    f_psf = adi_img_dir+f'{star_name}_psf_adi_conv_{temp}K_logg{logg}_met{met:.2f}_CO{CO:3.2f}_sep{sep}mas_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec.fits'
                    
                    conv_psf_pl = fits.getdata(f_psf)
                    
                    # norm_adi_pl = conv_adi_pl / max_psf_coro 
            
                    max_psf_pl = np.max(conv_psf_pl)
            
                    max_planet = np.max(conv_adi_pl) / max_psf_coro
                    
                    sep_max[j,:] = [sep,max_planet]
                
                all_planet_adi = np.array(all_planet_adi)
                
                indsort=np.argsort(sep_max[:,0])
    
                print(f'##### {quartile} {temp} #####')
                if (quartile == 'MED'):
                    print('--> Plot the planets')
                    # plt.figure(num=55)
                    ax55.plot(sep_max[indsort,0], sep_max[indsort,1], '--', alpha=.5, color=planet_color)        
                    m_pl, = ax55.plot(sep_max[:,0], sep_max[:,1], color=planet_color, marker=planet_marker[t], linestyle='', ms=6)#, label='planet '+quartile)
                        
                    marker_planet.append(m_pl)
                    temp_tab_.append(temp)
                    
                ##################################################################
                plt.figure(num=56, figsize=(8,8))
                plt.clf()
                pixscale = pixscale_in_mas/1000.
                ny, nx = conv_adi_coro.shape
                e = [-ny/2*pixscale, nx/2*pixscale, -ny/2*pixscale, nx/2*pixscale]
                plt.imshow(conv_adi_coro+np.sum(all_planet_adi, axis=0), cmap="afmhot", origin='lower',  extent = e,
                           #vmin=conv_adi_coro.min()/5, vmax=conv_adi_coro.max()/5.)
                           vmin = -3e4, vmax=4e5)
                # plt.suptitle(f"{star_name}: {coronograph} {quartile} ADI image at $\lambda$={lbd[i]:5.3f}, pxscale={pixscale_in_mas}, Mag {star_mag}",fontsize=12)
                plt.suptitle(f"{star_name}Star Mag={star_mag}, Dist={dist_pc}pc - {coronograph}, $\lambda$={lbd[i]:5.3f}um, {quartile}", fontsize=16)
                # plt.title(r'Planet: {} $R_J$, {} pc, {}K, logg={}, met={:.2f}, CO ratio={:3.2f}'.format(r_pl, dist_pc, temp, logg, met, CO),fontsize=10)
                plt.title(r'Planet: R={}$R_J$, T={}K, logg={}, met={:.2f}, CO ratio={:3.2f}'.format(r_pl, temp, logg, met, CO),fontsize=14)
                plt.xlabel('[arcsec]', fontsize=16)
                plt.ylabel('[arcsec]', fontsize=16)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.xlim(-.7,.7)
                plt.ylim(-.7,.7)
                
                if save_fig :
                    save_img_dir = f'/home/ehuby/WORK/SIMU/SIMU_MICADO/ANALYSIS/PLANETS/{star_name}/{coronograph}_{quartile}_noNCPA_'
                    save_file = f'lbd={lbd[i]:5.3f}_pxscale={pixscale_in_mas}_{star_name}_planet_adi_{temp}K_logg{logg}_met{met:.2f}_CO{CO:3.2f}_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec'
                    
                    plt.figure(num=56)
                    plt.savefig(save_img_dir+'all_adi_images_'+save_file+'.png', dpi=120, bbox_inches='tight')
                    
                    
                
                ##################################################################
                plt.figure(num=57, figsize=(8,5))
                plt.clf()
                plt.imshow(conv_adi_coro+np.sum(all_planet_adi, axis=0), cmap="afmhot", origin='lower',  extent = e,
                           #vmin=conv_adi_coro.min()/5, vmax=conv_adi_coro.max()/5.)
                           vmin = -3e4, vmax=4e5)
                plt.suptitle(f"{star_name}Star Mag={star_mag}, Dist={dist_pc}pc - {coronograph}, $\lambda$={lbd[i]:5.3f}um, {quartile}", fontsize=16)
                plt.title(r'Planet: R={}$R_J$, T={}K, logg={}, met={:.2f}, CO ratio={:3.2f}'.format(r_pl, temp, logg, met, CO),fontsize=14)
                plt.xlabel('[arcsec]', fontsize=16)
                plt.ylabel('[arcsec]', fontsize=16)
                mn, mx = -.2, .2
                plt.yticks(np.linspace(mn,mx,5), fontsize=16)
                plt.xticks(fontsize=16)
                plt.ylim(mn*1.1, mx*1.1)
                plt.xlim(-.15, .65)
                plt.hlines(0., -.7,.7,colors='.7', linestyles=':', alpha=.5, linewidth=2)
                plt.vlines(0., -.7,.7,colors='.7', linestyles=':', alpha=.5, linewidth=2)
                plt.vlines(sep_max[:,0]/1000., 0.01,.03,colors='cyan', linestyles='-', alpha=.7)
                                
                if save_fig :
                    plt.figure(num=57)
                    plt.savefig(save_img_dir+'all_adi_images_ZOOM_'+save_file+'.png', dpi=120, bbox_inches='tight')
                    
               ##################################################################
        else :
            sep_max = np.zeros((n_file, 2))
            all_planet_adi = []

    # plt.figure(num=55)

    ax55.set_yscale('log')
    
    # plt.title(f"{star_name}: {coronograph}, {quartile} seeing, $\lambda$={lbd[i]:5.3f}um, Star Mag={star_mag}",fontsize=16)
    # plt.title(r'Planet: {} $R_J$, {} pc, {}K, logg={}, met={:.2f}, CO ratio={:3.2f}'.format(r_pl, dist_pc, temp, logg, met, CO),fontsize=14)
    ax55.set_title(f"{star_name}Star Mag={star_mag}, Dist={dist_pc}pc - {coronograph}, $\lambda$={lbd[i]:5.3f}um",fontsize=16)

    ax55.set_xlabel("Angular separation [mas]", fontsize=16)
    ax55.set_ylabel("5-sigma Contrast", fontsize=16)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    ax55.tick_params(axis = 'both', which = 'major', labelsize = 16)
    ax55.set_yscale('log')
    ax55.grid(color='.9')
    ax55.set_xlim(0,)
    ax55.set_ylim(5e-8,5e-5)
    
    # plt.legend(fontsize=16)
    legend1 = ax55.legend(line_contrast, quartile_tab, loc=1, fontsize=16, title='Seeing', title_fontsize=16)
    # ax = ax55.gca().add_artist(legend1)
    
    ax55b = ax55.twiny()
    mn, mx = ax55.get_xlim()
    ax55b.set_xlim(dist_pc*mn/1000.,dist_pc*mx/1000.)
    ax55b.legend(marker_planet, temp_tab_, loc=5, fontsize=16, title='Planet temp.', title_fontsize=16)
    ax55b.tick_params(axis = 'x', which = 'major', labelsize = 16, colors='.4')
    ax55b.set_xlabel("Separation [AU]", fontsize=16, color='.4')
    
if save_fig :
    save_img_dir = f'/home/ehuby/WORK/SIMU/SIMU_MICADO/ANALYSIS/PLANETS/{star_name}/{coronograph}_'+'-'.join(quartile_tab)+'_noNCPA_'
    save_file = f'lbd={lbd[i]:5.3f}_pxscale={pixscale_in_mas}_{star_name}_planet_adi_{temp}K_logg{logg}_met{met:.2f}_CO{CO:3.2f}_{dist_pc}pc_{ron}e-ron_tint{frame_exp_time}sec'
    
    plt.figure(num=55)
    plt.savefig(save_img_dir+'rms-contrast_curve_'+save_file+'.png', dpi=300, bbox_inches='tight')

