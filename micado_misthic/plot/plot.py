# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pylab import cm
from pathlib import Path



def plot_checks(pupil_plane,
                fp_mask_amp, fp_mask_phase, fp_mask_extent,
                lbd0, xyunit, lyot_plane_intensity,
                parangle_tab, zenith_dist_tab, ha_hours,
                save_path, timestamp, save_plots=False):
    """
    Checking plots for a quick simulation overview:
        window 17: pupil planes (entrance pupil and Lyot plane) amplitude and
        phase components
        window 18: focal plane amplitude and phase components
        window 19: intensity before and after the Lyot stop

    Input parameters:
    -----------------
        pupil_plane: dict
            contains the different entrance pupil plane amplitude and phase
            components. The first dictionary key must be 'pupil mask', which is
            multiplied to every other pupil plane component. The value for each
            item is a list containing the pupil plane definition (2D array, or
            a float, e.g. 0), and an integer, 0 for an amplitude definition, 1
            for a phase definition (useful for using a proper colormap).
        lyot_plane: dict
            contains the different entrance pupil plane amplitude and phase
            components. Same structure as pupil_plane, except that the first
            dictionary key must be 'lyot mask'.

    2019-12-07: a bug was fixed. Should avoid weird error messages.
    """

    font = {'weight':'normal', 'size'   : 10}
    matplotlib.rc('font', **font)
    plt.style.use('default')


    ### PUPIL PLANES ----------------------------------------------------
    n_pupil_plane = len(pupil_plane)
    pupil_keys = list(pupil_plane.keys())

    fs = (13,8)
    nr = 3
    if n_pupil_plane > 15:
        fs = (13, 10.5)
        nr = 4

    plt.figure(num=17, figsize=fs)
    plt.clf()
    ff, axx = plt.subplots(nrows=nr, ncols=5, num=17,
                           sharex='all', sharey='all',
                           squeeze=True, clear=True)

    pupil_mask = pupil_plane['pupil mask'][0]
    phase_colormap = 'RdBu'
    ampli_colormap = 'gist_yarg' #'RdGy'
    for i, axi in enumerate(axx.ravel()):
        if pupil_keys[i] == 'lyot mask':
            pupil_mask = pupil_plane['lyot mask'][0]
        pupil_screen = pupil_plane[pupil_keys[i]][0]*pupil_mask

        if pupil_plane[pupil_keys[i]][1] == 1:     # PHASE MAP
            colormap = cm.get_cmap(phase_colormap)
            valmax = np.max(np.abs(pupil_screen))
            # valmax = np.max(pupil_screen)
            valmin = -valmax
            if valmax == 0 :
                valmax = 1
                valmin = -1
        else :
            colormap = cm.get_cmap(ampli_colormap) # AMPLITUDE MAP
            valmin = 0.
            valmax = np.max(np.abs(pupil_screen))

        axi.imshow(pupil_screen, cmap =colormap,
                   vmin=valmin, vmax=valmax, origin='lower')
        axi.set_title(pupil_keys[i])
        axi.axis('off')
    ff.suptitle('PUPIL PLANES')

    ### FOCAL PLANE ----------------------------------------------------
    plt.figure(num=18, figsize=(6,3))
    plt.clf()
    ff, axx = plt.subplots(nrows=1, ncols=2, num=18,
                           squeeze=True,  sharey='all',
                           clear=True)

    if type(fp_mask_amp) is np.ndarray:
        axx[0].imshow(fp_mask_amp, extent=fp_mask_extent)
        axx[0].set_title('Focal plane ampl.')
        axx[0].set_xlabel(xyunit)
        axx[0].set_ylabel(xyunit)
        axx[1].imshow(fp_mask_phase, extent=fp_mask_extent,
           cmap=cm.get_cmap('hsv'), vmin=np.min(fp_mask_phase),
           vmax=np.max(fp_mask_phase)+2.*np.pi, origin='lower')
        axx[1].set_title('Focal plane phase')
        axx[1].set_xlabel(xyunit)
        axx[1].set_ylabel(xyunit)
    ff.suptitle('FOCAL PLANE')

    ### LYOT PLANE ----------------------------------------------------
    plt.figure(num=19, figsize=(12,6))
    plt.clf()
    ff, axx = plt.subplots(nrows=1, ncols=2, num=19,
                           squeeze=True,
                           clear=True)
    lyot_clmap = cm.get_cmap('plasma')
    bef=axx[0].imshow(lyot_plane_intensity['img_before_lyot'],
           cmap = lyot_clmap, vmin=0, origin='lower')
    axx[0].set_title('Intensity before Lyot')
    ff.colorbar(bef,ax=axx[0],orientation='horizontal',
                fraction=0.046, pad=0.04)
    aft=axx[1].imshow(lyot_plane_intensity['img_after_lyot'],
           cmap = lyot_clmap, vmin=0, origin='lower')
    axx[1].set_title('Intensity after Lyot')
    ff.colorbar(aft,ax=axx[1],orientation='horizontal',
                fraction=0.046, pad=0.04)
    ff.tight_layout()

    ### Parallactic angle plot ------------------------------------------------
    plt.figure(num=20, figsize=(5,5))
    plt.clf()
    fig, ax1 = plt.subplots(num=20)
    ax1.plot(ha_hours, parangle_tab, 'c', alpha=.5)
    ax1.plot(ha_hours, parangle_tab, 'c.')
    ax1.set_xlabel('Hour angle')
    ax1.set_ylabel('Parallactic angle [deg]', color='c')

    ax2 = ax1.twinx()
    ax2.plot(ha_hours, zenith_dist_tab, 'r', alpha=.5)
    ax2.plot(ha_hours, zenith_dist_tab, 'r.')
    ax2.set_ylabel('Zenith distance', color='r')

    ### SAVE plots in png images ----------------------------------------------
    if save_plots:
        
        save_path = save_path + 'png_figures/' 
        
        p = Path(save_path)
        if p.exists() is False:
            os.makedirs(save_path)
        
        save_path = save_path + timestamp
        
        plt.figure(num=17)
        plt.savefig(save_path+'_pupil_planes.png', dpi=120, bbox_inches='tight')
        plt.figure(num=18)
        plt.savefig(save_path+'_focal_plane_mask.png', dpi=120, bbox_inches='tight')
        plt.figure(num=19)
        plt.savefig(save_path+'_before-after_lyot.png', dpi=120, bbox_inches='tight')
        plt.figure(num=20)
        plt.savefig(save_path+'_parallactic_angle.png', dpi=120, bbox_inches='tight')

    return 1