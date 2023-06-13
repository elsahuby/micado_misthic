"""
MISTHIC
MIcado SimulaTor for HIgh Contrast

Main simulation function.

@author: EHu, PBa

Output images are returned in a dictionary:
    {'img_cube': final_detector_img_cube,
     'psf_cube': final_detector_psf_cube,
     'perf_psf': perf_nocoro_poly,
     'parfile' : config_newname}

"""

import numpy as np
import time
import glob
from pathlib import Path
import sys
import os
import inspect

from configobj import ConfigObj
from validate import Validator

from astropy.table import Table
from astropy.io import fits

##### import misthic functions and submodules

from micado_misthic.imgproc import get_frame_center
from micado_misthic.utils import get_timestamp
# from micado_misthic.fits import read_fits
# from micado_misthic.fits import write_fits
from micado_misthic.imgproc import get_circle_mask
from micado_misthic.simu.coromask import get_fqpm_phase_mask
from micado_misthic.simu.coromask import get_vortex_phase_mask
from micado_misthic.imgproc.imgproc import pad_and_roll
from micado_misthic.imgproc.imgproc import rotate_frame
from micado_misthic.simu.randomaberr import get_opd_screen
from micado_misthic.utils import change_angle_interval
from micado_misthic.plot import plot_checks
from micado_misthic.config import edit_n_write_config

import micado_misthic.simu.zernike as zern
import micado_misthic.utils.obsparams as obsparams
import micado_misthic.simu as simu


# progress bar [pip install tqdm]
from tqdm import trange



def run_misthic(parameter_file, do_psf=False, silent=False,
                checking_plots=True, progress_bar=False,
                create_output_dir=False) :
    """
    MIcado SimulaTor for HIgh Contrast
    Runs a simulation of the coronagraphic image for MICADO.
    Simulation parameters are defined in a ConfigObj file ('.ini').

    Parameters
    ----------
    parameter_file : ConfigObj file (see 'misthic_config.ini' template)
        parameter file with all simulation parameters
    do_psf : boolean
        returns the non coronagraphic PSF computed with the same aberrations
        but without the focal plane mask.
        (with aberrations, no coro mask, with Lyot mask)
    silent : boolean
        displays and plots some intermediate results.
    checking_plots : boolean
        displays some images and plots to check the parameters of the simulation
        (pupil planes, focal planes, parallactic angle)
    progress_bar : boolean
        displays a progress bar to monitor the progress of the simulation
    create_output_dir : boolean
        automatically creates the directory to save the images if it does not
        exist.

    Returns
    -------
    final_output: dictionary
        dictionary data structure with the following keys:
        'img_cube' : final coronagraphic image cube (nb of frames = n_images)
        'psf_cube' : final (non coro) PSF cube (with aberrations, no focal plane
                     mask, with Lyot stop),
                     returned only if do_psf input parameter is True (nb of frames = n_images)
        'perf_psf' : perfect PSF image (no coro, no aberr),
                     this image is used for normalization by its maximal value.
        'par_file' : parameter file name saved for the record at the end of the
                     simulation.
    """

    ### Checking the parameter file ###
    parfile_check = os.path.exists(parameter_file)
    if parfile_check is False:
        print('ERROR: The path for the parameter file is not correct.')
        sys.exit(0)
    
    ### TIMING ###
    t_start           = time.time()
    stamp_start       = get_timestamp(stamp_format='full_tight')
    stamp_start2      = get_timestamp(stamp_format='full')
    if silent is not True:
        print(stamp_start2+ ' -- INITIATING SIMULATION')

    ### CONFIGURATION FILE ###
#    configspec_file   = 'misthic_configspec.ini'
    configspec_file   = os.path.split(inspect.getfile(run_misthic))[0]+'/misthic_configspec.ini'
    config            = ConfigObj(parameter_file,
                                  configspec=configspec_file)
    vtor              = Validator()
    checks            = config.validate(vtor,copy=True) # copy=True for copying the comments

    ### FILE I/O SETTINGS ###
    input_directory   = config['input_directory']
    output_directory  = config['output_directory']
    save_fits_img     = config['save_fits_img']
    save_fits_lyot    = config['save_fits_lyot']
    if checks['save_fits_lyot'] is False :
        save_fits_lyot_before = (save_fits_lyot[0].strip('()') == 'True')
        save_fits_lyot_after  = (save_fits_lyot[1].strip('()') == 'True')
    else:
        save_fits_lyot_before = save_fits_lyot[0]
        save_fits_lyot_after = save_fits_lyot[1]
    save_fits_poly    = config['save_fits_poly']
    save_png          = config['save_png']

    ### SIMULATION CONFIG ###
    simuconfig        = config['simuconfig']
    n_images          = simuconfig['n_images']
    n_img_mean        = simuconfig['n_img_mean']
    delta_t           = simuconfig['delta_t']
    zenith_distance   = simuconfig['zenith_distance']
    tel_diameter      = simuconfig['tel_diameter']
    tel_latitude      = simuconfig['tel_latitude']
    centering         = simuconfig['centering']

    ### PLANET CONFIGURATION ###
    planetconfig      = config['planetconfig']
    planet_ON         = planetconfig['planet_ON']
    planet_dist       = planetconfig['dist_planet']
    planet_pa         = planetconfig['planet_pa']
    planet_throughput = planetconfig['planet_throughput']
    dist_minmax       = planetconfig['dist_minmax']

    if ((planet_ON is True) and (planet_throughput is True)) :
        exit_simul = input('WARNING: planet_ON is True AND planet_throughput is True\n'+
                              '         These two modes are incompatible. By default planet_ON will prevail.\n'+
                              '         Would you like to exit?\n[y/n] -> ')
        if exit_simul.lower() == 'yes' or exit_simul.lower() == 'y':
            sys.exit()

    if planet_throughput is True:
        n_img_mean = 1
        if checks['planetconfig'] is False:
            if checks['planetconfig']['dist_minmax'] is False:
                for i, strval in enumerate(dist_minmax):
                    dist_minmax[i] = float(strval.strip('()[]'))
        print('Planet throughput simulation => n_img_mean forced to 1.')

    ### CORONAGRAPH CONFIGURATION ###
    coroconfig        = config['coroconfig']
    pup_fits          = coroconfig['pup_fits']
    lyot_fits         = coroconfig['lyot_fits']
    fp_mask           = coroconfig['fp_mask']

    ### DETECTOR CONFIGURATION ###
    detectorconfig    = config['detectorconfig']
    det_fov           = detectorconfig['detector_fov']
    det_sampling      = detectorconfig['detector_sampling']

    ### CHROMATIC CONFIGURATION ###
    waveconfig        = config['waveconfig']
    n_wave            = waveconfig['n_wave']
    lbd0              = waveconfig['lbd0']
    delta_lbd         = waveconfig['delta_lbd']

    ### ABERRATION CONFIGURATION ###
    aberrconfig       = config['aberrconfig']
    ### -- Optical aberrations: pre- and post-coronagraph
    pre_zernike_ON    = aberrconfig['pre_zernike_ON']
    pre_z_coeff       = aberrconfig['pre_z_coeff']
    if checks['aberrconfig']['pre_z_coeff'] is False:
        for i, strval in enumerate(pre_z_coeff):
            pre_z_coeff[i] = float(strval.strip('()[]'))
    # PHASE aberration map
    pre_aberr_fits    = aberrconfig['pre_aberr_fits']
    pre_aberr_file    = aberrconfig['pre_aberr_file']
    if type(pre_aberr_file) is str:
        pre_aberr_file = np.expand_dims(pre_aberr_file, 0)
    # AMPLITUDE aberration map
    pre_amp_fits    = aberrconfig['pre_amp_fits']
    pre_amp_file    = aberrconfig['pre_amp_file']
    if type(pre_amp_file) is str:
        pre_amp_file = np.expand_dims(pre_amp_file, 0)

    sphere_jitter     = aberrconfig['sphere_jitter'] # [SPHERE UPGRADE]
    sph_jitter_file   = aberrconfig['sph_jitter_file']
    post_zernike_ON   = aberrconfig['post_zernike_ON']
    post_z_coeff      = aberrconfig['post_z_coeff']
    if checks['aberrconfig']['post_z_coeff'] is False:
        for i, strval in enumerate(post_z_coeff):
            post_z_coeff[i] = float(strval.strip('()[]'))
    post_aberr_fits    = aberrconfig['post_aberr_fits']
    post_aberr_file    = aberrconfig['post_aberr_file']
    ### -- Turbulence residuals
    turbu_ON          = aberrconfig['turbu_ON']
    turbu_directory   = aberrconfig['turbu_directory']
    turbu_prefix      = aberrconfig['turbu_prefix']
    turbu_remove_tt   = aberrconfig['turbu_remove_tt']
    turbu_file_nb_init= aberrconfig['turbu_file_nb_init']
    
    if turbu_ON:
        turbu_file_list   = glob.glob(turbu_directory+turbu_prefix+'*.fits')
        if turbu_file_list == []:
            turbu_file_list = glob.glob(input_directory+turbu_directory+turbu_prefix+'*.fits')
        turbu_file_list.sort()
        turbu_nb_list = np.array([int(f[-8:-5]) for f in turbu_file_list])
        file_ind_init = np.where(turbu_nb_list == turbu_file_nb_init)[0][0]
        turbu_file_list = turbu_file_list[file_ind_init:]
    
    turbu_delta_n_phase = aberrconfig['turbu_delta_n_phase']
    ### -- Static aberrations
    static_fits       = aberrconfig['static_fits']
    static_file       = aberrconfig['static_file']
    static_rms_nm     = aberrconfig['static_rms_nm']
    ### -- Rotating aberrations
    rotat_fits        = aberrconfig['rotat_fits']
    rotat_file        = aberrconfig['rotat_file']
    rotat_rms_nm      = aberrconfig['rotat_rms_nm']
    ### -- Atmospheric Refraction
    atm_refrac_ON     = aberrconfig['atm_refrac_ON']
    post_ADC_ON       = aberrconfig['post_ADC_ON']
    pressure          = aberrconfig['pressure']
    temperature       = aberrconfig['temperature']
    rel_humidity      = aberrconfig['rel_humidity']
    refrac_angle      = aberrconfig['refrac_angle']

    ## Check if output_directory exists in case data will be saved
    if output_directory[-1] != '/' :
        output_directory += '/'
    p = Path(output_directory)
    save_data = (save_fits_img or save_fits_lyot_before or
                 save_fits_lyot_after or save_fits_poly)
    if (p.exists() is False) and ((save_data is True) or (checking_plots is True)):
        if create_output_dir is False:
            makedir = input('ERROR: Specified output_directory does not exist.\n'+
                            '       Create following directory?\n{}\n[y/n] -> '.format(output_directory))
            if makedir.lower() == 'yes' or makedir.lower() == 'y':
                os.makedirs(output_directory)
            else :
                sys.exit('Please specify the output_directory in the config file.')
        else:
            os.makedirs(output_directory)

    ### ENTRANCE PUPIL ###
    if pup_fits : # load entrance pupil mask from fits file
        pup_file     = coroconfig['pup_file']
        pup_mask     = fits.getdata(input_directory+pup_file)
        # redefine pupil parameters according to file
        grid_width   = pup_mask.shape[0]
        grid_shape   = (grid_width,grid_width)
        pup_diameter = coroconfig['pup_radius_in_fits'] * 2.
        pup_ratio    = pup_diameter/grid_width
        pup_name     = ('pupfits_D{:4f}_PR{:4.2f}').format(pup_diameter,pup_ratio)
        if silent is not True:
            print('Entrance pupil loaded from file', pup_file)
    else: # Entrance pupil defined by user parameters
        grid_width   = coroconfig['grid_width']
        grid_shape   = (grid_width,grid_width)
        pup_ratio    = coroconfig['pup_ratio']
        pup_obst     = coroconfig['pup_obst']
        pup_diameter = np.rint(grid_width * pup_ratio)
        pup_mask = get_circle_mask(grid_shape, pup_diameter/2.)

        # Define central obstruction #
        if pup_obst > 0. and pup_obst < 1.:
            obst_mask = (1. - get_circle_mask(grid_shape, pup_obst*pup_diameter/2.))
            pup_name  = ('pupman_D{:4f}_PR{:4.2f}_CO{:4.2f}').format(pup_diameter,pup_ratio, pup_obst)
        else:
            obst_mask = 1.
            pup_name  = ('pupman_D{:4f}_PR{:4.2f}').format(pup_diameter,pup_ratio)

        pup_mask     = pup_mask * obst_mask

        print('Circular entrance pupil created with diam = {:7.2f} pix'.format(pup_diameter))

    ### PUPIL APODIZATION
    ### -- Amplitude Apodization
    pre_apod_fits    = coroconfig['pre_apod_fits']
    pre_apod_file    = coroconfig['pre_apod_file']
    pre_phas_fits    = coroconfig['pre_phas_fits']
    pre_phas_file    = coroconfig['pre_phas_file']

    ### LYOT MASK ###
    if fp_mask == 'nocoro':
        ### No coronagraph => no Lyot stop
        lyot_mask = 1.
        if silent is not True:
            print('NO CORONAGRAPH => No Lyot stop')
    else:
        if lyot_fits : # load Lyot pupil mask from fits file
            lyot_file     = coroconfig['lyot_file']
            lyot_mask     = fits.getdata(input_directory+lyot_file)
            if (lyot_mask.shape)[0] == 2: # amplitude and phase mask
                lyot_mask = lyot_mask[0] * np.exp(1j*lyot_mask[1])
            lyot_name     = 'lyotfits'
            print('Lyot mask loaded from file', lyot_file)
        else: # Lyot stop from user parameters
            lyot_in       = coroconfig['lyot_in']
            lyot_in_diam  = lyot_in * pup_diameter
            lyot_out      = coroconfig['lyot_out']
            lyot_out_diam = lyot_out * pup_diameter
    
            lyot_cyx      = np.array(get_frame_center(grid_shape))
            lyot_mask     = get_circle_mask(grid_shape, lyot_out_diam/2.,
                                                cyx = lyot_cyx)
            lyot_mask     = lyot_mask*(1.-get_circle_mask(grid_shape,
                                                              lyot_in_diam/2.,
                                                              cyx = lyot_cyx))
            if silent is not True:
                print('Annular Lyot stop created with diams = ({:7.2f},{:7.2f}) pix'.format(lyot_in_diam/2., lyot_out_diam))

    ### CREATE THE WAVELENGTH TABLE ###
    if n_wave > 1 :
        wave_tab      = ((np.arange(n_wave)+0.5)/(n_wave)-0.5)*delta_lbd + lbd0
    else:
        wave_tab      = np.array([lbd0])


    ### ATM. REFRACTION COMPUTATION ###
    lbdd_tab          = wave_tab * 1e-6 / tel_diameter * 180 / np.pi * 3600. # [arcsec]
    n                 = obsparams.get_air_index(wave_tab, pressure,
                                          temperature, rel_humidity)
	# Fillipenko 1982 Eq. 4
    refraction_coef   = n-np.mean((n[0],n[-1]))  ## TO BE MULTIPLIED BY TAN OF THE REAL ZEN. DIST.
    refraction_lbdd   = refraction_coef * 180./np.pi*3600. / lbdd_tab # [lambda/D]

    ### FOCAL PLANE MASK ###
    # lamdba/D in mas:
    lbdd0 = lbd0 * 1e-6 / tel_diameter * 180 / np.pi * 3600. * 1000. # [mas]
    vapp_off_axis_psf2 = False
    vapp_leakage = 0.

    if fp_mask == 'occulter':
        occconfig     = coroconfig['occulter']
        occulter_rad_mas  = occconfig['occulter_rad'] # [mas]
        occulter_rad = occulter_rad_mas / lbdd0 # [l/D]
        occulter_rad_max = occulter_rad * lbd0 / np.min(wave_tab) # [l/D]
        mft_sampling  = occconfig['mft_sampling']
        coro_name     = 'clc_R{:3.1f}'.format(occulter_rad)

        if checking_plots is True :
            fp_mask_fov = np.rint(occulter_rad * 4) # [lambda/D]
#            occulter_rad_mas = occulter_rad * lbdd0 * 1000. # [mas]
            fp_mask_shape = (np.int(np.round(fp_mask_fov*mft_sampling)),np.int(np.round(fp_mask_fov*mft_sampling)))
            fp_mask_amp = 1.-get_circle_mask(fp_mask_shape, occulter_rad*mft_sampling)
            fp_mask_phase = fp_mask_amp * 0.
            sy, sx = fp_mask_shape
            cy, cx = get_frame_center((sy,sx))
            fp_mask_extent = (np.array([0-cx,sx-1.-cx,0-cy,sy-1.-cy])/mft_sampling*lbdd0)
            xyunit = 'mas'

    elif fp_mask == 'fqpm':
        fqpmconfig    = coroconfig['fqpm']
        fp_sampling   = fqpmconfig['fp_sampling']
        fp_fov        = fqpmconfig['fp_fov']
        fp_dim        = int(np.ceil(fp_fov * fp_sampling /2.)*2.)
        fp_shape      = (fp_dim, fp_dim)
        fp_phase_mask = get_fqpm_phase_mask(fp_shape, centering=centering)
        coro_name     = 'fqpm'
        occulter_rad  = None

    elif fp_mask == 'vortex':
        vortconfig    = coroconfig['vortex']
        v_charge      = vortconfig['v_charge']
        v_offset      = vortconfig['v_offset'] * np.pi/180. # [rad]
        fp_sampling   = vortconfig['fp_sampling']
        fp_fov        = vortconfig['fp_fov']
        v_sign        = vortconfig['v_sign']
        fp_dim        = int(np.ceil(fp_fov * fp_sampling /2.)*2.)
        fp_shape      = (fp_dim, fp_dim)
        occulter_rad  = 2./12.
        fp_phase_mask = get_vortex_phase_mask(fp_shape, charge=v_charge,
                                                offset_rad=v_offset,
                                                center_mask_diam=occulter_rad*fp_sampling*2.,
                                                centering=centering, sign=v_sign)
    elif fp_mask == 'vapp':
        fp_phase_mask = 1.
        fp_shape      = (10, 10)
        fp_sampling   = 1
        fp_fov        = 1.
        vappconfig    = coroconfig['vapp']
        vapp_ph_file  = vappconfig['vapp_phase_file']
        vapp_grating  = vappconfig['vapp_grating']
        vapp_phase    = fits.getdata(input_directory+vapp_ph_file)
        vapp_leakage  = vappconfig['vapp_leakage']

        if vapp_leakage != 0 :
            lyot_mask_0 = lyot_mask.copy()
        if vapp_grating is True :
            lyot_mask_2 = lyot_mask * np.exp(-1j*vapp_phase)
            vapp_off_axis_psf2 = ((fp_mask == 'vapp') and (vapp_grating is True))
        lyot_mask     = lyot_mask * np.exp(1j*vapp_phase)

    elif fp_mask == 'paplc' :
        paplcconfig   = config['coroconfig']['paplc']
        fp_sampling   = paplcconfig['fp_sampling']
        fp_fov        = paplcconfig['fp_fov']
        fp_dim        = int(np.ceil(fp_fov * fp_sampling /2.)*2.)
        fp_shape      = (fp_dim, fp_dim)
        fp_mask_file  = paplcconfig['fp_mask_file']
        fp_phase_mask = fits.getdata(input_directory + fp_mask_file)

    elif fp_mask == 'perfect' and (checking_plots is True) :
        fp_mask_amp = 1.
        fp_mask_phase = 0.
        fp_mask_extent= (0,1,0,1)
        xyunit = r'$\lambda/D$'

    elif ((fp_mask == 'nomask') | (fp_mask == 'nocoro')) :
        fp_phase_mask = 1.
        fp_shape      = (10, 10)
        fp_sampling   = 1
        fp_fov        = 1.

    focal_plane_coro = ((fp_mask=='vortex')|(fp_mask=='fqpm') |
            (fp_mask=='vapp')|(fp_mask=='paplc') |
            (fp_mask == 'nomask') | (fp_mask == 'nocoro'))
    if focal_plane_coro and (checking_plots is True) :
        fp_mask_amp = np.abs(fp_phase_mask)
        fp_mask_phase = np.arctan2(np.imag(fp_phase_mask),np.real(fp_phase_mask))
        sy, sx = fp_shape
        cy, cx = get_frame_center((sy,sx))
        fp_mask_extent= (np.array([0-cx,sx-1.-cx,0-cy,sy-1.-cy])/fp_sampling)
        xyunit = r'$\lambda/D$'



	### LYOT SHIFT ###
    lyot_shift    = coroconfig['lyot_shift']
    lyot_drift    = coroconfig['lyot_drift']

    if checks['coroconfig'] is not True:

        if checks['coroconfig']['lyot_shift'] is False:
            for i, strval in enumerate(lyot_shift):
                lyot_shift[i] = float(strval.strip('()[]'))

        if checks['coroconfig']['lyot_drift'] is False:
            for i, strval in enumerate(lyot_drift):
                lyot_drift[i] = float(strval.strip('()[]'))

    if lyot_shift[0] !=0. or lyot_shift[1] !=0. :
        shift_x = round(lyot_shift[0]/100.*pup_diameter)
        shift_y = round(lyot_shift[1]/100.*pup_diameter)
        lyot_mask = pad_and_roll(lyot_mask, (shift_x,shift_y))

        if silent is False :
            print('Lyot stop shift [% of D_pup / pixels] = '+
                  'x {0:4.2f}/{1:4.2f} ; y {2:4.2f}/{3:4.2f}'
                  .format(lyot_shift[0], shift_x, lyot_shift[1], shift_y ) )

        
        if fp_mask == 'vapp':
            if vapp_off_axis_psf2 is True :
                lyot_mask_2 = pad_and_roll(lyot_mask_2,(shift_x,shift_y))
            if vapp_leakage != 0 :
                lyot_mask_0 = pad_and_roll(lyot_mask_0,(shift_x,shift_y))
                
    else :
        shift_x, shift_y = 0., 0.

    delta_t_ind_hour = delta_t / n_img_mean / 3600. # time per individual frame





    ########################################################################
    ###### Generate OPD screens: initialization to default zero value ######
    planet_tilt             = 0.
    pre_zernike_opd_screen  = 0.
    pre_aberr_opd           = 0. # PHASE aberration
    pre_amp                 = 1. # AMPLITUDE aberration
    jitter_map              = 0.
    post_aberr_opd          = 0.
    post_zernike_opd_screen = 0.
    static_opd_screen       = 0.
    rotat_opd_screen_i      = 0.
    turbu_opd_screen_i      = 0.
    refraction_tilt         = 0.
    pre_apod                = 1. # amp. apodization: init. to 1 (no apod)
    pre_phas                = 0.
    ##### Initial Lyot mask #####
    if isinstance(lyot_mask, float):
        lyot_mask_drift     = 1.
    else :
        lyot_mask_drift         = lyot_mask.copy()

    if fp_mask == 'vapp' :
        if vapp_off_axis_psf2 is True:
            lyot_mask_drift_2   = lyot_mask_2.copy()
        if vapp_leakage != 0 :
            lyot_mask_drift_0   = lyot_mask_0.copy()

    ################## PRE-CORO ZERNIKE ABERRATIONS ####################
    if pre_zernike_ON is True:
        pre_zernike_opd_screen = zern.get_zernike_screen((grid_width,grid_width),
                                                        pre_z_coeff,
                                                        pup_diameter=pup_diameter)

    ################ PRE-CORO ABERRATIONS FROM FITS ####################
    if pre_aberr_fits is True:
        for paf in pre_aberr_file:
            pre_aberr_opd += fits.getdata(input_directory + paf)
    if pre_amp_fits is True:
        for paf in pre_amp_file:
            pre_amp = pre_amp * fits.getdata(input_directory + paf)

    ################# [SPHERE UPGRADE] SPHERE jitter ####################
    if sphere_jitter is True:
        sphere_jitter_data = fits.getdata(input_directory+sph_jitter_file)

    ################# POST-CORO ZERNIKE ABERRATIONS ####################
    if post_zernike_ON is True:
        post_zernike_opd_screen = zern.get_zernike_screen((grid_width,grid_width),
                                                         post_z_coeff,
                                                         pup_diameter=pup_diameter)

    ################ POST-CORO ABERRATIONS FROM FITS ####################
    if post_aberr_fits is True:
        post_aberr_opd = fits.getdata(input_directory + post_aberr_file)

    ####################### STATIC ABERRATIONS ########################
    if static_fits is True:
        static_opd_screen = fits.getdata(input_directory + static_file)
    elif static_rms_nm != 0:
        static_opd_screen = get_opd_screen(grid_width, static_rms_nm,
                                               high_freq_cut=grid_width/2.,
                                               pup_radius=pup_diameter/2.)

    ####################### ROTATING ABERRATIONS ######################
    if rotat_fits is True:
        rotat_opd_screen = fits.getdata(input_directory + rotat_file)
    elif rotat_rms_nm != 0.:
        rotat_opd_screen = get_opd_screen(grid_width, rotat_rms_nm,
                                              high_freq_cut=grid_width/2.,
                                              pup_radius=pup_diameter/2.)

    ##################### ATMOSPHERIC REFRACTION ######################
    if atm_refrac_ON is True :
        # tilted wavefront [1 lambda/D] for simulation of the atmospheric refraction
        refraction_tilt = zern.get_tilted_wavefront(pup_mask, pup_diameter,
                                                   angle = refrac_angle)

    ###################### AMPLITUDE APODIZATION #######################
    if pre_apod_fits is True :
        pre_apod = fits.getdata(input_directory + pre_apod_file)
        print('Apodization mask loaded from file '+pre_apod_file)
    if pre_phas_fits is True :
        pre_phas = fits.getdata(input_directory + pre_phas_file)
        print('Entrance pupil phase mask loaded from file '+pre_phas_file)


    ## DISPLAY [AMP APOD + LYOT] TRANSMISSION ##
    coro_throughput_max = np.sum(np.abs(pre_apod**2*lyot_mask))/np.sum(pup_mask)
    if silent is not True:
        print('---\nApodization transmission        = {0:4.3f}'.format(np.sum(np.abs(pre_apod**2*pup_mask))/np.sum(pup_mask)))
        print('Lyot stop transmission          = {0:4.3f}'.format(np.sum(np.abs(lyot_mask))/np.sum(pup_mask)))
        print('Max Coro Throughput (Apod+Lyot) = {0:4.3f}\n---'.format(coro_throughput_max))

    ##################### PERFECT PSF: NO CORO - NO ABERR ####################
    # -----------------------------------------------------------------------#
    ## NO CORO MASK- NO LYOT MASK - NO APOD MASK
    focal_npix  = int(np.ceil(det_fov * det_sampling /2.)*2.)
    x1          = int(pup_diameter * det_sampling/2.-focal_npix/2.)
    x2          = int(pup_diameter * det_sampling/2.+focal_npix/2.)

    if silent is not True:
        print(get_timestamp(stamp_format='full')+ ' -- STARTING LOOP\n')

    # Polychro PSF cube initialization
    nocoro_polycube = np.zeros((n_wave, x2-x1, x2-x1))

    if save_fits_poly is True:
        wave_loop = trange(n_wave, desc='Polyc')#, ascii=True)
    else:
        wave_loop = range(n_wave)

    for l in wave_loop : # ======= loop on wavelengths
        ### INPUT WAVEFRONT ###
        if post_ADC_ON is False:
            # no ADC correction
            wavefront = pup_mask #* np.exp(1j*(refraction_tilt * refraction_lbdd[l]))
        else :
            # perfect correction is assumed
            wavefront = pup_mask

        ### PROPAGATE WF DEPENDING ON FOCAL PLANE MASK ###
        #### 2019-01-15 removed the lyot stop for perfect PSF
        no_mask = 1.
        no_lyot = 1. # abs(pup_mask)
        if fp_mask == 'occulter':
            nocoro_mono = simu.propagate_mono_lyot(
                    wavefront, no_lyot, det_fov_pix=(x1,x2),
                    occ_rad_pix = 0., det_sampling=det_sampling*pup_ratio,
                    centering=centering, lbd=wave_tab[l],
                    lbd_ref=np.min(wave_tab))['img_detector']

        elif ((fp_mask == 'vortex')|(fp_mask=='fqpm')|
                (fp_mask=='vapp')|(fp_mask=='paplc')|
                (fp_mask == 'perfect')|(fp_mask=='nomask')|
                (fp_mask == 'nocoro')):
            nocoro_mono = simu.propagate_mono_vortex(
                    wavefront, no_mask, no_lyot, det_fov_pix=(x1,x2),
                    det_sampling=det_sampling*pup_ratio,
                    centering=centering, lbd=wave_tab[l],
                    lbd_ref=np.min(wave_tab))['img_detector']

        nocoro_polycube[l] = nocoro_mono

        if l == 0:
            perf_nocoro_poly = nocoro_mono.copy() / n_wave
            if save_fits_poly :
                perf_factor_poly = np.zeros(n_wave) + np.max(nocoro_mono)
        else :
            perf_nocoro_poly = perf_nocoro_poly + nocoro_mono / n_wave
            if save_fits_poly :
                perf_factor_poly[l] = np.max(nocoro_mono)

    # Simulated images are normalized by the maximum of the perfect PSF
    perf_factor      = np.max(perf_nocoro_poly)
    perf_nocoro_poly = np.float32(perf_nocoro_poly/perf_factor)


    ########################## CORONAGRAPHIC IMAGES ##########################
    # -----------------------------------------------------------------------#

    ######################### PLANET SIMULATOR ###############################
    # --------------------- THROUGHPUT ESTIMATION ----------------------------
    if planet_throughput is True:
        planet_tilt0  = zern.get_tilted_wavefront(pup_mask, pup_diameter,
                                                 angle = 90. - planet_pa)
        if n_images > 1 :
            delta_dist = (dist_minmax[1]-dist_minmax[0]) / (n_images-1.)
            planet_sep_tab  = np.zeros(n_images+1)
            planet_sep_tab[1:]  =  (delta_dist * np.arange(n_images)) + dist_minmax[0]
            n_images = n_images + 1
        else :
            delta_dist = 0.
            planet_sep_tab = np.expand_dims(dist_minmax[0],0)
            print('!!! WARNING !!! you must specify n_images > 1 to scan planet separations.')

    ##################### PARALLACTIC ANGLE COMPUTATION ######################
    ha_hours      = (np.arange(n_images) - (n_images-1.)/2. ) * delta_t / 3600.
    dec_deg       = (tel_latitude - zenith_distance)
    parangle_tab  = obsparams.get_parallactic_angle(ha_hours, dec_deg, tel_latitude)
    if zenith_distance < 0 : # at transit, parangle=180
        parangle_tab = change_angle_interval(parangle_tab)
    zenith_dist_tab = obsparams.get_zenith_distance(ha_hours, dec_deg, tel_latitude)

    # phase screen count variables
    t_count = 0 # count for total turbu OPD screens used
    n_turbu_cube = 0 # nb of turbu OPD screens in the current cube
    t_turbu_cube = 0 # count for the nb of loaded turbu cubes

    # Polychro cube initialization
    if (save_fits_poly):
        coro_polycube = np.zeros((n_wave, x2-x1, x2-x1), dtype='float32') # coronagraphic image cube
        if do_psf :
            noco_polycube = np.zeros((n_wave, x2-x1, x2-x1), dtype='float32') # non coro image cube
                                                                 # (but including aberrations
        else:
            noco_polycube = 0.

    lyot_output = (save_fits_lyot_before or save_fits_lyot_after or checking_plots)


    ##################### CORONAGRAPHIC IMAGE SIMULATION ######################

    if progress_bar is True:
        main_loop = trange(n_images, desc='total')#, ascii=True)
    else:
        main_loop = range(n_images)

    for i in main_loop :

        if (silent is not True and progress_bar is not True):
            # display progress
            if n_images < 10:
                print('\n '+get_timestamp(stamp_format='full')+' -- img nb {:1.0f} / {:1.0f}'.format(i+1, n_images))
            else :
                n_img2 = np.ceil(n_images/10.)
                if ((i) % n_img2) == 0:
                    print('\n '+get_timestamp(stamp_format='full')+' -- img nb {:3.0f} / {:3.0f}'.format(i+1, n_images))

        ################# PARALLACTIC ANGLE FOR THE SUB-IMAGEs #################
        sub_ha_hours = ((np.arange(n_img_mean)+0.5)/(n_img_mean)-0.5)*delta_t/3600. + ha_hours[i]
        sub_parangle = obsparams.get_parallactic_angle(sub_ha_hours, dec_deg, tel_latitude)
        if zenith_distance == 0 : # discard rotation
            sub_parangle *= 0.
        if zenith_distance < 0 :
            parangle_mean = np.mean(change_angle_interval(sub_parangle))
        else :
            parangle_mean = np.mean(sub_parangle)
        parangle_tab[i]= parangle_mean

        ############# REFRACTION AMPLITUDE FOR THE SUB-IMAGES #################
        sub_zendist           = obsparams.get_zenith_distance(sub_ha_hours,
                                                        dec_deg,
                                                        tel_latitude)
        sub_refraction_lbdd    = (np.expand_dims(refraction_lbdd, axis=0) *
                                 np.tan(np.expand_dims(sub_zendist*np.pi/180., axis=1)))

        if progress_bar is True:
            turb_loop = trange(n_img_mean, desc='turbu')#, ascii=True)
        else:
            turb_loop = range(n_img_mean)

        for t in turb_loop : # ======= loop on sub-images

            if turbu_ON : # ===== load turbulent OPD screen
                if t_count > n_turbu_cube -1 :
                    t_count = t_count - n_turbu_cube
                    # load next turbulence phase screen cube on the list
                    t_load0 = get_timestamp(stamp_format='time')
                    print('\n\t{} Load turbu cube... \n\t\t{}'.format(t_load0,turbu_file_list[t_turbu_cube]))
                    turbu_opd_screen_cube = - np.float32(fits.getdata(turbu_file_list[t_turbu_cube]))
                    t_load1 = get_timestamp(stamp_format='time')
                    print('\t{} Turbu cube loaded!'.format(t_load1))
                    turbu_cube_shape = turbu_opd_screen_cube.shape
                    if len(turbu_cube_shape) == 2:
                        n_turbu_cube = 1
                        turbu_opd_screen_cube = np.expand_dims(turbu_opd_screen_cube, 0)
                    else :
                        if ((turbu_cube_shape[0] == turbu_cube_shape[1]) and (turbu_cube_shape[0] == grid_width)) :
                            turbu_opd_screen_cube = np.moveaxis(turbu_opd_screen_cube, 2, 0)
                        n_turbu_cube = (turbu_opd_screen_cube.shape)[0]
                    t_turbu_cube = t_turbu_cube + 1
                    #t_count = 0

                    

                if len(turbu_opd_screen_cube) == 2 :
                    turbu_opd_screen_i = turbu_opd_screen_cube
                else:
                    turbu_opd_screen_i = turbu_opd_screen_cube[t_count]

                if turbu_remove_tt is True:
                        # z2 = get_zernike_screen((grid_width,grid_width), [1.], [2])
                        # z3 = get_zernike_screen((grid_width,grid_width), [1.], [3])
                        zernike_proj = zern.get_zernike_coeff(turbu_opd_screen_i, [2,3], res_level=1e-3)
                        print(zernike_proj['z_coeff'])
                        turbu_opd_screen_i = zernike_proj['res_wf']
                            
                t_count = t_count + turbu_delta_n_phase

            if (rotat_rms_nm != 0.) or (rotat_fits is True): # ==== rotate quasi-static aberration screen
                rotat_opd_screen_i = rotate_frame(rotat_opd_screen, sub_parangle[t])
#                rotat_opd_screen_i = rotat_opd_screen.copy()

            #### SPHERE JITTER ###
            if sphere_jitter is True:
                jitter_mas = sphere_jitter_data[:,i*n_img_mean+t] # [mas]
                jitter_rad = jitter_mas * 1e-3/3600.*np.pi/180.   # [radians]
                jitter_rms = jitter_rad * tel_diameter/4. * 1e6   # [microns rms]
                jitter_map = zern.get_zernike_screen((grid_width,grid_width),
                                                         [0., jitter_rms[0], jitter_rms[1]],
                                                         pup_diameter=pup_diameter)


            ###################### PLANET SIMULATOR ###########################
            # --------------------- SCIENCE PLANET ----------------------------
            if planet_ON is True:
                # Introduce an achromatic tilt to shift the image and simulate a planet
                planet_angle = 90.+ sub_parangle[t] - planet_pa
                planet_tilt  = zern.get_tilted_wavefront(pup_mask, pup_diameter,
                                                         angle = planet_angle)
                planet_tilt  *= planet_dist
            # ------------------ THROUGHPUT ESTIMATION ------------------------
            elif planet_throughput is True:
                planet_tilt  = planet_tilt0 * planet_sep_tab[i]

            ##################### TOTAL PRE-CORO OPD SCREEN ##################
            opd_screen = (turbu_opd_screen_i + static_opd_screen +
                          rotat_opd_screen_i + pre_zernike_opd_screen +
                          pre_aberr_opd      + jitter_map)

            ##################### LYOT DRIFT ##################
            if lyot_drift[0] != 0 or lyot_drift[1] != 0 :
                drift_x = round(lyot_drift[0] * (i*n_img_mean + t) * delta_t_ind_hour / 100. * pup_diameter)
                drift_y = round(lyot_drift[1] * (i*n_img_mean + t) * delta_t_ind_hour / 100. * pup_diameter)

                if silent is False :
                    print('Lyot stop drift [% of D_pup / pixels] = '+
                          'x {0:4.2f}/{1:4.2f} ; y {2:4.2f}/{3:4.2f}'
                          .format(lyot_shift[0] + lyot_drift[0] * (i*n_img_mean + t) * delta_t_ind_hour,
                                  shift_x + drift_x,
                                  lyot_shift[1] + lyot_drift[1] * (i*n_img_mean + t) * delta_t_ind_hour,
                                  shift_y + drift_y) )

                lyot_mask_drift = pad_and_roll(lyot_mask, (drift_x, drift_y))

                if vapp_off_axis_psf2 is True :
                    lyot_mask_drift_2 = pad_and_roll(lyot_mask_2, (drift_x, drift_y))
                if vapp_leakage != 0 :
                    lyot_mask_drift_0 = pad_and_roll(lyot_mask_0, (drift_x, drift_y))


            if progress_bar is True :
                wave_loop = trange(n_wave, desc='wave ')#, ascii=True)
            else :
                wave_loop = range(n_wave)

            for l in wave_loop : # ======= loop on wavelengths

                ####################### WAVEFRONT ########################
                # convert OPD into phase
                phase_screen = (opd_screen * 2. * np.pi / wave_tab[l] +
                                refraction_tilt * sub_refraction_lbdd[t,l] +
                                planet_tilt/wave_tab[l]*np.min(wave_tab))

                pre_phas_l = pre_phas * 2. * np.pi / wave_tab[l]

                ## 2018-02-08: Note: for PSF image, pupil planes are the same, only
                # the focal plane masks are removed -> simulates the off-axis PSF
                # at an infinite distance

                # INPUT WAVEFRONT
                wavefront  = (pup_mask * np.exp(1j * phase_screen) *
                              pre_apod * np.exp(1j * pre_phas_l) *
                              pre_amp )
                # wavefront for no coro PSF : the same
#                wavefront0 = wavefront.copy()

                focal_npix  = int(np.ceil(det_fov * det_sampling /2.)*2.)
                x1          = int(pup_diameter * det_sampling/2.-focal_npix/2.)
                x2          = int(pup_diameter * det_sampling/2.+focal_npix/2.)

                if post_ADC_ON is True:
                    # perfect refraction tilt correction is assumed
                    lyot_phase = np.exp(-1j*refraction_tilt*sub_refraction_lbdd[t,l] +
                                         1j*(post_zernike_opd_screen +
                                             post_aberr_opd) * 2.*np.pi / wave_tab[l])
                else:
                    # no correction of the atmospheric refraction
                    lyot_phase = np.exp(1j*(post_zernike_opd_screen + post_aberr_opd)
                                            *2.*np.pi / wave_tab[l])

                lyot_mask_l = (lyot_mask_drift * lyot_phase )
                if fp_mask == 'vapp' :
                    if vapp_off_axis_psf2 is True :
                        lyot_mask_l2 = (lyot_mask_drift_2 * lyot_phase )
                    if vapp_leakage != 0 :
                        lyot_mask_l0 = (lyot_mask_drift_0 * lyot_phase )

                ##############----- LYOT CORONAGRAPH -----###############
                if fp_mask == 'occulter':
                    occ_rad_pix  = (occulter_rad / wave_tab[l] * lbd0
                                    / pup_ratio * mft_sampling)

#                    occulter_fov = occulter_rad_max / pup_ratio * 2.5 ### 2019-03-26: change for larger FoV
                    occulter_fov = occulter_rad_max / pup_ratio * 4.
                    if occulter_fov == 0: # no occulting mask
                        occulter_fov = 1

                    propag_output = simu.propagate_mono_lyot(wavefront, lyot_mask_l,
                                                         occulter_fov=occulter_fov,
                                                         mft_sampling=mft_sampling,
                                                         occ_rad_pix=occ_rad_pix,
                                                         det_fov_pix=(x1,x2),
                                                         det_sampling=det_sampling*pup_ratio,
                                                         centering=centering,
                                                         lbd=wave_tab[l], lbd_ref=np.min(wave_tab),
                                                         lyot_output=lyot_output)
                    detector_img_mono = propag_output['img_detector']
                    if save_fits_lyot_before :
                        lyot_before_mono = propag_output['img_before_lyot']
                    if save_fits_lyot_after :
                        lyot_after_mono = propag_output['img_after_lyot']

                    if do_psf :
                        propag_mono_output= simu.propagate_mono_lyot(wavefront, lyot_mask_l,
                                                  det_fov_pix=(x1,x2), occ_rad_pix = 0.,
                                                  det_sampling=det_sampling*pup_ratio,
                                                  centering=centering,
                                                  lbd=wave_tab[l], lbd_ref=np.min(wave_tab))

                ###############----- VORTEX / FQPM CORONAGRAPH -----###############
                if ( (fp_mask == 'vortex') | (fp_mask == 'fqpm') |
                        (fp_mask == 'vapp') |(fp_mask=='paplc') |
                        (fp_mask == 'nomask') | (fp_mask == 'nocoro') ):
                    propag_output = simu.propagate_mono_vortex(wavefront, fp_phase_mask, lyot_mask_l,
                                                         fp_fov=fp_fov, fp_sampling=fp_sampling,
                                                         det_fov_pix=(x1,x2),
                                                         det_sampling=det_sampling*pup_ratio,
                                                         centering=centering,
                                                         lbd=wave_tab[l], lbd_ref=np.min(wave_tab),
                                                         lyot_output=lyot_output)
                    detector_img_mono = propag_output['img_detector']
                    if save_fits_lyot_before :
                        lyot_before_mono = propag_output['img_before_lyot']
                    if save_fits_lyot_after :
                        lyot_after_mono = propag_output['img_after_lyot']

                    if do_psf :
                        if ((fp_mask == 'vapp') | (fp_mask == 'nomask') | (fp_mask == 'nocoro'))  :
                            # no need to compute the 'non-coro' PSF in the case of the vAPP,
                            # since there is no coronagraphic mask
                            propag_mono_output = propag_output
                        else :
                            propag_mono_output= simu.propagate_mono_vortex(wavefront, no_mask, lyot_mask_l,
                                                    det_fov_pix=(x1,x2),
                                                    det_sampling=det_sampling*pup_ratio,
                                                    centering=centering,
                                                    lbd=wave_tab[l], lbd_ref=np.min(wave_tab))

                    if fp_mask == 'vapp':
                        ### vAPP case: computation of the second off-axis PSF
                        if vapp_off_axis_psf2 is True :
                            ### Second off-axis PSF, on the other side with lyot_mask_l2
                            propag_output = simu.propagate_mono_vortex(wavefront, fp_phase_mask, lyot_mask_l2,
                                                             fp_fov=fp_fov, fp_sampling=fp_sampling,
                                                             det_fov_pix=(x1,x2),
                                                             det_sampling=det_sampling*pup_ratio,
                                                             centering=centering,
                                                             lbd=wave_tab[l], lbd_ref=np.min(wave_tab),
                                                             lyot_output=lyot_output)
                            detector_img_mono_2 = propag_output['img_detector']

                            detector_img_mono = detector_img_mono/2. + detector_img_mono_2/2.

                        ### vAPP case: computation of the leakage term
                        if vapp_leakage != 0. :
                            ### Leakage term computed with lyot_mask_l0, which contains the
                            ### Lyot stop but not the vAPP phase (/!\ different from the
                            ### perfect PSF which does not contains the Lyot stop)
                            propag_output = simu.propagate_mono_vortex(wavefront, fp_phase_mask, lyot_mask_l0,
                                                             fp_fov=fp_fov, fp_sampling=fp_sampling,
                                                             det_fov_pix=(x1,x2),
                                                             det_sampling=det_sampling*pup_ratio,
                                                             centering=centering,
                                                             lbd=wave_tab[l], lbd_ref=np.min(wave_tab),
                                                             lyot_output=lyot_output)
                            detector_img_mono_0 = propag_output['img_detector']

                            detector_img_mono = detector_img_mono + vapp_leakage/100. * detector_img_mono_0

                    # if (fp_mask == 'nocoro'):
                    #     propag_output = simu.propagate_mono_perfect(wavefront, pup_mask, lyot_mask_l,
                    #                                          det_fov_pix=(x1,x2),
                    #                                          det_sampling=det_sampling*pup_ratio,
                    #                                          centering=centering,
                    #                                          lbd=wave_tab[l], lbd_ref=np.min(wave_tab),
                    #                                          lyot_output=lyot_output)
                    #     detector_img_mono = propag_output['img_detector']
                    #     if save_fits_lyot_before :
                    #         lyot_before_mono = propag_output['img_before_lyot']
                    #     if save_fits_lyot_after :
                    #         lyot_after_mono = propag_output['img_after_lyot']
                        
                ###############----- PERFECT CORONAGRAPH -----###############
                if (fp_mask == 'perfect'):
                    propag_output = simu.propagate_mono_perfect(wavefront, pup_mask, lyot_mask_l,
                                                         det_fov_pix=(x1,x2),
                                                         det_sampling=det_sampling*pup_ratio,
                                                         centering=centering,
                                                         lbd=wave_tab[l], lbd_ref=np.min(wave_tab),
                                                         lyot_output=lyot_output)
                    detector_img_mono = propag_output['img_detector']
                    if save_fits_lyot_before :
                        lyot_before_mono = propag_output['img_before_lyot']
                    if save_fits_lyot_after :
                        lyot_after_mono = propag_output['img_after_lyot']

                    if do_psf :
                        propag_mono_output= simu.propagate_mono_perfect(wavefront, pup_mask, lyot_mask_l, nocoro = True,
                                                    det_fov_pix=(x1,x2),
                                                    det_sampling=det_sampling*pup_ratio,
                                                    centering=centering,
                                                    lbd=wave_tab[l], lbd_ref=np.min(wave_tab))

                ### TO BE DEVELOPPED ###
#                ###############----- REFLECTIVE LYOT -----###############
#                if fp_mask == 'llowfs_vortex':
#                    detector_img_mono = simu.propagate_mono_vortex(wavefront, vortex_mask, 1.-lyot_mask,
#                                                         fp_fov=fp_fov, fp_sampling=fp_sampling,
#                                                         det_fov_pix=(x1,x2),
#                                                         det_sampling=det_sampling*pup_ratio)

                
                ############### CHECKING PLOTS ############################################
                ###########################################################################
                if (checking_plots is True and i == 0 and t==0 and l == 0):
#                    plot_time =  mgf.get_timestamp(stamp_format='full_tight')
                    plot_time = stamp_start
                    save_dir = output_directory + 'png_figures/'
                    p = Path(save_dir)
                    if (p.exists() is False and save_fits_img is True):
                        os.makedirs(save_dir)
                    if isinstance(lyot_mask,float):
                        disp_lyot_mask = pup_mask * 0.
                    else :
                        disp_lyot_mask = np.abs(lyot_mask).astype(float)
                    disp_lyot_mask_phase = np.arctan2(np.imag(disp_lyot_mask), np.real(disp_lyot_mask))
                    pupil_plane = {'pupil mask': [pup_mask,0],
                                   'pupil apod.': [pre_apod,0],
                                   'pupil phase' : [pre_phas, 1],
                                   'pupil amp' : [pre_amp, 0],
                                   'pre aberr': [pre_aberr_opd,1],
                                   'pre zernike': [pre_zernike_opd_screen,1],
                                   'static': [static_opd_screen,1],
                                   'quasi static': [rotat_opd_screen_i,1],
                                   'turbu': [turbu_opd_screen_i,1],
                                   'planet': [planet_tilt, 1],
                                   'sphere jitter': [jitter_map, 1],
                                   'lyot mask': [disp_lyot_mask,0],
                                  'lyot mask phase': [disp_lyot_mask_phase,1],
                                  'post zernike': [post_zernike_opd_screen,1],
                                  'post aberr': [post_aberr_opd, 1]}
                    plot_checks(pupil_plane, fp_mask_amp, fp_mask_phase,
                                    fp_mask_extent, lbd0, xyunit, propag_output,
                                    parangle_tab, zenith_dist_tab, ha_hours,
                                    save_dir+plot_time, save_plots = (save_fits_img&save_png) )
                ###########################################################################

#                detector_img_mono = propag_output['img_detector']
                if save_fits_lyot_before :
                    lyot_before_mono = propag_output['img_before_lyot']
                if save_fits_lyot_after :
                    lyot_after_mono = propag_output['img_after_lyot']
                if do_psf :
                    noco_mono= propag_mono_output['img_detector']

                # Coronagraphic image cube
                if (save_fits_poly and i==0) :
                    coro_polycube[l] = detector_img_mono / perf_factor_poly[l]

                if l == 0:
                    detector_img_poly = detector_img_mono / n_wave
                    if save_fits_lyot_before :
                        lyot_before_poly = lyot_before_mono / n_wave
                    if save_fits_lyot_after :
                        lyot_after_poly = lyot_after_mono / n_wave
                else :
                    detector_img_poly = detector_img_poly + detector_img_mono / n_wave
                    if save_fits_lyot_before :
                        lyot_before_poly = lyot_before_poly + lyot_before_mono / n_wave
                    if save_fits_lyot_after :
                        lyot_after_poly = lyot_after_poly + lyot_after_mono / n_wave

                # PSF image cube (no coro - no apod, no fp mask, no lyot -, with aberr)
                if do_psf :
                    if (save_fits_poly and i==0) :
                        noco_polycube[l] = noco_mono / perf_factor_poly[l]
                    if l == 0:
                        detector_psf_poly = noco_mono / n_wave
                    else :
                        detector_psf_poly = detector_psf_poly + noco_mono / n_wave


            # average over turbulence phase screens
            if t == 0:
                detector_img_turbu = detector_img_poly / n_img_mean
                if save_fits_lyot_before :
                    lyot_before_turbu = lyot_before_poly / n_img_mean
                if save_fits_lyot_after :
                    lyot_after_turbu = lyot_after_poly / n_img_mean
            else :
                detector_img_turbu = detector_img_turbu + detector_img_poly / n_img_mean
                if save_fits_lyot_before :
                    lyot_before_turbu = lyot_before_turbu + lyot_before_poly / n_img_mean
                if save_fits_lyot_after :
                    lyot_after_turbu = lyot_after_turbu + lyot_after_poly / n_img_mean

            if do_psf :
                if t == 0:
                    detector_psf_turbu = detector_psf_poly / n_img_mean
                else :
                    detector_psf_turbu = detector_psf_turbu + detector_psf_poly / n_img_mean

#            pbar_turb.next()
#        pbar_turb.clearln()
#        pbar_turb.bar_prefix('new')

        ### CUBE INITIALIZATION ###
        if i == 0 :
            sy, sx = detector_img_turbu.shape

            final_detector_img_cube = np.zeros((n_images, sy, sx), dtype='float32')

            if do_psf :
                final_detector_psf_cube = np.zeros((n_images, sy, sx), dtype='float32')
            else :
                final_detector_psf_cube = 0.

            if save_fits_lyot_before :
                lsy, lsx = lyot_before_turbu.shape
                final_lyot_before_cube = np.zeros((n_images, lsy, lsx), dtype='float32')
            if save_fits_lyot_after :
                lsy, lsx = lyot_after_turbu.shape
                final_lyot_after_cube = np.zeros((n_images, lsy, lsx), dtype='float32')

        # normalization by the maximum of the perfect PSF (no coro, no aberr)
        final_detector_img_cube[i] = detector_img_turbu / perf_factor

        if do_psf :
            final_detector_psf_cube[i] = detector_psf_turbu / perf_factor
        if save_fits_lyot_before :
            final_lyot_before_cube[i] = lyot_before_turbu
        if save_fits_lyot_after :
            final_lyot_after_cube[i] = lyot_after_turbu

#        bar_main.next()
#    bar_main.finish()

    ### TIMING ###
    t_end = time.time()
    stamp_end = get_timestamp(stamp_format='full')
    if silent is not True:
        print('\n\n\n'+stamp_end+ ' -- LOOP FINISHED')
    # total time
    total_time = t_end-t_start
    time_unit = 's'
    if total_time > 60. :
        total_time = total_time / 60.
        time_unit = 'min'

    ### SAVE FITS FILES ###
#    timestamp = mgf.get_timestamp(stamp_format='full_tight')
    timestamp = stamp_start
    final_output_path= output_directory+timestamp+f'_{fp_mask}'

    if save_fits_img is True:
        # Save the corresponding config file
        if silent is not True :
            print('Now saving FITS files...')

        # Extra config lines
        config_extra = {'date_start':stamp_start2, 'date_end':stamp_end,
                        'total_time ('+time_unit+')':(round(total_time*100.)/100.)}

        if static_rms_nm != 0:
            fits.writeto(final_output_path+'_static_OPD_screen.fits',
                           np.float32(static_opd_screen))
            config['aberrconfig']['static_file'] =timestamp+'_static_OPD_screen.fits'
        if rotat_rms_nm != 0 :
            fits.writeto(final_output_path+'_rotat_OPD_screen.fits',
                           np.float32(rotat_opd_screen))
            config['aberrconfig']['rotat_file'] = timestamp+'_rotat_OPD_screen.fits'
        if pre_zernike_ON is True:
            fits.writeto(final_output_path+'_pre-Zernike_OPD_screen.fits',
                           np.float32(pre_zernike_opd_screen))
        if post_zernike_ON is True:
            fits.writeto(final_output_path+'_post-Zernike_OPD_screen.fits',
                           np.float32(post_zernike_opd_screen))

        # Edit and save the config file
        config_newname = edit_n_write_config(config, config_extra,
                                path=output_directory, #+'misthic_config_files/',
                                timestamp=timestamp)

        fits.writeto(final_output_path+'_perfect_psf.fits',
                           perf_nocoro_poly)

        fits.writeto(final_output_path+'_image_cube.fits',
                           final_detector_img_cube)
        # save parallactic angle values
        t = Table([parangle_tab],names=('a'))
        t.write(final_output_path+'_parangle_tab.txt', format='ascii')

        # save planet sep for throughput
        if planet_throughput is True:
            t = Table([planet_sep_tab],names=('a'))
            t.write(final_output_path+'_planet_sep_tab.txt', format='ascii')

        if do_psf:
            fits.writeto(final_output_path+'_psf_cube.fits',
                           final_detector_psf_cube)
    else :
        config_newname = ''

    if save_fits_lyot_before :
        fits.writeto(final_output_path+'_lyot_before_cube.fits',
                           final_lyot_before_cube)
    if save_fits_lyot_after :
        fits.writeto(final_output_path+'_lyot_after_cube.fits',
                           final_lyot_after_cube)


    if (save_fits_poly) :
        # save polychromatic cube with individual monochromatic images
        fits.writeto(final_output_path+'_image_cube_polychro.fits', coro_polycube)
        if do_psf:
            fits.writeto(final_output_path+'_nocoro_psf_polychro.fits', noco_polycube)

        t = Table([wave_tab],names='l')
        t.write(final_output_path+'_wave_tab_polychro.txt', format='ascii')

#        fits.writeto(final_output_path+'_wave_tab_polychro.fits', wave_tab)
    ### --------------------------------------------------------------------------

    if silent is not True :
        print('FITS name reference: '+timestamp)
        print('\n--- Total Elapsed Time = {:4.2f} '.format(total_time)+time_unit+' ---')

        # number of monochromatic computations
        n_comp = n_images * n_wave * n_img_mean
        print('--- Time per mono comp = {:4.2f} '.format(total_time/n_comp)+time_unit+' ---')

    final_output = {'img_cube': final_detector_img_cube,
                    'psf_cube': final_detector_psf_cube,
                    'perf_psf': perf_nocoro_poly,
                    'parfile' : config_newname}

    return final_output
