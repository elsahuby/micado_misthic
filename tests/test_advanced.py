"""
MICADO MISTHIC advanced test script.

The purpose is to simulate coronagraphic images with aberrations (turbulence
residuals, static aberrations, Zernike polynomials.)
I used this script to generate grids of Zernike aberrations, in particular tip
and tilt.

@author: ehuby
"""

# import modules
import numpy as np
import pandas as pd
import os

# import misthic function
from micado_misthic import misthic_func
from micado_misthic.utils.timestamp import get_timestamp

from configobj import ConfigObj
from validate import Validator




## SIMULATION LIST
simulist = 'micado_simu_list_turbu.json'

## SELECT WAVELENGTH
wave_name_tab = ['J-mono', 'H-mono', 'K-mono']
wave_name_tab = ['J-mono']

## PARAMETER FILE: CORO CONFIG
paf_directory     = 'config_files/'

parameter_table = []

# parameter_table.append(paf_directory+"misthic_config_micado_CLC1")
parameter_table.append(paf_directory+"misthic_config_micado_CLC2")
#parameter_table.append(paf_directory+"misthic_config_micado_CLC0")
# parameter_table.append(paf_directory+"misthic_config_micado_vortex2")

nb_parameter = len(parameter_table)


#### CHOOSE THE ABERRATIONS
##### -- Residual turbulence #################################################
turbu_ON   = False 
n_cubes    = 11
j0         = 0
turbu_dn   = 50 # 1 image every turbu_dn images
n_images_per_cube = 60
remove_jitter = True

###### -- Static Aberrations #################################################
SEGM_ABERR = False

STAT_ABERR = False
STAT_ABERR2= False
LOZ_nb     = 0 #11
LO_HO      = 'LOZ-' #'HO-' # 'HO-' #'LOZ-'
AMPLI      = '60nm'

ROTA_ABERR = False
POST_ABERR = False
LYOT_DRIFT = False

STAT_ABERR_ZER = False
ZERN_BASIS = False
ZERN_RAMP  = False 
ZERN_GRID  = True

##### PARAMETER GENERATION ##################################################
if STAT_ABERR_ZER is True :
    z_init_nb = [16, 18, 20]
    n_order     = np.max(z_init_nb)
    
    # nmrms_to_mas = 0.0214
    # z_init_cf = [1./nmrms_to_mas/1000.] # um rms
    
    z_init_cf = [-0.01, 0.01, -.005]
    
    zerninit  = '_'.join([f'Z{z:0}-{z_init_cf[i]*1000.:03.0f}' for i,z in enumerate(z_init_nb)])

###########################################################
if ZERN_BASIS is True :
    zernike_num = [2,3]
    n_order     = np.max(zernike_num)
    coeff       = 0.200 # microns
        
###########################################################
if ZERN_RAMP is True :
    zn         = 3
    n_order    = zn
    coeff_min  = 0.      # microns
    coeff_max  = 0.8
    n_coeff    = 11 #41

###########################################################
if ZERN_GRID is True :
    zn         =[2] # [2,3]
    n_order    = np.max(zn)
    coeff_min  = -0.09      # microns
    coeff_max  =  0.09
    n_grid     =  2
    
    target_grid = False
    n_angle    = 12 # 12: every 30° 

###########################################################
if LYOT_DRIFT is True :
    lyot_init  = 0., 1.5 # in percent of the entrance pupil
    lyot_drift_rate = 0., 0
    n_cubes    = 1
    n_images_per_cube = 1
    delta_t = 60.
    
# # not used ################################################################
# jitter_ON  = False
# jitter_amp = '3mas'
# ampl_ON    = False
# LWE_ON     = False

####### SIMULATION CONFIG ####################################################
pixel_scale             = 1.5 #4. #1.5 #4. # mas [4. or 1.5 for MICADO]
det_fov                 = 128 # lamdba/D

local_test = True

### path to the configspec file (same as the misthic_func.py module)
configspec_file   = os.path.dirname(misthic_func.__file__)+'/misthic_configspec.ini'

if local_test is True :
    turbu_dir         = 'COMPASS/Turbu5_50hz_resized/'
else :
    turbu_dir         = '/volumes/hra/micado/Corono5_50hz_resized/'
        
for wave_name in wave_name_tab :

    ### Spectral band ############################################################
    if wave_name == 'K-mono' :
        lbd0, delta_lbd, n_wave       = 2.145, 0., 1
    if wave_name == 'H-mono' :
        lbd0, delta_lbd, n_wave       = 1.635, 0., 1
    if wave_name == 'J-mono' :
        lbd0, delta_lbd, n_wave       = 1.2475, 0., 1
    
    for pf in parameter_table:
    
        print('\n --- '+pf+' ---')
    
        new_output_dir = f'{wave_name}/'
    
        config  = ConfigObj(pf+"_default.ini", configspec=configspec_file)
    
        #output_directory = config['output_directory']
    
        ind_coro = pf.find('micado_') + len('micado_')
    
        if local_test is True :
            output_directory = '/home/ehuby/WORK/SIMU/SIMU_MICADO/OUTPUT/'+pf[ind_coro:]+'/'
            config['input_directory'] = '/home/ehuby/WORK/SIMU/SIMU_MICADO/INPUT/'
            checking_plots = True
        else :
            output_directory = '/data5/ehuby/SIMU_MICADO/OUTPUT/'+pf[ind_coro:]+'/'
            config['input_directory'] = '/data5/ehuby/SIMU_MICADO/INPUT/'
            checking_plots = False
    
        config_current = pf+"_current.ini"
        config.filename = config_current
    
        config['simuconfig']['n_images'] = 1
    
        config['aberrconfig']['pre_aberr_fits'] = False
        config['aberrconfig']['pre_amp_fits']   = False
        config['aberrconfig']['turbu_ON']       = False
        config['aberrconfig']['sphere_jitter']  = False
        config['aberrconfig']['turbu_ON']       = False
    
        config['waveconfig']['lbd0']                    = lbd0
        config['waveconfig']['n_wave']                  = n_wave
        config['waveconfig']['delta_lbd']               = delta_lbd
    
        # lamdba/D in mas:
        tel_diameter = float(config['simuconfig']['tel_diameter'])
        lbdd0 = lbd0 * 1e-6 / tel_diameter * 180 / np.pi * 3600. * 1000. # [mas]
        
        # Detector settings
        det_sampling                                    = lbdd0 / pixel_scale
        config['detectorconfig']['detector_fov']        = det_fov
        config['detectorconfig']['detector_sampling']   = det_sampling
    
        #-- turbulence residuals
        if turbu_ON is True :
            config['aberrconfig']['turbu_ON']       = True
            config['aberrconfig']['turbu_directory']= turbu_dir
            
            config['simuconfig']['n_images'] = n_images_per_cube
    
            #print(config['aberrconfig']['turbu_prefix'])
            
            config['aberrconfig']['turbu_delta_n_phase'] = turbu_dn
            temporal_sampling = 50./turbu_dn
    
            if remove_jitter is True :
                config['aberrconfig']['turbu_remove_tt'] = True
                string_jitter = '_rm-jitter'
            else : 
                config['aberrconfig']['turbu_remove_tt'] = False
                string_jitter = ''
     
            if new_output_dir == '':
                new_output_dir = f'turbu5_{temporal_sampling:02.0f}Hz{string_jitter}/'
            else :
                new_output_dir = new_output_dir+f'turbu5_{temporal_sampling:02.0f}Hz{string_jitter}/'
        else:
            if new_output_dir == '':
                new_output_dir = 'no_turbu/'
            else :
                new_output_dir = new_output_dir+'no_turbu/'
        
        # #-- jitter
        # if jitter_ON is True:
        #     config['aberrconfig']['sphere_jitter'] = True
        #     config['aberrconfig']['sph_jitter_file'] = 'ABERR/pfb_jitter_'+jitter_amp+'.fits'
        #     if new_output_dir == '':
        #         new_output_dir = 'jitt'+jitter_amp
        #     else:
        #        new_output_dir = new_output_dir+'_jitt'+jitter_amp
    
        #-- Aberrations
        file_name = []
            
        if SEGM_ABERR is True:
            config['aberrconfig']['pre_aberr_fits']= True
            config['aberrconfig']['pre_aberr_file'] = ['ABERR/Segment_aberration_micrometers_filtered_at_1perc.fits']
            file_name.append('segm-aberr')
                
        if STAT_ABERR is True:
            config['aberrconfig']['static_fits']= True
            if LOZ_nb != 0:
                config['aberrconfig']['static_file'] = f'ABERR/20180406_215639_static_OPD_screen_{LO_HO}{LOZ_nb:02}.fits'
                file_name.append(f'static_{LO_HO}{LOZ_nb:02}')
            elif AMPLI != '':
                config['aberrconfig']['static_file'] = f'ABERR/20180406_215639_static_OPD_screen_{AMPLI}.fits'
                file_name.append(f'static_{AMPLI}')
            else :
                config['aberrconfig']['static_file'] = 'ABERR/20180406_215639_static_OPD_screen.fits'
                file_name.append('static')

        elif STAT_ABERR2 is True:
            config['aberrconfig']['static_fits']= True
            config['aberrconfig']['static_file'] = 'ABERR/20180408_084425_static_OPD_screen.fits'
            file_name.append('static2')
            
        if STAT_ABERR_ZER is True:
            config['aberrconfig']['pre_zernike_ON'] = True
            n_order = np.max([n_order, np.max(z_init_nb)])
            zernike_coeff_init = np.zeros(n_order)
            for i, z in enumerate(z_init_nb):
                zernike_coeff_init[z-1] = z_init_cf[i]
            file_name.append(zerninit)
        else :
            n_order = 10
            zernike_coeff_init = np.zeros(n_order)
                
        if ROTA_ABERR is True:
            config['aberrconfig']['rotat_fits']= True
            config['aberrconfig']['rotat_file'] = 'ABERR/20180406_215639_rotat_OPD_screen.fits'
            file_name.append('rotat')
        
        if POST_ABERR is True:
            config['aberrconfig']['post_aberr_fits']= True
            config['aberrconfig']['post_aberr_file'] = 'ABERR/20180406_215639_static_Post-Coro_screen.fits'
            file_name.append('post-coro')
        
        # #-- amplitude aberrations
        # if ampl_ON is True :
        #     config['aberrconfig']['pre_amp_fits'] = True
        #     config['aberrconfig']['pre_amp_file'] = 'ABERR_AMP/sphere_pupil_clear_BH_field.fits'
        #     file_name.append('amp-aberr')
    
        # #-- low wind effect
        # if LWE_ON is True :
        #     config['aberrconfig']['turbu_ON'] = True
        #     config['aberrconfig']['turbu_directory']= 'LWE/'
        #     config['aberrconfig']['turbu_prefix'] = '2019-04-09_Sphere_LWE_microns'
        #     file_name.append('lwe')
        
        #-- lyot drift
        if LYOT_DRIFT is True:
            config['coroconfig']['lyot_shift'] = lyot_init
            config['coroconfig']['lyot_drift'] = lyot_drift_rate
            
            config['simuconfig']['delta_t'] = delta_t
            config['simuconfig']['n_images'] = n_images_per_cube
            
            file_name.append(f'lyot-drift-x{lyot_init[0]:2.1f}-'+
                             f'y{lyot_init[1]:2.1f}-'+
                             f'dx{lyot_drift_rate[0]:2.1f}-'+
                             f'dy{lyot_drift_rate[1]:2.1f}')
        
        if file_name == []:
            file_name.append('no_aberr')
        
        new_output_dir = new_output_dir + '_'.join(file_name) + '/'   
    
    
        if ZERN_BASIS is True :
            config['aberrconfig']['pre_zernike_ON'] = True
            
            new_output_dir0 = new_output_dir
            
            for zn in zernike_num :
            
                pre_z_coeff = np.zeros((n_order+1))
                pre_z_coeff[zn-1] = coeff
            
                # include Zernike coeff in config file : 
                # [piston, tt_hori, tt_verti, focus, astig1, astig2, coma1, coma2, Z9, Z10, ...]
                config['aberrconfig']['pre_z_coeff'] = list(pre_z_coeff)
    
                if new_output_dir == '':
                    new_output_dir = 'Zernike_Z{0:0d}_{1:03.0f}nmrms'.format(zn, coeff*1000.)
                else :
                    new_output_dir = new_output_dir0+'Zernike_Z{0:0d}_{1:03.0f}nmrms'.format(zn, coeff*1000.)
                
                config['output_directory'] = output_directory + new_output_dir + '/'
                
                for j in range(n_cubes) :
    
                    if turbu_ON is True:
                        # config['aberrconfig']['turbu_prefix'] = ('Turbu_reshape_cube50hz_{0:03d}'.format(j0+j))
                        config['aberrconfig']['turbu_prefix'] = ('Turbu_reshape_cube50hz_')
            
                        print(config['aberrconfig']['turbu_directory'])
                        print(config['aberrconfig']['turbu_prefix'])
            
                    config.write()
            
                    output    = misthic_func.run_misthic(config_current, do_psf = True, silent=False,
                                            checking_plots=checking_plots , progress_bar=not(local_test),
                                            create_output_dir=True)
        elif ZERN_RAMP is True :
            config['aberrconfig']['pre_zernike_ON'] = True
            
            new_output_dir0 = new_output_dir
            n_cubes         = n_coeff
            
            coeff_ramp = np.linspace(coeff_min, coeff_max, n_cubes)
            #pre_z_coeff = np.zeros((n_cubes, zn))
            pre_z_coeff = np.tile(zernike_coeff_init, (n_cubes, 1))
            pre_z_coeff[:,zn-1] = coeff_ramp
            
            #### conversion of the pre_z_coeff array into a pandas data frame
            col_names = ['Z'+str(i+1) for i in range(n_order)]
            z_pd = pd.DataFrame(pre_z_coeff, columns=col_names)
    
            ramp_dir = f'Zernike_ramp_Z{zn:0d}_{n_coeff:02d}_{coeff_min*1000.:04.0f}nmrms_{coeff_max*1000.:04.0f}nmrms/'
            if new_output_dir == '':
                new_output_dir = ramp_dir
            else :
                new_output_dir = new_output_dir0+ramp_dir
            
            config['output_directory'] = output_directory + new_output_dir + '/'
            
            for j in range(n_cubes) :
                
                # include Zernike coeff in config file : 
                # [piston, tt_hori, tt_verti, focus, astig1, astig2, coma1, coma2, Z9, Z10, ...]
                pre_z_coeff_j = pre_z_coeff[j]
                config['aberrconfig']['pre_z_coeff'] = list(pre_z_coeff_j)
    
                if turbu_ON is True:
                    # config['aberrconfig']['turbu_prefix'] = ('Turbu_reshape_cube50hz_{0:03d}'.format(j0+j))
                    config['aberrconfig']['turbu_prefix'] = (f'Turbu_reshape_cube50hz_{j0:0d}')
        
                    print(config['aberrconfig']['turbu_directory'])
                    print(config['aberrconfig']['turbu_prefix'])
        
                config.write()
        
                output    = misthic_func.run_misthic(config_current, do_psf = True, silent=False,
                                        checking_plots=checking_plots , progress_bar=not(local_test),
                                        create_output_dir=True)
            
            #### -- save the Zernike coefficients
            final_output_dir = output['parfile'][:output['parfile'].find('current')]
            z_pd.to_csv(final_output_dir + 'aberr_z_coeff_um.txt', sep='\t', float_format = '%.3f')
    
        elif ZERN_GRID is True :
            config['aberrconfig']['pre_zernike_ON'] = True
            
            new_output_dir0 = new_output_dir
            
            # ampl = np.random.normal(loc=0, scale=coeff_max, size=n_cubes)
            # thet = np.random.uniform(low=0, high=2*np.pi, size=n_cubes)
            
            # coeff_z2 = ampl * np.cos(thet)
            # coeff_z3 = ampl * np.sin(thet)
            # #pre_z_coeff = np.zeros((n_cubes, zn))
            # pre_z_coeff = np.tile(zernike_coeff_init, (n_cubes, 1))
            # pre_z_coeff[:,2-1] = coeff_z2
            # pre_z_coeff[:,3-1] = coeff_z3
            
            if target_grid is True:
                n_cubes = n_grid * n_angle + 1
                
                ampl = np.linspace(coeff_max/n_grid, coeff_max, n_grid)
                thet = np.linspace(0., 2.*np.pi * (n_angle - 1)/n_angle , n_angle) 
                
                coeff_z2 = np.append([0], np.outer(ampl, np.cos(thet)).ravel())
                coeff_z3 = np.append([0], np.outer(ampl, np.sin(thet)).ravel())                
            else :                
                n_cubes         = n_grid**2
                
                coeff_z2, coeff_z3 = (np.indices((n_grid,n_grid))/(n_grid-1.)*(coeff_max-coeff_min)+coeff_min)
                
                coeff_z2 = coeff_z2.ravel()
                coeff_z3 = coeff_z3.ravel()
            
            pre_z_coeff = np.tile(zernike_coeff_init, (n_cubes, 1))
            pre_z_coeff[:,2-1] += coeff_z2
            pre_z_coeff[:,3-1] += coeff_z3
            
            #### conversion of the pre_z_coeff array into a pandas data frame
            col_names = ['Z'+str(i+1) for i in range(n_order)]
            z_pd = pd.DataFrame(pre_z_coeff, columns=col_names)
    
            
            dir1 = 'Zernike_TTgrid/'
            if target_grid is True:
                dir2 = (f'TTtargetgrid-{n_grid:0d}x{n_angle:0d}_'+
                        f'{coeff_max*1000.:04.0f}nmrms/')
            else:
                dir2 = (f'TTgrid-{n_grid:0d}x{n_grid:0d}_'+
                        f'{coeff_min*1000.:04.0f}_{coeff_max*1000.:04.0f}nmrms/')
            if new_output_dir == '':
                new_output_dir = dir1+dir2
            else :
                new_output_dir = new_output_dir0+dir1+dir2
            
            if pixel_scale != 1.5 :
                new_output_dir = new_output_dir[:-1] + f'_PScale_{pixel_scale:.1f}mas/'
                
            config['output_directory'] = output_directory + new_output_dir 
                
    
            #### -- save the Zernike coefficients
            
            save_name = (get_timestamp(stamp_format='full_tight') + 
                         f'_random_tt_N{n_cubes:0d}_{coeff_max*1000.:04.0f}nmrms_aberr_z_coeff_um.txt')
            
            
            for j in range(n_cubes) :
                
                # include Zernike coeff in config file : 
                # [piston, tt_hori, tt_verti, focus, astig1, astig2, coma1, coma2, Z9, Z10, ...]
                pre_z_coeff_j = pre_z_coeff[j]
                config['aberrconfig']['pre_z_coeff'] = list(pre_z_coeff_j)
    
                if turbu_ON is True:
                    # config['aberrconfig']['turbu_prefix'] = ('Turbu_reshape_cube50hz_{0:03d}'.format(j0+j))
                    config['aberrconfig']['turbu_prefix'] = ('Turbu_reshape_cube50hz_')
                    config['aberrconfig']['turbu_file_nb_init'] = j0
        
                    print(config['aberrconfig']['turbu_directory'])
                    print(config['aberrconfig']['turbu_prefix'])
        
                config.write()
        
                output    = misthic_func.run_misthic(config_current, do_psf = True, silent=False,
                                        checking_plots=checking_plots , progress_bar=not(local_test),
                                        create_output_dir=True)
            
            #### -- save the Zernike coefficients
            final_output_dir = output['parfile'][:output['parfile'].find('current')]
            z_pd.to_csv(final_output_dir + 'aberr_z_coeff_um.txt', sep='\t', float_format = '%.3f')
            
            z_pd.to_csv(config['output_directory']+save_name, sep='\t', float_format = '%.3f')
        
        else :
            config['aberrconfig']['pre_z_coeff'] = list(zernike_coeff_init)
            
            config['output_directory'] = output_directory + new_output_dir + '/'
            
            for j in range(n_cubes) :
        
                if turbu_ON is True:
                    # config['aberrconfig']['turbu_prefix'] = ('Turbu_reshape_cube50hz_{0:03d}'.format(j0+j))
                    #config['aberrconfig']['turbu_prefix'] = ('Turbu_reshape_cube50hz_')
                    config['aberrconfig']['turbu_prefix'] = (f'Turbu_reshape_cube50hz_')
                    config['aberrconfig']['turbu_file_nb_init'] = j0 + j*30
        
                    print(config['aberrconfig']['turbu_directory'])
                    print(config['aberrconfig']['turbu_prefix'])
        
                config.write()
        
                output    = misthic_func.run_misthic(config_current, do_psf = True, silent=False,
                                        checking_plots=checking_plots , progress_bar=not(local_test),
                                        create_output_dir=True)
