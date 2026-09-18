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

wave_name = 'H-mono'


## PARAMETER FILE: CORO CONFIG

# put the path to your config files here
paf_directory     = 'C:/Users/Red Slottje/Documents/Repo git misthic/tests/config_files/'

# put the specific config file you need here
config_file = "misthic_config_micado_CLC0"

pf = paf_directory+config_file+'_default.ini'

## Output and Input directories here
output_directory = 'C:/Users/Red Slottje/Documents/Misthic_outputs/'
input_directory = 'C:/Users/Red Slottje/Documents/Misthic_inputs/'


###### -- Static Aberrations #################################################
SEGM_ABERR = False

STAT_ABERR = False
STAT_ABERR2= False

ROTA_ABERR = False
POST_ABERR = False

STAT_ABERR_ZER = False

ZERN_GRID  = False

##### PARAMETER GENERATION ##################################################
if STAT_ABERR_ZER is True :
    z_init_nb = [16, 18, 20]
    n_order     = np.max(z_init_nb)
    
    # nmrms_to_mas = 0.0214
    # z_init_cf = [1./nmrms_to_mas/1000.] # um rms
    
    z_init_cf = [-0.01, 0.01, -.005]
    
    zerninit  = '_'.join([f'Z{z:0}-{z_init_cf[i]*1000.:03.0f}' for i,z in enumerate(z_init_nb)])

###########################################################
if ZERN_GRID is True :
    zn         =[2] # [2,3]
    n_order    = np.max(zn)
    coeff_min  = -0.09      # microns
    coeff_max  =  0.09
    n_grid     =  2
    
    target_grid = False
    n_angle    = 12 # 12: every 30° 


####### SIMULATION CONFIG ####################################################
pixel_scale             = 1.5 #4. #1.5 #4. # mas [4. or 1.5 for MICADO]
det_fov                 = 128 # lamdba/D

local_test = True

### path to the configspec file (same as the misthic_func.py module)
    #using os.path.dirname in Windows puts the path with antislashes, which Python can't interpret
#configspec_file   = os.path.dirname(misthic_func.__file__)+'/misthic_configspec.ini'
configspec_file = 'C:/Users/Red Slottje/Documents/Repo git misthic/micado_misthic/misthic_configspec.ini'

### Spectral band ############################################################
if wave_name == 'K-mono' :
    lbd0, delta_lbd, n_wave       = 2.145, 0., 1
if wave_name == 'H-mono' :
    lbd0, delta_lbd, n_wave       = 1.635, 0., 1
if wave_name == 'J-mono' :
    lbd0, delta_lbd, n_wave       = 1.2475, 0., 1


print('\n --- '+pf+' ---')

new_output_dir = f'{wave_name}/'

config  = ConfigObj(pf, configspec=configspec_file)

#output_directory = config['output_directory']

ind_coro = pf.find('micado_') + len('micado_')

if local_test is True :
    checking_plots = True
else :
    checking_plots = False

config_current = pf+"_current.ini"
config.filename = config_current

config['simuconfig']['n_images'] = 10
config['simuconfig']['delta_t'] = 600


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

new_output_dir = 'no_turbu/'

#-- Aberrations
file_name = []
    
if SEGM_ABERR is True:
    config['aberrconfig']['pre_aberr_fits']= True
    config['aberrconfig']['pre_aberr_file'] = ['ABERR/Segment_aberration_micrometers_filtered_at_1perc.fits']
    file_name.append('segm-aberr')
        
if STAT_ABERR is True:
    config['aberrconfig']['static_fits']= True
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


if file_name == []:
    file_name.append('no_aberr')

new_output_dir = new_output_dir + '_'.join(file_name) + '/'   


if ZERN_GRID is True :
    config['aberrconfig']['pre_zernike_ON'] = True
    
    new_output_dir0 = new_output_dir    
           
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

    config.write()

    output    = misthic_func.run_misthic(config_current, do_psf = True, silent=False,
                            checking_plots=checking_plots , progress_bar=not(local_test),
                            create_output_dir=True)