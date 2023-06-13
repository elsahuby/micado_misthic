"""
MICADO MISTHIC test script for spectroscopic simulations.

The purpose is to simulate non coronagraphic images with/without aberrations 
(turbulence, residuals, static aberrations, Zernike polynomials.)

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

import getpass
user = getpass.getuser()


###### USER defined Params ###################################################
checking_plots = True
silent = False
## PARAMETER FILE: CORO CONFIG
paf_directory       = 'config_files/'
param_file          = paf_directory+"misthic_config_micado_NOCORO"
ind_coro            = param_file.find('micado_') + len('micado_')

if user == 'ehuby' :
    output_directory    = '/home/ehuby/WORK/SIMU/SIMU_MICADO/OUTPUT/'+param_file[ind_coro:]+'/'
    input_directory     = '/home/ehuby/WORK/SIMU/SIMU_MICADO/INPUT/'
elif user == 'paulina':
    output_directory    = 'your_path'+param_file[ind_coro:]+'/'
    input_directory     = 'your_path'

print('\n --- '+param_file+' ---')

### path to the configspec file (same as the misthic_func.py module)
configspec_file     = os.path.dirname(misthic_func.__file__)+'/misthic_configspec.ini'

### CONFIG file settings #####################################################
config              = ConfigObj(param_file+"_default.ini", configspec=configspec_file)
config_current      = param_file+"_current.ini"
config.filename     = config_current

# Simulation config
####### SIMULATION CONFIG ####################################################
config['input_directory']   = input_directory
config['save_fits_poly']    = True
config['simuconfig']['n_images'] = 1
### zenith distance
zenith_distance     = 50
### Wavelength
lbd0                = 2. # microns
n_wave              = 10
delta_lbd           = 1. # microns
#wave_tab    = ((np.arange(n_wave)+0.5)/(n_wave)-0.5)*delta_lbd + lbd0
lbd_min             = lbd0 + delta_lbd * (0.5 / n_wave - 0.5)
lbd_max             = lbd0 + delta_lbd * ((n_wave-0.5) / n_wave - 0.5)
### Field of view
det_fov             = 256 # lambda/D 
det_sampling        = 2. # pixels per lambda_min/D
### Note: Image size will be det_fov*det_sampling

### lamdba/D in mas:
tel_diameter        = float(config['simuconfig']['tel_diameter'])
l_over_d_min        = lbd_min * 1e-6 / tel_diameter * 180 / np.pi * 3600. * 1000. # [mas]
det_fov_arcsec      = det_fov * l_over_d_min/1000.
print(f'Full Field of view= {det_fov_arcsec:.1f} arcsec')

######################## EDIT PARAMETER FILE #################################
### Input aberrations
config['aberrconfig']['pre_aberr_fits'] = False
config['aberrconfig']['pre_amp_fits']   = False
config['aberrconfig']['turbu_ON']       = False
config['aberrconfig']['sphere_jitter']  = False
config['aberrconfig']['turbu_ON']       = False
config['aberrconfig']['pre_zernike_ON'] = False
config['aberrconfig']['rotat_fits']     = False
config['aberrconfig']['post_aberr_fits']= False
### --> include static NCPA (non common path aberrations)
config['aberrconfig']['static_fits']    = True
config['aberrconfig']['static_file']    = 'ABERR/20180406_215639_static_OPD_screen.fits'

### Dispersion & ADC
config['aberrconfig']['atm_refrac_ON'] = False
config['aberrconfig']['post_ADC_ON']   = False

# zenith distance
config['simuconfig']['zenith_distance'] = zenith_distance

# Wavelength
config['waveconfig']['lbd0']            = lbd0
config['waveconfig']['n_wave']          = n_wave
config['waveconfig']['delta_lbd']       = delta_lbd

# Detector settings
config['detectorconfig']['detector_fov']        = det_fov
config['detectorconfig']['detector_sampling']   = det_sampling

# data directory
if config['aberrconfig']['atm_refrac_ON'] is False:
    output_directory += f'{n_wave}ch_no_atm_disp'
    if config['aberrconfig']['static_fits']  :
        output_directory += '_NCPA'
else:
    output_directory += f'{n_wave}ch_zendist{zenith_distance:.0f}/'
    if config['aberrconfig']['static_fits']  :
        output_directory += 'NCPA'
config['output_directory'] = output_directory+'/'

# write down the config file
config.write()

# run the simulation
output    = misthic_func.run_misthic(config_current, do_psf = False, silent=silent,
                        checking_plots=checking_plots , progress_bar=False,
                            create_output_dir=True)
