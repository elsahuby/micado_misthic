"""
MICADO MISTHIC basic test script.

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

## path tristan
tristan_main_path = 'C:/Users/tdeseine/Desktop/MISTHIC/'

## PARAMETER FILE: CORO CONFIG
paf_directory     =tristan_main_path +  'micado_misthic/tests/config_files/'

# put the specific config file you need here
config_file = "misthic_config_micado_CLC0"

pf = paf_directory+config_file+'_default.ini'
# pf = paf_directory+"misthic_config_micado_CLC0"
# pf = paf_directory+"misthic_config_micado_CLC2"
# pf = paf_directory+"misthic_config_micado_vortex2"
# pf = paf_directory+"misthic_config_micado_CLC1"
print('\n --- '+pf+' ---')

###### USER defined Params ###########################################################
ind_coro = pf.find('micado_') + len('micado_')
output_directory = tristan_main_path + 'output/'+pf[ind_coro:]+'/'
checking_plots = True
wave_name = 'J-mono'

####### SIMULATION CONFIG ####################################################
pixel_scale             = 1.5 #4. #1.5 #4. # mas [4. or 1.5 for MICADO]
det_fov                 = 128 # lamdba/D

local_test = True

### path to the configspec file (same as the misthic_func.py module)
configspec_file   = os.path.dirname(misthic_func.__file__)+r'/misthic_configspec.ini'


### Spectral band ############################################################
if wave_name == 'K-mono' :
    lbd0, delta_lbd, n_wave       = 2.145, 0., 1
if wave_name == 'H-mono' :
    lbd0, delta_lbd, n_wave       = 1.635, 0., 1
if wave_name == 'J-mono' :
    lbd0, delta_lbd, n_wave       = 1.2475, 0., 1

### CONFIG file settings #####################################################
config              = ConfigObj(pf, configspec=configspec_file)

config_current      = pf+"_current.ini"
config.filename     = config_current

# Simulation config
config['input_directory'] = tristan_main_path +  'INPUT/'
config['simuconfig']['n_images'] = 50
config['delta_t'] = 30 
# Simulation aberrations
config['aberrconfig']['pre_aberr_fits'] = False
config['aberrconfig']['pre_amp_fits']   = False
config['aberrconfig']['turbu_ON']       = False
config['aberrconfig']['sphere_jitter']  = False
config['aberrconfig']['turbu_ON']       = False
config['aberrconfig']['static_fits']    = False
config['aberrconfig']['pre_zernike_ON'] = False
config['aberrconfig']['rotat_fits']     = False
config['aberrconfig']['post_aberr_fits']= False

# Wavelength
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

# data directory
config['output_directory'] = output_directory + 'misthic_test/'

# write down the config file
config.write()

# run the simulation
output    = misthic_func.run_misthic(config_current, do_psf = True, silent=False,
                        checking_plots=checking_plots , progress_bar=not(local_test),
                            create_output_dir=True)
