# -*- coding: utf-8 -*-

import numpy as np
import json
import os
import inspect

from configobj import ConfigObj
from validate import Validator


def sphere_simulist_append(parameter_file, simulist_file):
    """
    Generate a json file with a dictionary containing relevant simulation
    parameters in the context of SPHERE UPGRADE.

    Version
    -------
    v1: EHu 2018-12-19

    """
    #--- load the simulist if it exists
    try :
        with open(simulist_file, 'r') as fp:
            simulist = json.load(fp)
    except:
        simulist = {}

    if type(parameter_file) is str:
        parameter_file = np.expand_dims(parameter_file, 0)

    do_write = False

    for pfile in parameter_file:
        if pfile != '':
            #-- update the simulist with data from the parameter file
            #--- load parameters from parameter file
            configspec_file   = os.path.split(inspect.getfile(sphere_simulist_append))[0]+'/misthic_configspec.ini'
            config    = ConfigObj(pfile, configspec=configspec_file)
            vtor              = Validator()
            checks            = config.validate(vtor,copy=True)

            #-- coro type
            coro = config['coroconfig']['fp_mask']

            if coro == 'occulter':
                occulter_rad = config['coroconfig']['occulter']['occulter_rad']
                if occulter_rad == 92.5 :
                    name = 'ALC2'
                elif occulter_rad == 120 :
                    name = 'ALC3'
                else:
                    name='NC'
            elif coro == 'vortex':
                vcharge = config['coroconfig']['vortex']['v_charge']
                name = 'vortex' + str(int(vcharge))
            else :
                name = coro

            if config['coroconfig']['pre_apod_fits'] is True:
                file = config['coroconfig']['pre_apod_file']
                p = file.find('lovD')
                apod_num = file[p-3:p]
                if apod_num == '4.0':
                    name = 'APO1 '+name
                elif apod_num == '5.2':
                    name = 'APO2 '+name
                elif file.find('vlt_APLC') > -1 :
                    name = 'new_aplc'
                elif file.find('ring') > -1 :
                    name = 'RA '+name

            if config['coroconfig']['pre_phas_fits'] is True:
                file = config['coroconfig']['pre_phas_file']
                if file.find('HDH') > -1 :
                    name = name+' HDH'
                    if file.find('02') > -1 :
                        name = name+'2'
                    if file.find('03') > -1 :
                        name = name+'3'
                    if file.find('10pixFromCenter6') > -1 :
                        name = name+'-v201912'
                elif file.find('FDH') > -1 :
                    name = name+' FDH'

            #-- spectral band
            lbd0 = config['waveconfig']['lbd0']
            delta_lbd = config['waveconfig']['delta_lbd']
            n_wave = config['waveconfig']['n_wave']
            if lbd0 == 2.2:
                band = 'K band'
            elif (lbd0 == 1.68) or (lbd0 == 1.667) :
                band = 'H band'
            elif lbd0 == 1.3:
                band = 'IFS J (Sres30)'
            elif lbd0 == 1.:
                band = 'IFS Y (Sres30)'
            elif lbd0 == 1.6:
                if delta_lbd == 0.053:
                    band = 'IFS H (Sres30)'
                elif n_wave == 1:
                    band = 'H mono'
                else :
                    band = 'H band (1.6)'
            else:
                band = 'NC'

            #-- turbulence
            if config['aberrconfig']['turbu_ON'] is True:
                turbu_prefix = config['aberrconfig']['turbu_prefix']
                turb = ''
                if turbu_prefix.find('CAOS_AOC') > -1 :
                    turb = 'CAOS'
                if turbu_prefix.find('SAXO2') > -1 :
                    turb = turb+' SAXO2'
                if turbu_prefix.find('seeing') > -1:
                    turb = turb+' s'+turbu_prefix[turbu_prefix.find('seeing')+6:turbu_prefix.find('seeing')+10]
                if turbu_prefix.find('LWE') > -1 : #-- LWE
                    turb = 'LWE'
                if turb == '' :
                    turb = 'NC'

            else :
                turb = ''

            #-- jitter
            if config['aberrconfig']['sphere_jitter'] is True:
                jitter_file = config['aberrconfig']['sph_jitter_file']
                i1  = jitter_file.rfind('mas')
                i2 = jitter_file.rfind('_', 0, i1)
                jitt = jitter_file[i2+1:i1+3]
            else:
                jitt=''

            #-- NCPA
            dact = ''
            ncpa = ''
            if config['aberrconfig']['pre_aberr_fits'] is True:
                file = config['aberrconfig']['pre_aberr_file']
                total_file = ''.join(file)
                if total_file.find('deadact') > -1 :
                    dact = '6DA'
                if total_file.find('m123') > -1 :
                    ncpa += 'M123 '
                    if total_file.find('loop_opd_it0') > -1 :
                        ncpa += 'ZELDAit0'
                    elif total_file.find('loop_opd_it3') > -1 :
                        ncpa += 'ZELDAit3'
                elif total_file.find('loop_opd_it0') > -1 :
                    ncpa += 'ZELDAit0'
                elif total_file.find('loop_opd_it3') > -1 :
                    ncpa += 'ZELDAit3'
                elif total_file.find('opd_iter=01_microns') > --1 :
                    ncpa += 'ZELDAit1'
                if total_file.find('LWE') > -1 : #-- LWE
                    ncpa += 'LWE'

            ampl = ''
            if config['aberrconfig']['pre_amp_fits'] is True:
                file = config['aberrconfig']['pre_amp_file']
                total_file = ''.join(file)
                if total_file.find('sphere_pupil') > -1 :
                    ampl = 'ampl_aberr'

            if config['aberrconfig']['static_fits'] is True:
                file = config['aberrconfig']['static_file']
                if file.find('deadact') > -1 :
                    dact = '6DA'
                elif file.find('m123') > -1 :
                    ncpa = 'M123'
                elif file.find('loop_opd_it0') > -1 :
                    ncpa = 'ZELDAit0'
                elif file.find('loop_opd_it3') > -1 :
                    ncpa = 'ZELDAit3'


            #-- throughput data?
            thru = ''
            if config['planetconfig']['planet_throughput'] is True:
                dist = config['planetconfig']['dist_minmax']
                nimg = config['simuconfig']['n_images']
                thru = '{0}-{1}-{2}'.format(dist[0],dist[1],nimg)

            #-- data stamp
            i1 = pfile.rfind('/')
            data_stamp = pfile[i1+1:i1+16]
            simulist[data_stamp] = {'coro' : name,
                                    'band' : band,
                                    'turb' : turb,
                                    'jitt' : jitt,
                                    'dact' : dact,
                                    'ncpa' : ncpa,
                                    'thru' : thru,
                                    'ampl' : ampl}
            do_write = True

    #-- write the updated simulist
    if do_write is True :
        with open(simulist_file, 'w') as fp:
            json.dump(simulist, fp, indent=4)

        print('\nSimulist successfully written in\n'+simulist_file)
        print(simulist[data_stamp])

    return simulist[data_stamp]

def sphere_simulist_print(simulist_file, select=None):
    with open(simulist_file, 'r') as fp:
        simulist = json.load(fp)

    output_list= []

    for simu_num in simulist:
        simu_param= ' ; '.join(simulist[simu_num].values())
        if select is not None:
            if type(select) == str:
                select = np.expand_dims(select, 0)
            nb_sel = len(select)
            c = 0
            found=True
            for sel in select:
                if (simu_num+simu_param).find(sel) > -1 and found is True:
                    c += 1
                    found=True
                else :
                    found=False
            if c == nb_sel :
                output_list.append(simu_num+' ; '+ simu_param)
        else :
            output_list.append(simu_num+' ; '+simu_param)

    return output_list

def micado_simulist_append(parameter_file, simulist_file):
    """
    Generate a json file with a dictionary containing relevant simulation
    parameters in the context of MICADO coronagraph simulations.

    Version
    -------
    v1: EHu 2021-02-04

    """
    #--- load the simulist if it exists
    try :
        with open(simulist_file, 'r') as fp:
            simulist = json.load(fp)
    except:
        simulist = {}

    if type(parameter_file) is str:
        parameter_file = np.expand_dims(parameter_file, 0)

    do_write = False

    for pfile in parameter_file:
        if pfile != '':
            #-- update the simulist with data from the parameter file
            #--- load parameters from parameter file
            configspec_file   = os.path.split(inspect.getfile(sphere_simulist_append))[0]+'/misthic_configspec.ini'
            config    = ConfigObj(pfile, configspec=configspec_file)
            vtor              = Validator()
            checks            = config.validate(vtor,copy=True)

            #-- coro type
            coro = config['coroconfig']['fp_mask']
            
            #-- nb of images
            nimg = str(config['simuconfig']['n_images'])

            if coro == 'occulter':
                occulter_rad = config['coroconfig']['occulter']['occulter_rad']
                if occulter_rad == 25.34 :
                    name = 'CLC1'
                elif occulter_rad == 50.68 :
                    name = 'CLC2'
                else:
                    name='NC'
            elif coro == 'vortex':
                vcharge = config['coroconfig']['vortex']['v_charge']
                name = 'vortex' + str(int(vcharge))
            else :
                name = coro

            #-- spectral band
            lbd0 = config['waveconfig']['lbd0']
            delta_lbd = config['waveconfig']['delta_lbd']
            n_wave = config['waveconfig']['n_wave']
            if n_wave == 1:
                if lbd0 == 2.145:
                    band = 'K-mono'
                elif lbd0 == 1.635:
                    band = 'H-mono'
                elif lbd0 == 1.2475:
                    band = 'J-mono'
            else:
                band = 'NC'

            #-- turbulence
            if config['aberrconfig']['turbu_ON'] is True:
                turbu_prefix = config['aberrconfig']['turbu_prefix']
                turbu_delta_n_phase = config['aberrconfig']['turbu_delta_n_phase']
                turb = ''
                if turbu_prefix.find('50hz') > -1 :
                    temporal_sampling = 50./turbu_delta_n_phase
                    turb = f'{temporal_sampling:02.0f}Hz'
                if turb == '' :
                    turb = 'NC'
            else :
                turb = 'no-turbu'

            # #-- jitter
            jitt=''
            # if config['aberrconfig']['sphere_jitter'] is True:
            #     jitter_file = config['aberrconfig']['sph_jitter_file']
            #     i1  = jitter_file.rfind('mas')
            #     i2 = jitter_file.rfind('_', 0, i1)
            #     jitt = jitter_file[i2+1:i1+3]
            # else:
            #     jitt=''

            #-- throughput data?
            thru = ''
            if config['planetconfig']['planet_throughput'] is True:
                dist = config['planetconfig']['dist_minmax']
                nimg = config['simuconfig']['n_images']
                thru = '{0}-{1}-{2}'.format(dist[0],dist[1],nimg)
                
            #-- data stamp
            i1 = pfile.rfind('/')
            data_stamp = pfile[i1+1:i1+16]
            simulist[data_stamp] = {'coro' : name,
                                    'band' : band,
                                    'nimg' : nimg,                                    
                                    'turb' : turb,
                                    'jitt' : jitt,
                                    'thru' : thru}
            do_write = True

    #-- write the updated simulist
    if do_write is True :
        with open(simulist_file, 'w') as fp:
            json.dump(simulist, fp, indent=4)

        print('\nSimulist successfully written in\n'+simulist_file)
        print(simulist[data_stamp])

    return simulist[data_stamp]

def micado_simulist_print(simulist_file, select=None):
    with open(simulist_file, 'r') as fp:
        simulist = json.load(fp)

    output_list= []

    for simu_num in simulist:
        simu_param= ' ; '.join(simulist[simu_num].values())
        if select is not None:
            if type(select) == str:
                select = np.expand_dims(select, 0)
            nb_sel = len(select)
            c = 0
            found=True
            for sel in select:
                if (simu_num+simu_param).find(sel) > -1 and found is True:
                    c += 1
                    found=True
                else :
                    found=False
            if c == nb_sel :
                output_list.append(simu_num+' ; '+ simu_param)
        else :
            output_list.append(simu_num+' ; '+simu_param)

    return output_list