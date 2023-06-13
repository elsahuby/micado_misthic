# -*- coding: utf-8 -*-

import os
import inspect

from ..utils import get_timestamp



def get_configspec_path():

#    # Search path from PYTHON PATH
#    pp = sys.path
#    misthic_path = []
#    for p in pp :
#        if p.find('misthic') > -1 :
#            misthic_path.append(p)
#    if len(misthic_path) > 1 :
#        print('Error : There are several misthic folders in python path.')
#    elif len(misthic_path) == 0:
#        print('Error : No misthic folder found in python path.')
#        misthic_path = ''
#    else :
#        misthic_path = misthic_path[0]

    # Search path from misthic_func path

    misthic_path = os.path.split(inspect.getfile(get_configspec_path))[0]+'/'

    return misthic_path


def edit_n_write_config(config, new_section_items, path = '', new_filename=None, timestamp=None):
    """
    Edit and write the given config file.

    Parameters
    ----------
    config: ConfigObj
        ConfigObj parameter object.
    new_section_items: dict
        dictionary with keys corresponding to the item name, and corresponding
        value.
    path = ''
        path where the config file will be saved
    new_filename = None
        new name for the config file if desired. If None (default), the same
        name as the current config file is used, preceded by a time stamp
        (given or automatically generated).
    timestamp = None
        Time stamp used at the beginning of the config file name.
        If None (default), the time stamp is automatically generated.

    Returns
    -------
    Parameter file new name.
    """

    if new_filename is None:
        name_part = (config.filename.split)('/')[-1]
        new_filename = (name_part.split)('.')[0]

    if timestamp is None:
        timestamp = get_timestamp(stamp_format='full_tight')

    config_newname = path + timestamp + '_' + new_filename + '.ini'
    config.filename = config_newname

    config['post-simu data'] = {}

    for k in new_section_items :
        config['post-simu data'][k] = new_section_items[k]

    config.write()

    return config_newname