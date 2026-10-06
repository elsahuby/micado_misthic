"""
Author: Tristan Deseine
"""

import os
import shutil
from configobj import ConfigObj
from pathlib import Path
import numpy as np
from astropy.io import fits
import re


def find_input_file(directory, pattern):
    """
    This function find an input file according to a specific pattern. 
    It can be used to find the .ini config file or a simulation.

    Input:
    directory: str
        The directory where to search for the file.
    pattern: str
        The pattern of the filename to search for (e.g. ".ini").

    Output:
    path: Path
        The path to the first file found that matches the pattern.
    """

    return next(path for path in Path(directory).glob(pattern)
                if not path.name.startswith("._"))


def copy_aux_files(source_dir, dest_dir):
    """
    This function copies auxiliary files into an output directory.
    It copies configuration files ending in ".ini" and files whose names
    contain "parangle" or "parall". Files starting with "._" are ignored.

    If the source directory does not exist, the function returns without
    copying anything. Copy failures are printed.

    Input:
    source_dir: str
        The directory containing the auxiliary files to copy.

    dest_dir: str
        The existing destination directory where the files should be copied.
    """

    if not os.path.isdir(source_dir):
        return

    for filename in os.listdir(source_dir):
        if filename.startswith('._'):
            continue
        lower_name = filename.lower()
        if (lower_name.endswith('.ini')
                or 'parangle' in lower_name
                or 'parall' in lower_name):
            src = os.path.join(source_dir, filename)
            dst = os.path.join(dest_dir, filename)
            try:
                shutil.copy2(src, dst)
                print(f"Copied auxiliary file {filename} to {dest_dir}")
            except Exception as e:
                print(f"Failed to copy {src} to {dst}: {e}")


def select_observation_time(desired_time, cube, angles, delta_t):

    """
    This function selects consecutive frames from an already loaded cube.
    The number of frames is calculated by rounding the desired time multiplied
    by 60 and divided by the exposure time.

    The selection is centred on the frame whose parallactic angle is closest
    to zero, corresponding to the passage through the meridian. The selection
    is shifted if necessary to stay within the available frames.

    If the requested duration exceeds the available data, the whole cube is
    returned.

    Input:
    desired_time: float
        The requested observation time in minutes.
    cube: numpy.ndarray
        The NumPy array containing the images, with frames along the first axis.
    angles: numpy.ndarray
        The array of parallactic angles corresponding to the cube frames.
    delta_t: float
        The exposure time in seconds.

    Output:
    selected_cube: numpy.ndarray
        The slice of the cube containing the selected frames.
    selected_angles: numpy.ndarray
        The slice of the angle array corresponding to the selected frames.
    """

    desired_seconds = float(desired_time) * 60
    Number_of_img = round(desired_seconds / delta_t)

    # At meridian passage, the parallactic angle is 0 (near to 0 here)
    meridian_index = np.argmin(np.abs(angles))

    start = meridian_index - Number_of_img // 2
    # Ensure that we verify 0<start<cube.shape[0]-selected_count
    start = max(0, min(start, cube.shape[0] - Number_of_img)) 
    stop = start + Number_of_img

    return cube[start:stop], angles[start:stop]

def format_observation_time(duration_seconds):
    """
    This function converts an effective observation duration into a readable
    folder label after rounding to the nearest second.

    It is used for naming ADI output directories.

    Input:
    duration_seconds: float
        The effective observation duration in seconds.

    Output:
    label: str
        The string used to name the ADI output folder.
    """

    rounded_seconds = round(duration_seconds)
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        label = f"{hours}h"
        if minutes:
            label += f"{minutes:02d}"
        if seconds:
            label += f"min{seconds:02d}s"
        return label
    if minutes:
        return f"{minutes}min" if not seconds else f"{minutes}min{seconds:02d}s"
    return f"{seconds}s"


REQUIRED_FITS = ("image_cube", "perfect_psf", "psf_cube")


def parse_simulation_metadata(folder, filter_dir):
    """
    This function extracts the simulation parameters from directory names.
    It accepts both CLC_seeing_ncpa and CLC_ncpa_seeing ordering.

    The seeing value is converted to uppercase and the NCPA value is normalised
    to "NCPA" or "NoNCPA".

    Input:
    folder: str
        The simulation folder.
    filter_dir: str
        The directory name containing the wavelength as "Lambda=..."
        (e.g. "Lambda=2.2").

    Output:
    corono: str
        The coronagraph identifier extracted from the first part of the folder name.
    seeing: str
        The seeing value in uppercase.
    ncpa: str
        The NCPA value, normalised to "NCPA" or "NoNCPA".
    filter_str: str
        The filter name built from the wavelength (e.g. "filter2.2").
    wavelength: numpy.float64
        The wavelength in microns.

    Warning:
        This function may need to be adapted if directory naming conventions
        change or if additional simulation parameters must be extracted from
        directory names.
    """
    parts = folder.split('_')
    if not parts or not parts[0].upper().startswith('CLC'):
        raise ValueError(f"Unexpected folder name: {folder}")

    corono = parts[0]
    seeing = next(
        (part for part in parts[1:]
         if re.fullmatch(r"Q\d+|MED(?:IAN)?", part, re.IGNORECASE)),
        None,
    )
    ncpa_value = next(
        (part for part in parts[1:] if 'ncpa' in part.lower()),
        'NCPA',
    )
    if seeing is None:
        raise ValueError(f"No seeing value found in folder name: {folder}")

    seeing = seeing.upper()
    ncpa = 'NoNCPA' if ncpa_value.casefold() == 'noncpa' else 'NCPA'

    match = re.search(
        r"lambda=([0-9]+(?:\.[0-9]+)?)", filter_dir, re.IGNORECASE
    )
    if match is None:
        raise ValueError(f"No Lambda=... value found in: {filter_dir}")

    wavelength = match.group(1)
    return corono, seeing, ncpa, f"filter{wavelength}", np.float64(wavelength)

def discover_simulations(input_directory, planet_on):
    """
    This function recursively searches the input directory for simulation
    folders containing FITS filenames matching each required name:
    image_cube, perfect_psf and psf_cube.

    It ignores files starting with "._" and skips incomplete simulations or
    simulations with invalid metadata.

    Each branch must contain a Lambda=... directory, and its first directory
    below the input directory must describe the coronagraph, seeing and NCPA
    parameters. Optional z=... and p_dist_mas=... directories can appear at
    any position along the branch.

    Input:
    input_directory: str
        The root directory containing the simulation branches.
    planet_on: int
        Select simulations with a planet (1) or without a planet (0).
        Simulations with a planet are identified by a p_dist_mas=... directory
        in their path.

    Output:
    simulations: list
        The list of dictionaries, one per matching simulation.
        Each dictionary contains source_dir, corono, seeing, ncpa, z, filter,
        wavelength and distance.
        The z and distance values retain their directory names, or are None
        when absent. The list is empty if no simulation matches.

    Warning:
        This function may need to be adapted if new simulation parameters are
        added, directory naming conventions change, or an unsupported directory
        structure is introduced.
    """
    simulations = []

    for current_dir, _, filenames in os.walk(input_directory):
        relative_path = os.path.relpath(current_dir, input_directory)
        branch = relative_path.split(os.sep)
        lambda_folder = next(
            (part for part in branch if 'lambda=' in part.lower()), None)
        if lambda_folder is None:
            continue

        fits_filenames = [
            filename.lower() for filename in filenames
            if filename.lower().endswith('.fits')
            and not filename.startswith('._')
        ]
        missing_fits = [
            fits for fits in REQUIRED_FITS
            if not any(fits in filename for filename in fits_filenames)
        ]
        if missing_fits:
            if fits_filenames or os.path.basename(current_dir) == lambda_folder:
                print(
                    f"Skipping incomplete branch {current_dir}: missing "
                    f"{', '.join(missing_fits)}"
                )
            continue

        distance_folder = next(
            (part for part in branch
             if part.lower().startswith('p_dist_mas=')),
            None,
        )
        has_planet = distance_folder is not None
        if has_planet != (planet_on == 1):
            continue

        try:
            corono, seeing, ncpa, filter_str, wavelength = parse_simulation_metadata(
                branch[0], lambda_folder
            )
        except ValueError as error:
            print(f"Skipping branch {current_dir}: {error}")
            continue

        z_folder = next(
            (part for part in branch if part.lower().startswith('z=')),
            None,
        )
        simulations.append({
            'source_dir': current_dir,
            'corono': corono,
            'seeing': seeing,
            'ncpa': ncpa,
            'z': z_folder,
            'filter': filter_str,
            'wavelength': wavelength,
            'distance': distance_folder,
        })

    return simulations



def get_initial_sampling_factor(simulation_dir):
    """
    This function uses find_input_file() to locate the single .ini configuration
    file in a simulation directory.

    It reads detector_sampling from the [detectorconfig] section and converts
    it to a float.

    Input:
    simulation_dir: str
        The directory containing the single simulation .ini configuration file.

    Output:
    detector_sampling: float
        The initial detector sampling factor and the wavelength it applies to (minimal lambda simulated instead of centre wavelength)
    """

    try:
        config_path = find_input_file(simulation_dir, "*.ini")
        config = ConfigObj(str(config_path))
        lbd0 = float(config['waveconfig']['lbd0'])
        delta_lbd = float(config['waveconfig']['delta_lbd'])
        n_wave = float(config['waveconfig']['n_wave'])
        if n_wave > 1 :
            lbd_min      = np.min(((np.arange(n_wave)+0.5)/(n_wave)-0.5)*delta_lbd + lbd0)
        else:
            lbd_min      = np.min(np.array([lbd0]))
        return float(config["detectorconfig"]["detector_sampling"]), lbd_min
    except (StopIteration, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"No valid .ini containing [detectorconfig] detector_sampling "
            f"was found in {simulation_dir}"
        ) from error


def matches_selection(simulation, corono, ncpa, seeing):
    """
    This function checks whether a simulation matches the requested coronagraph,
    NCPA and seeing values.

    Comparisons are case-insensitive. A selection set to None is ignored, so
    that parameter does not restrict the selection.

    This function is used when we do not want to resample all simulations.

    Input:
    simulation: dict
        The dictionary containing the simulation metadata.
    corono: str or None
        The requested coronagraph identifier, or None to accept any coronagraph.
    ncpa: str or None
        The requested NCPA value, or None to accept either value.
    seeing: str or None
        The requested seeing value, or None to accept any seeing.

    Output:
    match: bool
        True if all specified criteria match, otherwise False.
    """

    for key, selected in (("corono", corono), ("ncpa", ncpa), ("seeing", seeing)):
        if selected is None:
            continue
        if simulation[key].casefold() != str(selected).casefold():
            return False
    return True


def get_output_directory(output_main_directory, simulation, planet_on):
    
    """
    This function builds the output directory path from the simulation metadata.

    The path follows the order:
    output_main_directory / With_Planet or No_Planet / corono / ncpa /
    seeing / z / filter / distance.

    The z, filter and distance components are omitted when their values are None.

    Input:
    output_main_directory: str
        The root directory for the preprocessed data.
    simulation: dict
        The simulation dictionary containing corono, ncpa, seeing, z, filter
        and distance.
    planet_on: int
        Set to 1 to use the With_Planet directory, or 0 to use No_Planet.

    Output:
    output_directory: str
        The constructed output path.
        The function does not create the directory.

    Warning:
        This function may need to be adapted if new simulation parameters are
        added or if the order of the tree configuration changes.
    """
    
    planet_directory = "With_Planet" if planet_on == 1 else "No_Planet"
    parts = [
        output_main_directory,
        planet_directory,
        simulation['corono'],
        simulation['ncpa'],
        simulation['seeing'],
    ]
    for key in ('z', 'filter', 'distance'):
        if simulation[key] is not None:
            parts.append(simulation[key])
    return os.path.join(*parts)
