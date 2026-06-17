from micado_misthic.imgproc import fft_resize
import numpy as np
from astropy.io import fits
from configobj import ConfigObj
import os
import shutil

sampling_factor = [1.5, 4.0]
planet_on = 1
hard_drive_path = r'D:\planet_sims'
output_main_directory = r'D:\ressempled data'
filter_directories = ["filter1.190", "filter1.245", "filter1.270", "filter1.582","filter1.635" ,"filter1.693", "filter2.100", "filter2.145", "filter2.235"]
D=38.542 #m


def parse_simulation_folder(folder):
    """Extract coronagraph, seeing and NCPA values from a simulation folder."""
    parts = folder.split('_')
    if len(parts) < 2 or not parts[0].startswith('CLC'):
        raise ValueError(f"Unexpected folder name: {folder}")

    corono = parts[0]
    seeing = parts[1]
    ncpa = parts[2] if len(parts) >= 3 and parts[2] else 'NCPA'
    return corono, seeing, ncpa


def matches_simulation_selection(folder, corono=None, ncpa=None, seeing=None):
    """Return whether a simulation folder matches the requested selection."""
    folder_corono, folder_seeing, folder_ncpa = parse_simulation_folder(folder)
    selections = (
        (folder_corono, corono),
        (folder_ncpa, ncpa),
        (folder_seeing, seeing),
    )
    return all(
        selected is None or value.casefold() == str(selected).casefold()
        for value, selected in selections
    )


def parse_filter_dir(filter_dir, filter_directories = filter_directories):
    """
    Extract the wavelength from a Lambda=... folder name and validate the filter.

    Parameters
    ----------
    filter_dir : str
        Folder name expected to start with 'Lambda='
    filter_directories : list
        Valid filter directory names

    Returns
    -------
    tuple[str, str]
        The validated filter folder name and the extracted wavelength string.
    """
    wavelength = filter_dir.split('=', 1)[1]
    filter_str = f"filter{wavelength}"
    if filter_str not in filter_directories:
        raise ValueError(f"Filter {wavelength} not in the list of filter directories.")

    return filter_str, np.float64(wavelength)


def get_directory_output(hard_drive_path, output_main_directory, folder, planet_on=planet_on):
    """
    Parse the CLC folder metadata and return the base output path for that folder.

    Parameters
    ----------
    hard_drive_path : str
        Path to the hard drive containing the images to be resampled
    output_main_directory : str
        Path where the resampled images will be saved
    folder : str
        Folder name under hard_drive_path starting with 'CLC'
    """

    CLC, quartile, quasi_static_speckle = parse_simulation_folder(folder)

    planet_directory = "With_Planet" if planet_on == 1 else "No_Planet"
    return os.path.join(
        output_main_directory,
        planet_directory,
        CLC,
        quasi_static_speckle,
        quartile,
    )



def copy_aux_files(source_dir, dest_dir):
    """Copy parallactic angle table and config files into the output directory."""
    if not os.path.isdir(source_dir):
        return

    for filename in os.listdir(source_dir):
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


def get_initial_sampling_factor(simulation_dir):
    """Read detector_sampling from a simulation configuration file."""
    for filename in os.listdir(simulation_dir):
        if not filename.lower().endswith(".ini"):
            continue

        config_path = os.path.join(simulation_dir, filename)
        try:
            config = ConfigObj(config_path)
            return float(config["detectorconfig"]["detector_sampling"])
        except (KeyError, TypeError, ValueError):
            continue

    raise ValueError(
        f"No valid .ini containing [detectorconfig] detector_sampling "
        f"was found in {simulation_dir}"
    )


def get_simulation_directories(filter_dir, planet_on):
    """Return simulation directories and their optional planet-distance folder."""
    if planet_on != 1:
        return [(filter_dir, None)]

    simulation_directories = []
    for folder_name in os.listdir(filter_dir):
        distance_dir = os.path.join(filter_dir, folder_name)
        if not os.path.isdir(distance_dir) or not folder_name.startswith("p_dist_mas="):
            continue

        try:
            float(folder_name.split("=", 1)[1])
        except ValueError:
            print(f"Skipping invalid planet distance folder: {distance_dir}")
            continue

        simulation_directories.append((distance_dir, folder_name))

    return simulation_directories


def resampling(
        output_main_directory,
        sampling_factor=sampling_factor,
        planet_on=planet_on,
        corono=None,
        ncpa=None,
        seeing=None):
    """
    This function aims to automatically resample the simulation image to the desire sampling rate (1.5 mas and 4 mas)

    Parameters
    ----------
    output_main_directory : str
        Path where the resampled image will be saved
    
    sampling_factor : list of float, optional
        The desired sampling factors (MICADO is [1.5, 4.0])
    planet_on : int, optional
        Set to 1 to process simulations stored in p_dist_mas=... directories.
    corono : str, optional
        Only process this coronagraph (for example "CLC1").
    ncpa : str, optional
        Only process this NCPA configuration (for example "NoNCPA").
    seeing : str, optional
        Only process this seeing value (for example "Q4" or "MED").

    If this 3 last parameters are set to None, all files will be processed.
    """

    processed_files = 0
    found_filters = 0
    print(
        f"Starting resampling from {hard_drive_path} "
        f"({'With_Planet' if planet_on == 1 else 'No_Planet'} mode), "
        f"selection: corono={corono or 'all'}, ncpa={ncpa or 'all'}, "
        f"seeing={seeing or 'all'}"
    )

    for folder in os.listdir(hard_drive_path):
        folder_path = os.path.join(hard_drive_path, folder)
        if not os.path.isdir(folder_path):
            continue
            

        # Skip folders that don't match the requested simulation selection, but print a message if the folder name is not in the expected format.
        try:
            if not matches_simulation_selection(folder, corono, ncpa, seeing):
                continue
        except ValueError as e:
            print(f"Skipping folder {folder_path}: {e}")
            continue

        output_directory_base = get_directory_output(
            hard_drive_path,
            output_main_directory,
            folder,
            planet_on=planet_on,
        )

        for f in os.listdir(folder_path):
            if 'lambda=' not in f.lower():
                continue

            found_filters += 1
            filter_str, wavelength = parse_filter_dir(f, filter_directories)
            
            output_directory_filter = os.path.join(output_directory_base, filter_str)

            filter_path = os.path.join(folder_path, f)
            if not os.path.isdir(filter_path):
                continue

            simulation_directories = get_simulation_directories(filter_path, planet_on)
            if planet_on == 1 and not simulation_directories:
                print(f"No p_dist_mas=... folders found in {filter_path}")

            for simulation_path, distance_folder in simulation_directories:
                try:
                    initial_sampling_factor = get_initial_sampling_factor(simulation_path)
                    print(
                        f"Initial sampling factor read from config: "
                        f"{initial_sampling_factor} pixels per lambda/D"
                    )
                except ValueError as e:
                    print(f"Skipping folder {simulation_path}: {e}")
                    continue

                for files in os.listdir(simulation_path):
                    if not files.endswith('.fits'):
                        continue

                    filename = os.path.splitext(files)[0]
                    image_path = os.path.join(simulation_path, files)

                    try:
                        with fits.open(image_path) as hdul:
                            image_data = hdul[0].data
                    except Exception as e:
                        print(f"Skipping file {image_path}: cannot open/read FITS ({e})")
                        continue

                    if image_data is None:
                        print(f"Skipping file {image_path}: no data in FITS")
                        continue

                    for sf in sampling_factor:
                        # Check if the desired sampling factor is achievable given the wavelength and telescope diameter, if not, skip the resampling for that case
                        if wavelength*10**-6/D *2.063*10**8  < 2*sf:
                            print(f"Desired sampling factor {sf} mas is not achievable for wavelength {wavelength} microns. Skipping resampling for this case.")
                            continue
                        output_parts = [output_directory_filter]
                        if distance_folder is not None:
                            output_parts.append(distance_folder)
                        output_parts.append(f"Samp{sf}")
                        output_directory = os.path.join(*output_parts)
                        output_path = os.path.join(output_directory, filename + '.fits')

                        # Ensure the output directory exists so auxiliary files can be copied even if the FITS already exists.
                        os.makedirs(output_directory, exist_ok=True)
                        copy_aux_files(simulation_path, output_directory)

                        if os.path.exists(output_path):
                            print(f"Skipping {output_path}: already exists.")
                            continue

                        print(
                            f"Resampling {folder}, filter {wavelength}, "
                            f"distance {distance_folder or 'no planet'} to {sf} mas."
                        )
                        
                        resizeddata, _ = fft_resize(image_data, wavelength, 
                                                    initial_sampling_factor,
                                                    sf,
                                                    write_dir=0,
                                                    plotting=False)
                        fits.writeto(output_path, resizeddata.astype(np.float32), overwrite=True)
                        processed_files += 1

    print(
        f"Resampling completed: {found_filters} filter folders found, "
        f"{processed_files} FITS files written."
    )

                            
if __name__ == "__main__":
    resampling(output_main_directory, corono="CLC1", ncpa="NoNCPA", seeing="Q1")
