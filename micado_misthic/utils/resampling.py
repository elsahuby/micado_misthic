from micado_misthic.imgproc import fft_resize
import numpy as np
from astropy.io import fits
import os
import shutil

sampling_factor = [1.5, 4.0]
hard_drive_path = r'D:\NEW_PHASE2021'
output_main_directory = r'C:\Users\tdeseine\Desktop\MISTHIC\output\opti_tristan'
filter_directories = ["filter1.190", "filter1.245", "filter1.270", "filter1.582","filter1.635" ,"filter1.693", "filter2.100", "filter2.145", "filter2.235"]
D=38.542 #m

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


def get_directory_output(hard_drive_path, output_main_directory, folder):
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

    if not folder.startswith('CLC'):
        raise ValueError(f"Unexpected folder name: {folder}")

    parts = folder.split('_')
    CLC = parts[0]
    quartile = parts[1]
    quasi_static_speckle = parts[2] if len(parts) >= 3 and parts[2] else 'NCPA'

    return os.path.join(output_main_directory, CLC, quasi_static_speckle, quartile)



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


def resampling(output_main_directory, initial_sampling_factor, sampling_factor = sampling_factor): 
    """
    This function aims to automatically resample the simulation image to the desire sampling rate (1.5 mas and 4 mas)

    Parameters
    ----------
    output_main_directory : str
        Path where the resampled image will be saved
    
    initial_sampling_factor : float
        The initial sampling factor of the image
    sampling_factor : list of float, optional
        The desired sampling factors (MICADO is [1.5, 4.0])
    """

    for folder in os.listdir(hard_drive_path):
        folder_path = os.path.join(hard_drive_path, folder)
        if not os.path.isdir(folder_path):
            continue

        output_directory_base = get_directory_output(hard_drive_path, output_main_directory, folder)

        for f in os.listdir(folder_path):
            if 'Lambda=' not in f:
                continue

            filter_str, wavelength = parse_filter_dir(f, filter_directories)
            
            output_directory_filter = os.path.join(output_directory_base, filter_str)

            subfolder_path = os.path.join(folder_path, f)
            if not os.path.isdir(subfolder_path):
                continue

            for files in os.listdir(subfolder_path):
                if not files.endswith('.fits'):
                    continue

                filename = os.path.splitext(files)[0]
                image_path = os.path.join(subfolder_path, files)

                # Try to open and read the FITS file
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
                        output_directory = os.path.join(output_directory_filter, f"Samp{sf}")
                        output_path = os.path.join(output_directory, filename + '.fits')

                        # Ensure the output directory exists so auxiliary files can be copied even if the FITS already exists.
                        os.makedirs(output_directory, exist_ok=True)
                        copy_aux_files(subfolder_path, output_directory)

                        if os.path.exists(output_path):
                            print(f"Skipping {output_path}: already exists.")
                            continue

                        print(
                            f"Resampling the files of the folder {folder} with the filter {wavelength} to the sampling factor {sf} mas."
                        )
                        
                        resizeddata, _ = fft_resize(image_data, wavelength, 
                                                    initial_sampling_factor,
                                                    sf,
                                                    write_dir=0,
                                                    plotting=False)
                        fits.writeto(output_path, resizeddata, overwrite=True)

                            
if __name__ == "__main__":
    resampling(output_main_directory, initial_sampling_factor=2.0)                                           
