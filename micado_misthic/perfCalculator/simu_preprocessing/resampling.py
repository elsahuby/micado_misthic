"""
Author: Tristan Deseine
"""

import os
import re
import shutil

import numpy as np
from astropy.io import fits
from micado_misthic.imgproc import fft_resize
from micado_misthic.perfCalculator.simu_preprocessing.file_utils import *


D = 38.542  # Telescope diameter in meters.


def resampling(
        output_main_directory,
        input_directory,
        sampling_factor,
        planet_on,
        corono=None,
        ncpa=None,
        seeing=None,
        ):
    """
    Resample simulation images to the requested sampling rates in mas.

    Only the image_cube, perfect_psf and psf_cube FITS products are needed and
    resampled. Other FITS products, such as Lyot-plane images, are ignored.

    Input:
    output_main_directory: str
        The root directory where the resampled simulations will be saved.
    input_directory: str
        The root directory containing the input simulation branches.
    sampling_factor: list
        The list of target pixel scales in milliarcseconds per pixel
        (e.g. [1.5, 4.0]).
    planet_on: int
        Select simulations with a planet (1) or without a planet (0).
        Simulations with a planet are identified by a p_dist_mas=... directory
        in their path. This parameter also determines whether outputs are stored
        under With_Planet or No_Planet.
    corono: str or None
        The requested coronagraph identifier (e.g. "CLC1"), or None to accept
        any coronagraph.
    ncpa: str or None
        The requested NCPA value ("NCPA" or "NoNCPA"), or None to accept either value.
    seeing: str or None
        The requested seeing value, or None to accept any seeing.
    """

    processed_files = 0
    print(
        f"Starting resampling from {input_directory} "
        f"({'With_Planet' if planet_on == 1 else 'No_Planet'} mode), "
        f"selection: corono={corono or 'all'}, ncpa={ncpa or 'all'}, "
        f"seeing={seeing or 'all'}"
    )

    simulations = discover_simulations(input_directory, planet_on)

    for simulation in simulations:
        if not matches_selection(simulation, corono, ncpa, seeing):
            continue

        simulation_path = simulation['source_dir']
        wavelength = simulation['wavelength']

        output_directory_filter =get_output_directory(
            output_main_directory, simulation, planet_on
        )

        try:
            initial_sampling_factor = get_initial_sampling_factor(simulation_path)
        except ValueError as error:
            print(f"Skipping folder {simulation_path}: {error}")
            continue

        prepared_directories = set()
        for fits_filename in os.listdir(simulation_path):
            lower_filename = fits_filename.lower()
            if (fits_filename.startswith('._')
                    or not lower_filename.endswith('.fits')
                    or not any(product in lower_filename
                               for product in REQUIRED_FITS)):
                continue

            filename = os.path.splitext(fits_filename)[0]
            image_path = os.path.join(simulation_path, fits_filename)

            try:
                with fits.open(image_path) as hdul:
                    image_data = hdul[0].data
            except Exception as error:
                print(f"Skipping file {image_path}: {error}")
                continue

            if image_data is None:
                print(f"Skipping file {image_path}: no data in FITS")
                continue

            for sf in sampling_factor:
                if wavelength * 10**-6 / D * 2.063 * 10**8 < 2 * sf:
                    print(
                        f"Sampling {sf} mas is not achievable for "
                        f"{wavelength} microns. Skipping.")
                    continue

                output_directory = os.path.join(
                    output_directory_filter, f"Samp{sf}")
                output_path = os.path.join(
                    output_directory, filename + '.fits')
                if output_directory not in prepared_directories:
                    os.makedirs(output_directory, exist_ok=True)
                    copy_aux_files(simulation_path, output_directory)
                    prepared_directories.add(output_directory)

                if os.path.exists(output_path):
                    print(f"Skipping {output_path}: already exists.")
                    continue

                print(
                    f"Resampling {simulation_path}, filter {wavelength}, "
                    f"distance {simulation['distance'] or 'no planet'} "
                    f"to {sf} mas.")
                resizeddata, _ = fft_resize(
                    image_data,
                    wavelength,
                    initial_sampling_factor,
                    sf,
                    write_dir=0,
                    plotting=False,
                )
                fits.writeto(
                    output_path,
                    resizeddata.astype(np.float32),
                    overwrite=True,
                )
                processed_files += 1

    print(
        f"Resampling completed: "
        f"{processed_files} FITS files written."
    )



