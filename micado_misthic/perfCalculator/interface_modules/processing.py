"""
Author: Tristan Deseine
"""

import os
import time
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox
from astropy.io import fits


from micado_misthic.imgproc import get_circle_mask
from micado_misthic.utils.obsparams import scale_to_photons
from micado_misthic.perfCalculator.utils import (
    add_noise_after_adi,
    convolve_adi_image,
    find_simulation_fits,
    get_planet_contrast,
    normalize_by_psf_max,
    reconstruct_coronographic_image,
)
from micado_misthic.imgproc import micado_adi


class ProcessingMethods:
    """Stellar and planetary ADI processing helpers for the interface."""

    def initialize_processing_run(self, params, path, apply_noise, noise_stage, convolved, output_dir):
        """Store the parameters and clear results from the previous run.

        Input:
        params: dict
            Parameters collected when processing started.
        path: Path
            Directory containing the selected simulation products.
        apply_noise: bool
            Whether noise was requested.
        noise_stage: str
            Processing stage at which noise is applied.
        convolved: bool
            Whether this processing chain applies convolution.
        output_dir: Path or None
            Existing output directory.

        Output:
        values: tuple
            Copied parameters, normalized output path and start time.
        """
        params = dict(params if params is not None else self.get_parameter_values())
        params["noise"] = apply_noise
        params.update(noise_stage=noise_stage, convolved=convolved, normalization={})
        self.result_params = params
        self.last_adi_sum = self.last_psf_sum = None
        self.last_no_planet_output_dir = None
        self.last_contrast_curve_path = None
        self.last_adi_source_dir = Path(path)
        self.last_reconstructed_adi_sum = None
        self.last_planet_contrast = None
        return params, Path(output_dir) if output_dir is not None else None, time.time()

    def resolve_output_directory(self, output_dir, save_root, params):
        """Return or create the output directory requested for this run.

        Input:
        output_dir, save_root: Path or None
            Existing run directory or its parent directory.
        params: dict
            Parameters collected when processing started.

        Output:
        output_dir: Path or None
            Directory used for saved results.
        """
        if output_dir is not None or not params["save_adi"]:
            return output_dir
        if save_root is None:
            self.choose_save_root_folder()
            save_root = self.save_root
        return self.build_simulation_output_dir(save_root) if save_root is not None else None

    def get_preprocessed_adi_files(self, path, require_noise_inputs=False):
        """Find and validate the preprocessed files required by an ADI run.

        Input:
        path: Path
            Directory containing the preprocessed products.
        require_noise_inputs: bool
            Require the cube mean and perfect PSF used to add noise.

        Output:
        files: dict
            Paths to the ADI, PSF and optional noise products.
        """
        files = find_simulation_fits(path)
        if files["adi_sum"] is None or files["psf_sum"] is None:
            raise FileNotFoundError("Could not find the preprocessed *_ADI_sum.fits / *_PSF_sum.fits pair.")
        if require_noise_inputs and files["mean"] is None:
            raise FileNotFoundError("Could not find the matching *_image_cube_mean.fits produced during preprocessing.")
        return files

    def finish_planet_processing(self, planet_adi_sum, output_dir, message="Planet ADI completed."):
        """Normalize, optionally save and record a processed planet image.

        Input:
        planet_adi_sum: ndarray
            Processed planet ADI image.
        output_dir: Path or None
            Run directory used to save the image.
        message: str
            Completion message displayed in the interface.

        Output:
        planet_adi_sum: ndarray
            Planet image normalized with the stellar PSF.
        """
        planet_adi_sum = self.normalize_adi_by_psf(
            planet_adi_sum, self.last_psf_sum, "planet", "star",
        )
        if output_dir is not None:
            self.save_planet_adi_result(planet_adi_sum, output_dir)
        self.last_planet_contrast = float(get_planet_contrast(planet_adi_sum))
        self.files_box.append(message)
        return planet_adi_sum

    def process_and_reconstruct_planet(
            self, stellar_adi_sum, output_dir, params, noise_applied,
            planet_noise_params, planet_processor):
        """Process the selected planet and combine it with the stellar image.

        Input:
        stellar_adi_sum: ndarray
            Final stellar ADI image.
        output_dir: Path or None
            Run directory used to save results.
        params: dict
            Parameters collected when processing started.
        noise_applied: bool
            Whether the stellar run successfully added noise.
        planet_noise_params: tuple or None
            Exposure time and zenith distance used for planet noise.
        planet_processor: callable
            Planet-processing method matching the active processing chain.

        Output:
        None
        """
        if params["planet"] != "With_Planet":
            return
        try:
            kwargs = {"output_dir": output_dir, "params": params}
            if noise_applied and planet_noise_params is not None:
                kwargs.update(
                    apply_photon_noise=True,
                    delta_t=planet_noise_params[0],
                    zenith_distance=planet_noise_params[1],
                )
            planet_adi_sum = planet_processor(**kwargs)
            if planet_adi_sum is not None:
                self.reconstruct_adi_image(stellar_adi_sum, planet_adi_sum, output_dir=output_dir)
        except Exception as error:
            self.files_box.append(f"Warning: Could not process planet ADI: {error}")

    def process_planet_adi_with_noisy_cubes(
            self,
            output_dir=None,
            apply_photon_noise=False,
            delta_t=None,
            zenith_distance=None, params=None):
        """Process the planet cube with optional noise before ADI.

        Input:
        output_dir: Path, optional
            Run directory used to save the planet result.
        apply_photon_noise: bool
            Add photon noise to every frame before ADI.
        delta_t: float, optional
            Frame exposure time in seconds.
        zenith_distance: float, optional
            Zenith distance in degrees.
        params: dict, optional
            Parameters collected when processing started.

        Output:
        planet_adi_sum: ndarray or None
            Final normalized planet image, or None when inputs are unavailable.
        """


        params = params if params is not None else self.get_parameter_values()
        planet_path = self.get_with_planet_path(params)
        if planet_path is None:
            self.files_box.append("Warning: With_Planet path is not available. Skipping planet ADI.")
            return None

        input_files = find_simulation_fits(planet_path)
        cube_file = input_files["image_cube"]
        psf_cube_file = input_files["psf_cube"]
        if cube_file is None or psf_cube_file is None:
            self.files_box.append(
                "Warning: Could not find image_cube or psf_cube in "
                f"{planet_path}. Skipping planet ADI."
            )
            return None

        self.files_box.append(
            f"Starting legacy planet ADI processing on: "
            f"{cube_file.name} and {psf_cube_file.name}..."
        )
        QApplication.processEvents()

        cube_data = np.asarray(fits.getdata(str(cube_file)), dtype=np.float32)
        psf_cube_data = np.asarray(fits.getdata(str(psf_cube_file)), dtype=np.float32)

        if apply_photon_noise:
            if delta_t is None or zenith_distance is None:
                raise ValueError("delta_t and zenith_distance are required for planet photon noise.")

            self.files_box.append("Creating planet cubes with photon noise and no read noise...")
            perf_psf_data = fits.getheader(cube_file)["S_PERPSF"]
            frame_exp_time = np.float32(delta_t)
            planet_photon_flux, planet_emission_per_pix, _ = self.get_planet_micado_flux(
                path=planet_path,
                delta_t=delta_t,
                zenith_distance=zenith_distance,
                params=params,
            )

            cube_data, _ = scale_to_photons(
                cube_data,
                perf_psf_data,
                planet_photon_flux,
                planet_emission_per_pix,
                frame_exp_time,
                sig_ron=0.,
                no_noise=False,
                silent=True,
            )
            psf_cube_data, _ = scale_to_photons(
                psf_cube_data,
                perf_psf_data,
                planet_photon_flux,
                planet_emission_per_pix,
                frame_exp_time,
                sig_ron=0.,
                no_noise=False,
                silent=True,
            )
            cube_data = np.asarray(cube_data, dtype=np.float32)
            psf_cube_data = np.asarray(psf_cube_data, dtype=np.float32)

        planet_adi_sum, _ = micado_adi(
            cube_data,
            psf_cube_data,
            str(planet_path) + os.sep,
        )
        return self.finish_planet_processing(
            planet_adi_sum, output_dir, message="Legacy planet ADI completed.",
        )

    def process_precomputed_planet_adi_then_add_noise(
            self,
            output_dir=None,
            apply_photon_noise=False,
            delta_t=None,
            zenith_distance=None, params=None):
        """Load a preprocessed planet ADI, then optionally add image-level noise.

        Input:
        output_dir: Path, optional
            Run directory used to save the planet result.
        apply_photon_noise: bool
            Add photon and emission noise after ADI.
        delta_t: float, optional
            Frame exposure time in seconds.
        zenith_distance: float, optional
            Zenith distance in degrees.
        params: dict, optional
            Parameters collected when processing started.

        Output:
        planet_adi_sum: ndarray or None
            Final convolved and normalized planet image.
        """

        params = params if params is not None else self.get_parameter_values()
        planet_path = self.get_with_planet_path(params)
        if planet_path is None:
            self.files_box.append("Warning: With_Planet path is not available. Skipping planet ADI.")
            return None

        try:
            input_files = self.get_preprocessed_adi_files(
                planet_path, require_noise_inputs=apply_photon_noise,
            )
        except FileNotFoundError as error:
            self.files_box.append(f"Warning: {error} Folder: {planet_path}")
            return None
        adi_file = input_files["adi_sum"]
        mean_file = input_files["mean"]

        self.files_box.append(f"Loading preprocessed planet ADI: {adi_file.name}")
        QApplication.processEvents()

        if apply_photon_noise:
            if delta_t is None or zenith_distance is None:
                raise ValueError("delta_t and zenith_distance are required for planet photon noise.")

            self.files_box.append(
                "Adding photon and emission noise to the preprocessed planet ADI..."
            )
            perf_psf_data = fits.getheader(adi_file)["S_PERPSF"]
            frame_exp_time = np.float32(delta_t)
            planet_photon_flux, planet_emission_per_pix, _ = self.get_planet_micado_flux(
                path=planet_path,
                delta_t=delta_t,
                zenith_distance=zenith_distance,
                params=params,
            )
            # sigron=0 because the noise has already been applied to the stellar ADI, and the planet is added to that image. The read noise is already present in the stellar ADI.
            planet_adi_sum = add_noise_after_adi(
                fits.getdata(adi_file),
                perf_psf_data,
                mean_file,
                photon_flux=planet_photon_flux,
                sig_ron=0.,
                emission_flux=planet_emission_per_pix,
                frame_exp_time=frame_exp_time,
                silent=True,
            )  #################Maybe a problem of exp_time here. Muste be hte same than for the star... to be checked.
        else:
            planet_adi_sum = fits.getdata(adi_file)

        wavelength, pixel_scale = params["wavelength"], params["sampling"]
        if not apply_photon_noise:
            delta_mag_text = params["delta_magnitude"]
            if delta_mag_text is None:
                raise ValueError("No delta magnitude selected.")
            delta_mag = float(delta_mag_text)
            if not np.isfinite(delta_mag):
                raise ValueError("Delta magnitude must be finite.")
            planet_flux_ratio = 10.0 ** (-0.4 * delta_mag)
            planet_adi_sum = np.asarray(planet_adi_sum, dtype=np.float64) * planet_flux_ratio
            self.files_box.append(
                f"Noiseless planet scaled by delta magnitude {delta_mag:g}: "
                f"flux ratio = {planet_flux_ratio:.6e}."
            )
        planet_adi_sum, _ = self.convolve_adi_and_psf(
            planet_adi_sum, None, wavelength, pixel_scale
        )
        return self.finish_planet_processing(planet_adi_sum, output_dir)

    def perform_adi_with_noisy_cubes(
            self, path, save_root=None, apply_noise=False, output_dir=None, params=None):
        """Run the stellar NOISE-then-ADI processing chain.

        Input:
        path: str or Path
            Stellar simulation input directory.
        save_root: Path, optional
            Parent directory for a new timestamped run.
        apply_noise: bool
            Add detector and photon noise to each cube frame.
        output_dir: Path, optional
            Existing run directory used instead of ``save_root``.
        params: dict, optional
            Parameters collected when processing started.

        Output:
        None
        """

        params, output_dir, total_time_start = self.initialize_processing_run(
            params, path, apply_noise, "BEFORE", False, output_dir,
        )
        input_files = find_simulation_fits(path)
        cube_file = input_files["image_cube"]
        psf_cube_file = input_files["psf_cube"]

        if cube_file is None or psf_cube_file is None:
            QMessageBox.warning(
                self,
                "Incomplete ADI files",
                "Could not find image_cube or psf_cube files in the selected folder.",
            )
            return

        self.files_box.append(
            f"Starting legacy ADI processing on: "
            f"{cube_file.name} and {psf_cube_file.name}..."
        )
        QApplication.processEvents()
        start_time = time.time()

        cube_data = np.asarray(fits.getdata(str(cube_file)), dtype=np.float32)
        psf_cube_data = np.asarray(fits.getdata(str(psf_cube_file)), dtype=np.float32)
        perf_psf_data = fits.getheader(cube_file)["S_PERPSF"]

        noise_applied = False
        planet_noise_params = None

        output_dir = self.resolve_output_directory(output_dir, save_root, params)

        original_cube_data = cube_data if output_dir is not None and apply_noise else None
        original_psf_cube_data = psf_cube_data if output_dir is not None and apply_noise else None

        if apply_noise:
            key_values = self.read_config_values_from_path(
                path,
                "simuconfig",
                ["delta_t", "zenith_distance"],
            )
            if key_values is None:
                self.files_box.append("Warning: Could not read config file. Skipping noise application.")
                key_values = {}

            delta_t = key_values.get("delta_t")
            zenith_distance = key_values.get("zenith_distance")

            if delta_t is None or zenith_distance is None:
                self.files_box.append(
                    "Warning: Could not find delta_t or zenith_distance in "
                    "config file. Skipping noise application."
                )
            else:
                try:
                    zenith_distance = np.float32(zenith_distance)
                    start_time_flux = time.time()
                    photon_flux, emission_per_pix, _ = self.get_micado_flux(
                        delta_t=delta_t,
                        zenith_distance=zenith_distance,
                        params=params,
                    )
                    ellapsed_flux = time.time() - start_time_flux
                    print(
                        "Photon flux and emission per pixel computed in "
                        f"{ellapsed_flux:.2f} s."
                    )
                    frame_exp_time = np.float32(delta_t)

                    self.files_box.append("Creating image and PSF cubes with photon noise...")
                    start_time_image_noise = time.time()
                    image_cube_noise, _ = scale_to_photons(
                        cube_data,
                        perf_psf_data,
                        photon_flux,
                        emission_per_pix,
                        frame_exp_time,
                        sig_ron=15.,
                        no_noise=False,
                        silent=True,
                    )
                    ellapsed_image_noise = time.time() - start_time_image_noise
                    print(f"Image cube with photon noise created in {ellapsed_image_noise:.2f} s.")

                    start_time_psf_noise = time.time()
                    psf_cube_noise, _ = scale_to_photons(
                        psf_cube_data,
                        perf_psf_data,
                        photon_flux,
                        emission_per_pix,
                        frame_exp_time,
                        sig_ron=15.,
                        no_noise=False,
                        silent=True,
                    )
                    ellapsed_cube_noise = time.time() - start_time_psf_noise
                    print(f"PSF cube with photon noise created in {ellapsed_cube_noise:.2f} s.")
                    cube_data = np.asarray(image_cube_noise, dtype=np.float32)
                    psf_cube_data = np.asarray(psf_cube_noise, dtype=np.float32)
                    noise_applied = True
                    planet_noise_params = (delta_t, zenith_distance)
                    self.files_box.append(f"Applied photon noise scaling with delta_t={delta_t}s")
                except Exception as e:
                    self.files_box.append(f"Warning: Could not apply noise scaling: {str(e)}")

        params["noise"] = noise_applied
        start_time_adi = time.time()
        adi_sum, psf_sum = micado_adi(cube_data, psf_cube_data, str(path) + os.sep)
        adi_sum = self.normalize_adi_by_psf(adi_sum, psf_sum, "star")
        ellapsed_adi = time.time() - start_time_adi
        print(f"Legacy ADI processing completed in {ellapsed_adi:.2f} s.")
        elapsed = time.time() - start_time
        self.files_box.append(f"Legacy ADI processing completed in {elapsed:.1f} s.")

        if output_dir is not None:
            if noise_applied:
                self.save_noise_inputs(
                    output_dir,
                    original_cube_data,
                    cube_data,
                    original_psf_cube_data,
                    psf_cube_data,
                )
            self.save_adi_results(adi_sum, psf_sum, output_dir)

        self.last_adi_sum = adi_sum
        self.last_psf_sum = psf_sum
        self.show_stellar_adi_result(adi_sum)

        self.process_and_reconstruct_planet(
            adi_sum, output_dir, params, noise_applied, planet_noise_params,
            self.process_planet_adi_with_noisy_cubes,
        )

        total_time_elapsed = time.time() - total_time_start
        self.files_box.append(
            f"Total legacy ADI processing function time: {total_time_elapsed:.1f} s."
        )

    def process_precomputed_stellar_adi_then_add_noise(
            self, path, save_root=None, apply_noise=False, output_dir=None, params=None):
        """Load preprocessed stellar ADI products, then optionally add noise.

        Input:
        path: str or Path
            Stellar simulation input directory.
        save_root: Path, optional
            Parent directory for a new timestamped run.
        apply_noise: bool
            Add detector, photon and emission noise after ADI.
        output_dir: Path, optional
            Existing run directory used instead of ``save_root``.
        params: dict, optional
            Parameters collected when processing started.

        Output:
        None
        """

        params, output_dir, total_time_start = self.initialize_processing_run(
            params, path, apply_noise, "AFTER", True, output_dir,
        )
        try:
            input_files = self.get_preprocessed_adi_files(
                path, require_noise_inputs=apply_noise,
            )
        except FileNotFoundError as error:
            QMessageBox.warning(
                self,
                "Incomplete processing files",
                f"Folder: {path}\n{error}",
            )
            return
        adi_file, psf_sum_file = input_files["adi_sum"], input_files["psf_sum"]
        mean_file = input_files["mean"]

        self.files_box.append(f"Loading preprocessed ADI: {adi_file.name}")
        QApplication.processEvents()
        start_time = time.time()

        perf_psf_data = fits.getheader(adi_file)["S_PERPSF"]

        noise_applied = False
        noise_parameters = None
        planet_noise_params = None

        output_dir = self.resolve_output_directory(output_dir, save_root, params)

        # This chain applies noise to the final 2D ADI image
        if apply_noise:
            key_values = self.read_config_values_from_path(path, "simuconfig", ["delta_t", "zenith_distance"])
            if key_values is None:
                self.files_box.append("Warning: Could not read config file. Skipping noise application.")
                key_values = {}

            delta_t = key_values.get("delta_t")
            zenith_distance = key_values.get("zenith_distance")

            if delta_t is None or zenith_distance is None:
                self.files_box.append("Warning: Could not find delta_t or zenith_distance in config file. Skipping noise application.")
            else:
                try:
                    zenith_distance = np.float32(zenith_distance)
                    start_time_flux = time.time()
                    photon_flux, emission_per_pix, _ = self.get_micado_flux(delta_t=delta_t, zenith_distance=zenith_distance, params=params)
                    ellapsed_flux = time.time() - start_time_flux
                    print(f"Photon flux and emission per pixel computed in {ellapsed_flux:.2f} s.")
                    frame_exp_time = np.float32(delta_t)
                    noise_parameters = {
                        "photon_flux": photon_flux,
                        "sig_ron": 15.,
                        "emission_flux": emission_per_pix,
                        "frame_exp_time": frame_exp_time,
                    }
                    noise_applied = True
                    planet_noise_params = (delta_t, zenith_distance)
                    self.files_box.append(
                        "Noise parameters ready: 2D noise will be added to the "
                        f"preprocessed ADI (delta_t={delta_t}s)."
                    )
                except Exception as e:
                    self.files_box.append(f"Warning: Could not prepare ADI noise: {str(e)}")
                    # A failed noise setup falls back to the clean ADI result.

        params["noise"] = noise_applied
        start_time_adi = time.time()
        if noise_applied:
            adi_sum_clean = fits.getdata(adi_file)
            self.files_box.append(f"Using preprocessed cube mean: {mean_file.name}")
            try:
                adi_sum = add_noise_after_adi(
                    adi_sum_clean,
                    perf_psf_data,
                    mean_file,
                    silent=True,
                    **noise_parameters,
                )
            except (OSError, ValueError, KeyError) as error:
                QMessageBox.warning(self, "Invalid noise inputs", str(error))
                return
            psf_sum = np.asarray(fits.getdata(psf_sum_file), dtype=np.float64)
            psf_sum *= noise_parameters["photon_flux"] / perf_psf_data
        else:
            adi_sum = fits.getdata(adi_file)
            psf_sum = fits.getdata(psf_sum_file)

        wavelength, pixel_scale = params["wavelength"], params["sampling"]
        adi_sum, psf_sum = self.convolve_adi_and_psf(
            adi_sum, psf_sum, wavelength, pixel_scale
        )
        adi_sum = self.normalize_adi_by_psf(adi_sum, psf_sum, "star")
        ellapsed_adi = time.time() - start_time_adi
        print(f"ADI processing completed in {ellapsed_adi:.2f} s.")
        elapsed = time.time() - start_time
        self.files_box.append(f"ADI processing completed in {elapsed:.1f} s.")

        if output_dir is not None:
            if noise_applied:
                self.files_box.append(
                    "The preprocessed ADI file was kept unchanged; noise was applied "
                    "to the in-memory 2D result."
                )
            self.save_adi_results(adi_sum, psf_sum, output_dir)

        self.last_adi_sum = adi_sum
        self.last_psf_sum = psf_sum
        self.show_stellar_adi_result(adi_sum)

        self.process_and_reconstruct_planet(
            adi_sum, output_dir, params, noise_applied, planet_noise_params,
            self.process_precomputed_planet_adi_then_add_noise,
        )
        

        total_time_elapsed = time.time() - total_time_start
        self.files_box.append(f"Total ADI processing function time: {total_time_elapsed:.1f} s.")

    def normalize_adi_by_psf(self, adi_sum, psf_sum, label, psf_label=None):
        """Normalize an ADI image by a PSF maximum and report the operation.

        Input:
        adi_sum: ndarray
            ADI image to normalize.
        psf_sum: ndarray or None
            Reference PSF image.
        label: str
            Product label stored in runtime metadata.
        psf_label: str, optional
            Reference label used in messages.

        Output:
        normalized: ndarray
            Normalized image, or the input image when no valid PSF exists.
        """
        psf_label = psf_label or label
        if self.result_params is not None:
            self.result_params.setdefault("normalization", {})[label] = None
        if psf_sum is None:
            self.files_box.append(f"Warning: {psf_label} PSF is missing. {label} ADI image was not normalized.")
            return adi_sum

        max_psf = np.nanmax(psf_sum)
        if max_psf == 0:
            self.files_box.append(f"Warning: {psf_label} PSF max is zero. {label} ADI image was not normalized.")
            return adi_sum

        self.files_box.append(f"Normalized {label} ADI by {psf_label} PSF max: {max_psf:.6e}")
        if self.result_params is not None:
            self.result_params["normalization"][label] = max_psf
        return normalize_by_psf_max(adi_sum, psf_sum)

    def reconstruct_adi_image(self, star_adi_path, planet_adi_path, output_dir=None):
        """Combine final star and planet images and update the display.

        Input:
        star_adi_path, planet_adi_path: ndarray or Path
            Final images or their FITS paths.
        output_dir: Path, optional
            Run directory used to save the combined image.

        Output:
        reconstructed: ndarray
            Sum of the stellar and planet images.
        """
        save_path = None
        if output_dir is not None:
            output_dir = Path(output_dir) / "Reconstructed"
            output_dir.mkdir(parents=True, exist_ok=True)
            params_slug = self.build_params_slug(planet_folder="With_Planet")
            save_path = output_dir / f"{params_slug}_star_planet_adi.fits"

        reconstructed = reconstruct_coronographic_image(
            star_adi_path=star_adi_path,
            planet_adi_path=planet_adi_path,
            save_path=None,
            display=False,
        )
        self.last_reconstructed_adi_sum = reconstructed

        self.update_adi_image_options()
        self.adi_image_combo.setCurrentText("ADI star + planet")
        if save_path is not None:
            self.write_fits(save_path, reconstructed, product="COMBINED")
            self.files_box.append(f"Reconstructed ADI result saved:\n- {save_path}")
        self.files_box.append("Reconstructed ADI image completed with perfCalculator.utils.")
        return reconstructed

    def convolve_adi_and_psf(self, adi_sum, psf_sum, wavelength, pixel_scale, tel_diameter=38.542):
        """Convolve ADI and PSF images with a lambda/(2D) aperture.

        Input:
        adi_sum: ndarray
            ADI image to convolve.
        psf_sum: ndarray or None
            PSF image to convolve when needed.
        wavelength: float
            Wavelength in micrometres.
        pixel_scale: float
            Angular sampling in mas/pixel.
        tel_diameter: float
            Telescope diameter in metres.

        Output:
        images: tuple
            Cropped convolved ADI and PSF images
        """
        adi_sum = np.asarray(adi_sum)
        psf_sum = np.asarray(psf_sum) if psf_sum is not None else None
        if adi_sum.ndim != 2 or (psf_sum is not None and psf_sum.shape != adi_sum.shape):
            raise ValueError("ADI and PSF must be 2D images with the same shape.")

        l_over_d_pix = wavelength * 1e-6 / tel_diameter * 180. / np.pi * 3600 * 1e3 / pixel_scale
        mask_radius = 0.5 * l_over_d_pix
        mask = get_circle_mask(adi_sum.shape, mask_radius)
        self.files_box.append(f"Convolving ADI and PSF: aperture radius = {mask_radius:.3f} pixels.")
        return convolve_adi_image(adi_sum, mask), (
            convolve_adi_image(psf_sum, mask) if psf_sum is not None else None
        )

    def run_processing(self):
        """Validate current inputs and process the preprocessed ADI products.

        Output:
        None
        """
        try:
            params = self.get_parameter_values(validate=True)
        except ValueError as error:
            QMessageBox.warning(self, "Invalid parameters", str(error))
            return
        print("========== PARAMETERS ==========", params)
        if self.adi_checkbox.currentText() != "Yes":
            return
        path = self.build_input_path("No_Planet", params)
        self.path_box.setText(str(path) + os.sep)

        # Process the ADI products created during preprocessing.
        self.process_precomputed_stellar_adi_then_add_noise(
            path, save_root=self.save_root if params["save_adi"] else None,
            apply_noise=params["noise"], params=params,
        )
