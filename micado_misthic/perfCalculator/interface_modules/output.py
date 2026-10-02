"""
Author: Tristan Deseine
"""

from datetime import datetime
from pathlib import Path

import numpy as np
from configobj import ConfigObj

from micado_misthic.perfCalculator.utils import (
    save_contrast_curve as save_contrast_curve_fits,
)


class OutputMethods:
    """FITS and contrast-product output helpers for the interface."""

    def build_params_slug(self, planet_folder=None):
        """Build a filesystem-safe label from the calculation parameters.

        Input:
        planet_folder: str, optional
            Override the product category used in the label.

        Output:
        slug: str
            Parameter label used in exported filenames.
        """
        params = dict(self.result_params or self.get_parameter_values())
        params["planet"] = planet_folder or params["planet"]
        slug_parts = [
            params["planet"],
            params["clc"],
            params["ncpa"],
            params["seeing"],
            f"{params['filter']}{params['wavelength']:.3f}um",
            f"samp{params['sampling']}",
        ]
        if params["observation_time"]:
            slug_parts.append(params["observation_time"])
        if params["magnitude"] is not None:
            slug_parts.append(f"mag{params['magnitude']:g}")
        if params["planet"] == "With_Planet":
            slug_parts.append(f"pdist{params['planet_distance']:g}mas")
            if params["delta_magnitude"] is not None:
                slug_parts.append(f"dmag{params['delta_magnitude']:g}")
        raw_slug = "_".join(slug_parts)
        return "".join(
            char if char.isascii() and (char.isalnum() or char in "._-") else "_"
            for char in raw_slug
        )

    def write_fits(self, path, data, product=None):
        """Write an image FITS with processing metadata.

        Input:
        path: Path
            Output FITS path.
        data: ndarray
            Image or cube to save.
        product: str, optional
            Product type used to populate final-image metadata.

        Output:
        None
        """
        from astropy.io import fits

        image = np.asarray(data, dtype=np.float32)
        header = fits.Header()
        if product is not None and self.result_params is not None:
            params = self.result_params
            is_psf = product == "PSF"
            noise = params["noise"] and (not is_psf or params["noise_stage"] == "BEFORE")
            labels = {"STAR_ADI": ("star",), "PLANET_ADI": ("planet",),
                      "COMBINED": ("star", "planet"), "PSF": ()}[product]
            normalization = params.get("normalization", {})
            normalized = bool(labels) and all(normalization.get(label) is not None for label in labels)

            header["NOISEADD"] = (bool(noise), "Random noise added by this processing run")
            header["NOISEPOS"] = (params["noise_stage"] if noise else "NONE", "Noise before/after ADI, or NONE")
            header["CONVOLVD"] = (params["convolved"], "Circular aperture convolution applied")
            header["NORMALIZ"] = (normalized, "Divided by the stellar PSF maximum")
            header["PIXSCALE"] = (params["sampling"], "Angular pixel scale [mas/pixel]")
            header["WAVEUM"] = (params["wavelength"], "Wavelength [micrometre]")

            if params["convolved"]:
                header["CONVKERN"] = "CIRCLE_LAMBDA/2D"
                header["TELDIAM"] = (38.542, "Diameter used for convolution [m]")
        fits.PrimaryHDU(image, header=header).writeto(path, overwrite=True)

    def build_simulation_output_dir(self, save_root):
        """Create a timestamped directory for one processing run.

        Input:
        save_root: Path
            Parent directory selected by the user.

        Output:
        output_dir: Path
            Newly created run directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(save_root) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        self.last_no_planet_output_dir = None
        return output_dir

    def save_noise_inputs(self, output_dir, original_cube, processed_cube, original_psf, processed_psf):
        """Save original and noisy cubes from the NOISE-then-ADI chain.

        Input:
        output_dir: Path
            Run output directory.
        original_cube, processed_cube: ndarray
            Stellar cubes before and after noise.
        original_psf, processed_psf: ndarray
            PSF cubes before and after noise.

        Output:
        None
        """
        noise_dir = Path(output_dir) / "No_Planet" / "noisy_images"
        noise_dir.mkdir(parents=True, exist_ok=True)
        params_slug = self.build_params_slug(planet_folder="No_Planet")

        paths = [
            noise_dir / f"{params_slug}_image_cube_original.fits",
            noise_dir / f"{params_slug}_image_cube_noise.fits",
            noise_dir / f"{params_slug}_psf_cube_original.fits",
            noise_dir / f"{params_slug}_psf_cube_noise.fits",
        ]

        self.write_fits(paths[0], original_cube)
        self.write_fits(paths[1], processed_cube)
        self.write_fits(paths[2], original_psf)
        self.write_fits(paths[3], processed_psf)

        self.files_box.append(
            "Noise inputs saved:\n"
            f"- {paths[0]}\n"
            f"- {paths[1]}\n"
            f"- {paths[2]}\n"
            f"- {paths[3]}"
        )

    def read_config_values_from_path(self, path, section_name, keys):
        """Read selected values from the simulation configuration file.

        Input:
        path: str or Path
            Directory containing the simulation ``.ini`` file.
        section_name: str
            Configuration section to inspect.
        keys: str or list[str]
            Key or keys to retrieve.

        Output:
        values: str, dict or None
            Requested values, or None when the file or section is unavailable.
        """

        path = Path(path)

        config_files = list(path.glob("*.ini"))

        if not config_files:
            self.files_box.append(
                "Warning: Config file (*.ini) not found in path."
            )
            return None

        if len(config_files) > 1:
            self.files_box.append(
                f"Warning: Multiple config files (*.ini) found in {path}."
            )
            return None

        config_file = config_files[0]

        try:
            config = ConfigObj(str(config_file))

            if section_name not in config:
                self.files_box.append(
                    f"Warning: section [{section_name}] not found in {config_file.name}."
                )
                return None

            section = config[section_name]

            # Preserve the scalar return type when one key is requested.
            if isinstance(keys, str):
                value = section.get(keys)

                if value is None:
                    self.files_box.append(
                        f"Warning: {keys} not found in [{section_name}]."
                    )
                    return None

                self.files_box.append(
                    f"Config loaded from {config_file.name}: "
                    f"[{section_name}] {keys}={value}"
                )

                return value

            values = {}

            for key in keys:
                value = section.get(key)

                if value is None:
                    self.files_box.append(
                        f"Warning: {key} not found in [{section_name}]."
                    )
                else:
                    values[key] = value

            self.files_box.append(
                f"Config loaded from {config_file.name}: "
                f"[{section_name}] {values}"
            )

            return values

        except Exception as e:
            self.files_box.append(
                f"Error reading config file {config_file.name}: {str(e)}"
            )
            return None

    def save_adi_results(self, adi_sum, psf_sum, output_dir):
        """Save the final stellar ADI and PSF products.

        Input:
        adi_sum, psf_sum: ndarray
            Final stellar ADI and PSF images.
        output_dir: Path
            Run output directory.

        Output:
        paths: tuple[Path, Path]
            Saved ADI and PSF paths.
        """
        output_dir = Path(output_dir) / "No_Planet"
        output_dir.mkdir(parents=True, exist_ok=True)
        params_slug = self.build_params_slug(planet_folder="No_Planet")
        self.last_no_planet_output_dir = output_dir

        adi_sum_path = output_dir / f"{params_slug}_adi.fits"
        psf_sum_path = output_dir / f"{params_slug}_psf.fits"

        self.write_fits(adi_sum_path, adi_sum, product="STAR_ADI")
        self.write_fits(psf_sum_path, psf_sum, product="PSF")

        self.files_box.append(
            f"No_Planet ADI results saved:\n- {adi_sum_path}\n- {psf_sum_path}"
        )
        return adi_sum_path, psf_sum_path

    def save_planet_adi_result(self, adi_sum, output_dir):
        """Save the final planet ADI image.

        Input:
        adi_sum: ndarray
            Final planet image.
        output_dir: Path
            Run output directory.

        Output:
        adi_path: Path
            Saved planet FITS path.
        """
        output_dir = Path(output_dir) / "With_Planet"
        output_dir.mkdir(parents=True, exist_ok=True)
        params_slug = self.build_params_slug(planet_folder="With_Planet")

        adi_sum_path = output_dir / f"{params_slug}_planet_adi.fits"
        self.write_fits(adi_sum_path, adi_sum, product="PLANET_ADI")

        self.files_box.append(f"With_Planet ADI result saved:\n- {adi_sum_path}")
        return adi_sum_path

    def save_contrast_curve(
            self,
            x,
            contrast,
            output_dir,
            planet_distance_mas=None,
            planet_contrast=None):
        """Save the contrast curve as FITS data and a PNG plot.

        Input:
        x, contrast: ndarray
            Angular separations and 5-sigma contrast values.
        output_dir: Path
            Contrast-curve output directory.
        planet_distance_mas, planet_contrast: float, optional
            Planet marker metadata.

        Output:
        fits_path: Path
            Saved contrast-table path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        params_slug = self.build_params_slug(planet_folder="No_Planet")

        fits_path = output_dir / f"{params_slug}_contrast_curve.fits"

        save_contrast_curve_fits(
            fits_path, x, contrast,
            planet_distance_mas=planet_distance_mas, planet_contrast=planet_contrast,
        )
        png_path = fits_path.with_suffix(".png")
        self.contrast_figure.savefig(png_path, dpi=200, bbox_inches="tight")

        self.files_box.append(
            f"Contrast curve saved:\n- {fits_path}\n- {png_path}"
        )
        return fits_path
