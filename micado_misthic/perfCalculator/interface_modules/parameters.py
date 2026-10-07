"""
Author: Tristan Deseine
"""

import os

import numpy as np

from micado_misthic.perfCalculator.utils import get_planet_flux_with_deltaMag
from micado_misthic.utils.obsparams import (
    get_aperture_surface,
    get_micado_flux,
    get_star_spectrum,
)


class ParameterPathMethods:
    """Parameter, flux and simulation-path helpers for the interface."""

    def get_planet_folder(self):
        
        """
        Return the directory label for the current planet selection.

        Output:
        planet_folder: str
            ``With_Planet`` or ``No_Planet``.
        """
        return "With_Planet" if self.planet_on_checkbox.isChecked() else "No_Planet"

    def get_with_planet_path(self, params=None):
        """Build the planet simulation path from one set of parameters.

        Input:
        params: dict, optional
            Calculation parameters; UI values are read when omitted.

        Output:
        path: Path or None
            Planet input directory, or None when planet processing is disabled.
        """
        params = params if params is not None else self.get_parameter_values()
        return self.build_input_path("With_Planet", params) if params["planet"] == "With_Planet" else None

    def auto_detect_flux_folder(self):

        """
        Detect ``Flux_input`` inside the selected simulation root.

        Output:
        None
        """
        flux_path = self.base_folder / "Flux_input"
        
        if flux_path.exists() and flux_path.is_dir():
            self.flux_folder = flux_path
            self.flux_folder_label.setText(str(flux_path))
        else:
            self.flux_folder = None
            self.flux_folder_label.setText("not found")

    def compute_nearest_zenithal_distance(self, declination):

        """
        Compute the nearest zenithal distance based on declination and the latitude.

        Input:
        declination: float
            Target declination in degrees.

        Output:
        nearest_z: float
            Nearest supported zenith distance in degrees.
        """

        z_reference = np.array([20., 40.])

        latitude = -24.5 


        z_meridien = np.abs(latitude - float(declination))


        nearest_z = float(z_reference[np.argmin(abs(z_reference - z_meridien))])
        self.files_box.append(
            f"Nearest zenithal distance for declination {declination} "
            f"and latitude {latitude} is {nearest_z:g}"
        )
        return nearest_z

    def get_selected_observation_time(self):
        """
        Return the selected observation-time directory label.

        Output:
        observation_time: str or None
            Selected label, or None when unspecified.
        """
        if self.obs_time_combo.currentIndex() == 0:
            return None
        obs_time_text = self.obs_time_combo.currentText()
        return obs_time_text if obs_time_text else None

    def get_star_spectrum(self, star_mag=None, wavelength=None, display_plot=False, params=None):
        """
        Compute the selected stellar spectrum using obsparams.get_star_spectrum().

        Input:
        star_mag: float, optional
            Stellar magnitude; defaults to the parameters collected when processing started.
        wavelength: float, optional
            Reference wavelength in micrometres.
        display_plot: bool
            Ask the spectrum routine to display its diagnostic plot.
        params: dict, optional
            Parameters collected when processing started.

        Output:
        spectrum: tuple
            Wavelength samples and stellar photon flux.
        """
        params = params if params is not None else self.get_parameter_values()
        if params["flux_folder"] is None:
            raise FileNotFoundError("Flux folder not found. Select a base folder containing 'Flux_input'.")
        
        star_mag = params["magnitude"] if star_mag is None else float(star_mag)
        
        if star_mag is None:
            raise ValueError("No star magnitude selected.")
        wavelength = params["wavelength"] if wavelength is None else float(str(wavelength).replace('µm', '').replace('um', '').strip())
        return get_star_spectrum(str(params["flux_folder"]) + os.sep, star_mag, wavelength, display_plot=display_plot)

    def get_micado_flux(self, delta_t, zenith_distance, flux_dir=None, star_flux=None, wavelength=None, params=None):
        """
        Compute MICADO photon flux and background emission using obsparams.get_micado_flux()

        Input:
        delta_t: float
            Frame exposure time in seconds.
        zenith_distance: float
            Zenith distance in degrees.
        flux_dir: str, optional
            Calibration directory.
        star_flux: ndarray, optional
            Precomputed source spectrum.
        wavelength: float, optional
            Filter wavelength in micrometres.
        params: dict, optional
            Parameters collected when processing started.

        Output:
        flux_data: tuple
            Photon flux, emission per pixel and global transmission.
        """
        params = params if params is not None else self.get_parameter_values()
        if flux_dir is None:
            if params["flux_folder"] is None:
                raise FileNotFoundError("Flux folder not found. Select a base folder containing 'Flux_input'.")
            flux_dir = str(params["flux_folder"]) + os.sep
        wavelength = params["wavelength"] if wavelength is None else float(str(wavelength).replace('µm', '').replace('um', '').strip())
        if star_flux is None:
            _, star_flux = self.get_star_spectrum(wavelength=wavelength, params=params)
        aperture_surface = get_aperture_surface(str(params["base_folder"] / "PUPIL" / "Pupil_ELT_v03.fits"))
        return get_micado_flux(
            flux_dir, star_flux, f"{wavelength:.3f}", np.float32(delta_t), aperture_surface,
            airmass=1 / np.cos(np.radians(zenith_distance)), pixel_scale=params["sampling"],
        )

    def get_planet_micado_flux(self, delta_t=None, zenith_distance=None, wavelength=None, path=None, params=None):
        """Compute MICADO flux for the selected planet.

        Input:
        delta_t: float, optional
            Frame exposure time in seconds.
        zenith_distance: float, optional
            Zenith distance in degrees.
        wavelength: float, optional
            Filter wavelength in micrometres.
        path: Path, optional
            Planet simulation directory used to read missing configuration values.
        params: dict, optional
            Parameters collected when processing started.

        Output:
        flux_data: tuple
            Planet photon flux, emission per pixel and global transmission.
        """
        params = params if params is not None else self.get_parameter_values()
        if params["planet"] != "With_Planet":
            raise ValueError("Planet on must be checked before computing planet flux.")

        if path is None:
            path = self.get_with_planet_path(params)
        if path is None:
            raise ValueError("With_Planet path is not available.")

        if delta_t is None or zenith_distance is None:
            key_values = self.read_config_values_from_path(
                path,
                "simuconfig",
                ["delta_t", "zenith_distance"],
            )
            if key_values is None:
                raise ValueError(f"Could not read simuconfig from {path}")
            delta_t = key_values.get("delta_t") if delta_t is None else delta_t
            zenith_distance = key_values.get("zenith_distance") if zenith_distance is None else zenith_distance

        if delta_t is None or zenith_distance is None:
            raise ValueError("delta_t and zenith_distance are required to compute planet flux.")

        delta_mag, star_mag = params["delta_magnitude"], params["magnitude"]
        if delta_mag is None or star_mag is None:
            raise ValueError("Star magnitude and delta magnitude are required for planet flux.")
        _, star_flux = self.get_star_spectrum(star_mag=star_mag, wavelength=wavelength, params=params)
        planet_flux = get_planet_flux_with_deltaMag(star_flux, star_mag, delta_mag)
        return self.get_micado_flux(
            delta_t=delta_t,
            zenith_distance=np.float32(zenith_distance),
            star_flux=planet_flux,
            wavelength=wavelength,
            params=params,
        )

    def update_filter_wavelength_options(self):
        """
        Refresh wavelengths allowed by the selected pixel sampling.
        Some filters are unavailable at 4 mas/pixel  so they are removed from the wavelength combo-box.


        Output:
        None
        """
        sampling = self.sampling_combo.currentText()
        # J-band simulations are unavailable at 4 mas/pixel.
        try:
            sampling_val = float(sampling)
        except Exception:
            sampling_val = None
        allowed_filters = ["H", "K"] if sampling_val == 4.0 else ["J", "H", "K"]

        current = self.filter_wavelength_combo.currentData()
        self.filter_wavelength_combo.clear()

        for filt, wavelengths in self.filter_wavelength_map.items():
            if filt in allowed_filters:
                for wavelength in wavelengths:
                    self.filter_wavelength_combo.addItem(f"{filt} - {wavelength}", (filt, wavelength))

        # Restore the previous wavelength when it remains valid.
        if current and current[0] in allowed_filters:
            for i in range(self.filter_wavelength_combo.count()):
                if self.filter_wavelength_combo.itemData(i) == current:
                    self.filter_wavelength_combo.setCurrentIndex(i)
                    break

        if self.filter_wavelength_combo.count() > 0 and self.filter_wavelength_combo.currentIndex() == -1:
            self.filter_wavelength_combo.setCurrentIndex(0)

        self.update_paths()

    def get_parameter_values(self, validate=False, for_paths=False):
        """
        Read the current UI values into a typed parameter dictionary.

        Input:
        validate: bool
            Require values needed by the selected processing options.
        for_paths: bool
            Ignore parameters that do not affect the input directory structure (e.g magnitude).

        Output:
        params: dict
            Parameters used consistently by paths, calculations and exports.
        """
        filt, wavelength = self.filter_wavelength_combo.currentData() or ("H", "1.582 µm")
        params = {
            "planet": self.get_planet_folder(),
            "planet_distance": float(self.Distance_combo.currentText().replace(" mas", "")),
            "clc": self.clc_combo.currentText(),
            "ncpa": "NCPA" if self.ncpa_combo.currentText() == "Yes" else "NoNCPA",
            "seeing": self.seeing_combo.currentText(),
            "filter": filt,
            "wavelength": float(wavelength.replace("µm", "").strip()),
            "sampling": float(self.sampling_combo.currentText()),
            "observation_time": self.get_selected_observation_time(),
            "noise": True,#self.noise_checkbox.currentText() == "Yes",
            "save_adi": self.save_adi_checkbox.currentText() == "Yes",
            "base_folder": self.base_folder,
            "flux_folder": self.flux_folder,
        }
        for key, widget in (("magnitude", self.magnitude_combo),
                            ("delta_magnitude", self.delta_mag_combo),
                            ("declination", self.declination_combo)):
            
            # Magnitudes do not determine the input directory.
            text = "" if for_paths and key != "declination" else widget.text().strip()
            if text and (not widget.hasAcceptableInput() or not np.isfinite(float(text))):
                raise ValueError(f"Invalid {key}: {text}")
            params[key] = float(text) if text else None

        if validate:
            if params["noise"] and params["magnitude"] is None:
                raise ValueError("A star magnitude is required for noise calculation.")
            if params["planet"] == "With_Planet" and params["delta_magnitude"] is None:
                raise ValueError("A delta magnitude is required for the planet.")
        return params

    def build_input_path(self, planet_folder, params):
        """Build a simulation input directory from typed parameters.

        Input:
        planet_folder: str
            ``With_Planet`` or ``No_Planet``.
        params: dict
            Parameters collected when processing started.

        Output:
        input_path: Path
            Expected simulation directory.
        """
        input_root = params["base_folder"] / planet_folder
        if planet_folder == "No_Planet" and not input_root.is_dir():
            input_root = params["base_folder"]
        input_path = input_root / self.resolve_clc_folder(input_root, params["clc"]) / params["ncpa"] / params["seeing"]
        if params["declination"] is not None:
            zenith = self.compute_nearest_zenithal_distance(params["declination"])
            input_path /= f"z={zenith:g}"
        input_path /= f"filter{params['wavelength']:.3f}"
        if planet_folder == "With_Planet":
            input_path /= f"p_dist_mas={params['planet_distance']:g}/Samp{params['sampling']}"
        else:
            input_path /= f"samp{params['sampling']}"
        return input_path / params["observation_time"] if params["observation_time"] else input_path

    def update_paths(self, *_):
        """Refresh the input path preview after a relevant UI change.

        Input:
        *_: object
            Values emitted by Qt signals and intentionally ignored.

        Output:
        None
        """
        try:
            params = self.get_parameter_values(for_paths=True)
        except ValueError:
            self.no_planet_path = None
            self.path_box.clear()
            return  # A numeric field can be incomplete while the user is typing.
        self.no_planet_path = self.build_input_path("No_Planet", params)
        self.path_box.setText(str(self.no_planet_path) + os.sep)

    def get_clc_folder_candidates(self, clc):
        """Return current and legacy directory names for a coronagraph.

        Input:
        clc: str
            Coronagraph identifier selected in the UI.

        Output:
        candidates: list[str]
            Directory names in lookup order.
        """
        candidates = [clc]
        legacy_clc = self.clc_legacy_folder_map.get(clc)
        if legacy_clc is not None:
            candidates.append(legacy_clc)
        return candidates

    def resolve_clc_folder(self, parent_folder, clc):
        """Select the existing coronagraph directory name.

        Input:
        parent_folder: Path
            Directory containing coronagraph subdirectories.
        clc: str
            Selected coronagraph identifier.

        Output:
        folder_name: str
            Existing current or legacy name, otherwise the current name.
        """
        for candidate in self.get_clc_folder_candidates(clc):
            if (parent_folder / candidate).exists():
                return candidate
        return clc
