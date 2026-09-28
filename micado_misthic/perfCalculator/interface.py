import sys
import ast
import inspect
from pathlib import Path
import numpy as np
import time
from datetime import datetime


from micado_misthic.utils.obsparams import (
    get_aperture_surface,
    get_micado_flux,
    get_star_spectrum,
    scale_to_photons,
)
from micado_misthic.perfCalculator.utils import (
    add_noise_after_adi,
    convolve_adi_image,
    find_simulation_fits,
    save_contrast_curve as save_contrast_curve_fits,
    get_planet_contrast,
    get_planet_flux_with_deltaMag,
    normalize_by_psf_max,
    psf_sum_in_photons,
    reconstruct_coronographic_image,
    run_adi_without_noise,
)
from configobj import ConfigObj
from micado_misthic.perfCalculator import user_func as user_functions


from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QComboBox,
    QCheckBox,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFrame,
    QTextEdit,
    QTabWidget,
)
from PyQt6.QtCore import Qt, QLocale
from PyQt6.QtGui import QDoubleValidator

import matplotlib
matplotlib.use('QtAgg')
from micado_misthic.imgproc import get_circle_mask
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class InputParametersWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Performance Calculator")
        self.setMinimumSize(1000, 640)
        self.resize(1080, 700)

        self.base_folder = Path("C:/...")
        self.flux_folder = None
        self.result_params = None
        self.last_adi_sum = None
        self.last_psf_sum = None
        self.last_reconstructed_adi_sum = None
        self.last_planet_contrast = None
        self.last_no_planet_output_dir = None
        self.last_contrast_curve_path = None
        self.last_adi_source_dir = None
        self.no_planet_path = None

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(16)

        # Left side: controls
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.Shape.StyledPanel)
        left_frame.setMinimumWidth(420)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setSpacing(14)

        # Titre
        title = QLabel("Input parameters")
        title.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            text-decoration: underline;
        """)
        left_layout.addWidget(title)

        ### Base folder
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("Base folder: C:/...")
        self.folder_label.setStyleSheet("color: #444;")
        browse_button = QPushButton("Choose folder")
        browse_button.clicked.connect(self.choose_base_folder)
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(browse_button)
        left_layout.addLayout(folder_layout)
        ###

        
        ### Instrument inputs
        self.clc_legacy_folder_map = {
            "CLC15": "CLC0",
            "CLC25": "CLC1",
            "CLC50": "CLC2",
        }
        self.clc_combo = QComboBox()
        self.clc_combo.addItems(["CLC15", "CLC25", "CLC50"])

        self.ncpa_combo = QComboBox()
        self.ncpa_combo.addItems(["Yes", "No"])

        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems(["1.5", "4.0"])

        # Band + wavelength 
        self.filter_wavelength_map = {
            "J": ["1.190 µm", "1.247 µm", "1.270 µm"],
            "H": ["1.582 µm", "1.635 µm", "1.693 µm"],
            "K": ["2.100 µm", "2.150 µm", "2.235 µm"],
        }
        self.filter_wavelength_combo = QComboBox()

        # Detection noise 
        self.noise_checkbox = QComboBox()
        self.noise_checkbox.addItems(["No", "Yes"])
        ###

        # Observation time
        self.obs_time_combo = QComboBox()
        self.obs_time_combo.addItem("Not specified", None)
        self.obs_time_combo.addItems(["15min", "30min", "1h", "1h30","2h"])


        ### Star inputs
        self.seeing_combo = QComboBox()
        self.seeing_combo.addItems(["Q1", "MED", "Q4"])

        # Star magnitude input (real number between 0 and 15)
        self.magnitude_combo = QLineEdit()
        self.magnitude_combo.setPlaceholderText("0.00 - 15.00")
        magnitude_validator = QDoubleValidator(0.0, 15.0, 3, self.magnitude_combo)
        magnitude_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        magnitude_validator.setLocale(QLocale(QLocale.Language.C))
        self.magnitude_combo.setValidator(magnitude_validator)

        self.declination_combo = QLineEdit()
        self.declination_combo.setPlaceholderText("0°- 90°")
        declination_validator = QDoubleValidator(0.0, 90.0, 3, self.declination_combo)
        declination_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        declination_validator.setLocale(QLocale(QLocale.Language.C))
        self.declination_combo.setValidator(declination_validator)

        # Flux folder label (auto-detected)
        self.flux_folder_label = QLabel("not selected")
        self.flux_folder_label.setStyleSheet("color: #444;")
        ###

        ### Planet-specific inputs
        self.planet_on_checkbox = QCheckBox("Planet on")
        self.planet_on_checkbox.setChecked(False)

        #distance to the star in mas
        self.Distance_combo = QComboBox()
        self.Distance_combo.addItems([
            "15 mas", "30 mas", "50 mas", "75 mas", "100 mas", "125 mas",
            "150 mas", "200 mas", "250 mas", "300 mas", "500 mas", "700 mas",
            "1000 mas", "1400 mas",
        ])

        # Delta magnitude between the star and the planet (0 to 20).
        self.delta_mag_combo = QLineEdit()
        self.delta_mag_combo.setPlaceholderText("0.00 - 20.00")
        delta_mag_validator = QDoubleValidator(0.0, 20.0, 3, self.delta_mag_combo)
        delta_mag_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        delta_mag_validator.setLocale(QLocale(QLocale.Language.C))
        self.delta_mag_combo.setValidator(delta_mag_validator)
        ###


        ### Creating tabs for input parameters
        self.input_tabs = QTabWidget()

        # Instrument tab 
        instrument_tab = QWidget()
        instrument_layout = QGridLayout(instrument_tab)
        instrument_layout.setHorizontalSpacing(12)
        instrument_layout.setVerticalSpacing(8)
        instrument_layout.addWidget(QLabel("Coronograph:"), 0, 0)
        instrument_layout.addWidget(self.clc_combo, 0, 1)
        instrument_layout.addWidget(QLabel("NCPA:"), 1, 0)
        instrument_layout.addWidget(self.ncpa_combo, 1, 1)
        instrument_layout.addWidget(QLabel("Sampling:"), 2, 0)
        instrument_layout.addWidget(self.sampling_combo, 2, 1)
        instrument_layout.addWidget(QLabel("Filter / Wavelength:"), 3, 0)
        instrument_layout.addWidget(self.filter_wavelength_combo, 3, 1)
        instrument_layout.addWidget(QLabel("Detection noise:"), 4, 0)
        instrument_layout.addWidget(self.noise_checkbox, 4, 1)
        instrument_layout.addWidget(QLabel("Observation time:"), 5, 0)
        instrument_layout.addWidget(self.obs_time_combo, 5, 1)

        # Star 
        star_tab = QWidget()
        star_layout = QGridLayout(star_tab)
        star_layout.setHorizontalSpacing(12)
        star_layout.setVerticalSpacing(8)
        star_layout.addWidget(QLabel("Seeing:"), 0, 0)
        star_layout.addWidget(self.seeing_combo, 0, 1)
        star_layout.addWidget(QLabel("Magnitude:"), 1, 0)
        star_layout.addWidget(self.magnitude_combo, 1, 1)
        star_layout.addWidget(QLabel("Declination:"), 2, 0)
        star_layout.addWidget(self.declination_combo, 2, 1)

        star_layout.addWidget(QLabel("Flux folder:"), 3, 0)
        star_layout.addWidget(self.flux_folder_label, 3, 1)

        # Planet tab 
        planet_tab = QWidget()
        planet_layout = QGridLayout(planet_tab)
        planet_layout.setHorizontalSpacing(12)
        planet_layout.setVerticalSpacing(8)
        self.distance_label = QLabel("Distance:")
        self.delta_mag_label = QLabel("DeltaMagnitude:")
        planet_layout.addWidget(self.distance_label, 0, 0)
        planet_layout.addWidget(self.Distance_combo, 0, 1)
        planet_layout.addWidget(self.delta_mag_label, 1, 0)
        planet_layout.addWidget(self.delta_mag_combo, 1, 1)
        

        planet_layout.addWidget(self.planet_on_checkbox)
        self.input_tabs.addTab(instrument_tab, "Instrument")
        self.input_tabs.addTab(star_tab, "Star")
        self.input_tabs.addTab(planet_tab, "Planet")

        left_layout.addWidget(self.input_tabs)

        # Final path after choosing all inputs
        self.path_box = QTextEdit()
        self.path_box.setReadOnly(True)
        self.path_box.setFixedHeight(90)
        self.path_box.setStyleSheet("""
            QTextEdit {
                background-color: #f7f7f7;
                border: 1px solid #aaa;
                font-family: Consolas;
                font-size: 13px;
            }
        """)
        left_layout.addWidget(self.path_box)

        # Separation line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        left_layout.addWidget(line)
        ###

        ### Post-processing box
        post_title = QLabel("Post-processing:")
        post_title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
        """)
        post_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(post_title)

        self.adi_checkbox = QComboBox()
        self.adi_checkbox.addItems(["No", "Yes"])
        self.adi_checkbox.currentTextChanged.connect(self.on_adi_changed)
        self.adi_enabled = False

        self.save_adi_checkbox = QComboBox()
        self.save_adi_checkbox.addItems(["No", "Yes"])
        self.save_adi_checkbox.setCurrentText("Yes")
        self.save_adi_checkbox.currentTextChanged.connect(self.update_save_controls)
        self.save_root = None

        self.save_folder_button = QPushButton("Choose save folder")
        self.save_folder_button.clicked.connect(self.choose_save_root_folder)
        self.save_folder_button.setEnabled(False)

        self.save_folder_label = QLabel("No save folder selected")
        self.save_folder_label.setStyleSheet("color: #555;")

        self.plot_contrast_button = QPushButton("Plot contrast curve")
        self.plot_contrast_button.setFixedHeight(30)
        self.plot_contrast_button.clicked.connect(self.plot_contrast_curve)
        self.plot_contrast_button.setStyleSheet("background-color: #4caf50; color: white; border-radius: 4px;")

        post_layout = QGridLayout()
        post_layout.addWidget(QLabel("ADI:"), 0, 0)
        post_layout.addWidget(self.adi_checkbox, 0, 1)
        post_layout.addWidget(QLabel("Save ADI:"), 1, 0)
        post_layout.addWidget(self.save_adi_checkbox, 1, 1)
        post_layout.addWidget(QLabel("Save folder:"), 2, 0)
        post_layout.addWidget(self.save_folder_button, 2, 1)
        left_layout.addLayout(post_layout)
        left_layout.addWidget(self.save_folder_label)
        left_layout.addWidget(self.plot_contrast_button)

        # Bouton Run
        run_button = QPushButton("Run")
        run_button.setFixedHeight(35)
        run_button.setStyleSheet("""
            QPushButton {
                font-size: 15px;
                font-weight: bold;
                background-color: #2d89ef;
                color: white;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #1b5fa7;
            }
        """)
        run_button.clicked.connect(self.run_processing)
        left_layout.addWidget(run_button)
        left_layout.addStretch(1)

        main_layout.addWidget(left_frame, 0)
        ###


        # Right side: reserved for files / preprocessing output
        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.Shape.StyledPanel)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setSpacing(12)

        # Image display section
        # Tabbed display for ADI image and contrast curve
        self.tab_widget = QTabWidget()

        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        image_layout.setSpacing(8)
        image_layout.setContentsMargins(0, 0, 0, 0)

        image_selector_layout = QHBoxLayout()
        image_selector_layout.addWidget(QLabel("Image:"))
        self.adi_image_combo = QComboBox()
        self.adi_image_combo.addItem("ADI star")
        self.adi_image_combo.currentTextChanged.connect(self.on_adi_image_changed)
        image_selector_layout.addWidget(self.adi_image_combo)
        image_selector_layout.addStretch(1)
        image_layout.addLayout(image_selector_layout)

        self.adi_figure = Figure(figsize=(10, 4), dpi=80)
        self.adi_canvas = FigureCanvas(self.adi_figure)
        image_layout.addWidget(self.adi_canvas)
        self.adi_toolbar = NavigationToolbar(self.adi_canvas, self)
        image_layout.addWidget(self.adi_toolbar)
        self.tab_widget.addTab(image_tab, "ADI Image")

        contrast_tab = QWidget()
        contrast_layout = QVBoxLayout(contrast_tab)
        contrast_layout.setSpacing(8)
        contrast_layout.setContentsMargins(0, 0, 0, 0)

        self.contrast_figure = Figure(figsize=(10, 4), dpi=80)
        self.contrast_canvas = FigureCanvas(self.contrast_figure)
        contrast_layout.addWidget(self.contrast_canvas)
        self.contrast_toolbar = NavigationToolbar(self.contrast_canvas, self)
        contrast_layout.addWidget(self.contrast_toolbar)
        self.tab_widget.addTab(contrast_tab, "Contrast Curve")

        right_layout.addWidget(self.tab_widget)

        # Text output section
        files_title = QLabel("Files / Preprocessing")
        files_title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
        """)
        right_layout.addWidget(files_title)

        self.files_box = QTextEdit()
        self.files_box.setReadOnly(True)
        self.files_box.setPlaceholderText("")
        self.files_box.setStyleSheet("""
            QTextEdit {
                background-color: #f6f6f6;
                border: 1px solid #bbb;
                font-family: Consolas;
                font-size: 13px;
            }
        """)
        self.files_box.setMaximumHeight(80)
        right_layout.addWidget(self.files_box)

        self.preprocessing_function_combo = QComboBox()
        self.user_function_map = self.get_user_function_map()
        self.preprocessing_function_combo.addItems(self.user_function_map.keys())

        self.preprocessing_argument = QLineEdit()
        self.preprocessing_argument.setPlaceholderText("Distance in mas")

        self.preprocessing_run_button = QPushButton("Run function")
        self.preprocessing_run_button.clicked.connect(self.run_preprocessing_function)

        preprocessing_input_layout = QHBoxLayout()
        preprocessing_input_layout.addWidget(self.preprocessing_function_combo, 2)
        preprocessing_input_layout.addWidget(self.preprocessing_argument, 3)
        preprocessing_input_layout.addWidget(self.preprocessing_run_button, 1)
        right_layout.addLayout(preprocessing_input_layout)

        main_layout.addWidget(right_frame, 1)
        self.setLayout(main_layout)

        # Connexions pour mise à jour automatique du chemin
        self.clc_combo.currentTextChanged.connect(self.update_paths)
        self.ncpa_combo.currentTextChanged.connect(self.update_paths)
        self.seeing_combo.currentTextChanged.connect(self.update_paths)
        self.obs_time_combo.currentTextChanged.connect(self.update_paths)
        self.filter_wavelength_combo.currentTextChanged.connect(self.update_paths)
        self.sampling_combo.currentTextChanged.connect(self.update_filter_wavelength_options)
        self.declination_combo.textChanged.connect(self.update_paths)
        self.Distance_combo.currentTextChanged.connect(self.update_paths)
        self.planet_on_checkbox.stateChanged.connect(self.update_paths)
        self.planet_on_checkbox.stateChanged.connect(self.update_planet_controls)

        self.update_planet_controls()
        self.update_filter_wavelength_options()
        self.update_save_controls()

    def get_user_function_map(self):
        return {
            name: func
            for name, func in inspect.getmembers(user_functions, inspect.isfunction)
            if func.__module__ == user_functions.__name__ and not name.startswith("_")
        }

    def parse_function_arguments(self, argument_text):
        if not argument_text:
            return []

        try:
            parsed = ast.literal_eval(f"({argument_text},)")
        except Exception:
            parsed = tuple(part.strip() for part in argument_text.split(","))

        arguments = []
        for value in parsed:
            if isinstance(value, str):
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                else:
                    try:
                        value = ast.literal_eval(value)
                    except Exception:
                        pass
            arguments.append(value)

        return arguments

    def choose_base_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose base folder")

        if folder:
            self.base_folder = Path(folder)
            self.folder_label.setText(f"Base folder: {folder}")
            self.auto_detect_flux_folder()
            self.update_paths()

    def update_planet_controls(self):
        planet_on = self.planet_on_checkbox.isChecked()
        self.Distance_combo.setEnabled(planet_on)
        self.delta_mag_combo.setEnabled(planet_on)
        self.distance_label.setEnabled(planet_on)
        self.delta_mag_label.setEnabled(planet_on)

    def get_planet_folder(self):
        return "With_Planet" if self.planet_on_checkbox.isChecked() else "No_Planet"


    def get_with_planet_path(self, params=None):
        params = params if params is not None else self.get_parameter_values()
        return self.build_input_path("With_Planet", params) if params["planet"] == "With_Planet" else None


    def auto_detect_flux_folder(self):
        """Automatically detect Flux_input folder inside base_folder."""
        flux_path = self.base_folder / "Flux_input"
        
        if flux_path.exists() and flux_path.is_dir():
            self.flux_folder = flux_path
            self.flux_folder_label.setText(str(flux_path))
        else:
            # If not found, set to None and show message
            self.flux_folder = None
            self.flux_folder_label.setText("not found")



    def compute_nearest_zenithal_distance(self, declination):
        """Compute the nearest zenithal distance based on declination and the latitude, find in the .ini file."""

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
        if self.obs_time_combo.currentIndex() == 0:
            return None
        obs_time_text = self.obs_time_combo.currentText()
        return obs_time_text if obs_time_text else None


    


    
    def get_star_spectrum(self, star_mag=None, wavelength=None, display_plot=False, params=None):
        """Prepare the selected spectrum using the parameters of this calculation."""
        params = params if params is not None else self.get_parameter_values()
        if params["flux_folder"] is None:
            raise FileNotFoundError("Flux folder not found. Select a base folder containing 'Flux_input'.")
        star_mag = params["magnitude"] if star_mag is None else float(star_mag)
        if star_mag is None:
            raise ValueError("No star magnitude selected.")
        wavelength = params["wavelength"] if wavelength is None else float(str(wavelength).replace('µm', '').replace('um', '').strip())
        return get_star_spectrum(str(params["flux_folder"]) + "\\", star_mag, wavelength, display_plot=display_plot)
        
    def get_micado_flux(self, delta_t, zenith_distance, flux_dir=None, star_flux=None, wavelength=None, params=None):
        """Prepare instrument inputs and delegate the photon-flux calculation."""
        params = params if params is not None else self.get_parameter_values()
        if flux_dir is None:
            if params["flux_folder"] is None:
                raise FileNotFoundError("Flux folder not found. Select a base folder containing 'Flux_input'.")
            flux_dir = str(params["flux_folder"]) + "\\"
        wavelength = params["wavelength"] if wavelength is None else float(str(wavelength).replace('µm', '').replace('um', '').strip())
        if star_flux is None:
            _, star_flux = self.get_star_spectrum(wavelength=wavelength, params=params)
        aperture_surface = get_aperture_surface(str(params["base_folder"] / "PUPIL" / "Pupil_ELT_v03.fits"))
        return get_micado_flux(
            flux_dir, star_flux, f"{wavelength:.3f}", np.float32(delta_t), aperture_surface,
            airmass=1 / np.cos(np.radians(zenith_distance)), pixel_scale=params["sampling"],
        )

    def get_planet_micado_flux(self, delta_t=None, zenith_distance=None, wavelength=None, path=None, params=None):
        """Compute MICADO photon flux for the planet selected in the UI."""
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
        sampling = self.sampling_combo.currentText()
        # si sampling == 4, on n'autorise pas la bande J
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

        # restaurer la sélection si possible
        if current and current[0] in allowed_filters:
            for i in range(self.filter_wavelength_combo.count()):
                if self.filter_wavelength_combo.itemData(i) == current:
                    self.filter_wavelength_combo.setCurrentIndex(i)
                    break

        if self.filter_wavelength_combo.count() > 0 and self.filter_wavelength_combo.currentIndex() == -1:
            self.filter_wavelength_combo.setCurrentIndex(0)

        self.update_paths()

    def on_adi_changed(self, text):
        self.adi_enabled = text == "Yes"
        if self.adi_enabled:
            self.files_box.setPlaceholderText(
                "ADI enabled: ADI processing will run on the selected path."
            )
        else:
            self.files_box.setPlaceholderText(
                ""
            )
        self.update_save_controls()

    def update_save_controls(self):
        save_enabled = self.adi_enabled and self.save_adi_checkbox.currentText() == "Yes"
        self.save_folder_button.setEnabled(save_enabled)
        if not save_enabled:
            self.save_root = None
            self.save_folder_label.setText("No save folder selected")

    def run_preprocessing_function(self):
        function_name = self.preprocessing_function_combo.currentText()
        argument = self.preprocessing_argument.text().strip()
        selected_function = self.user_function_map.get(function_name)

        if selected_function is None:
            self.files_box.append(f"Unknown function: {function_name}")
            return

        try:
            arguments = self.parse_function_arguments(argument)
            if function_name == "get_contrast_from_contrast_curve_file":
                if self.last_contrast_curve_path is None:
                    self.files_box.append(
                        "No contrast curve FITS available. Generate and save the contrast curve first."
                    )
                    return
                if len(arguments) != 1:
                    self.files_box.append("Please enter only the distance in mas.")
                    return
                arguments = [self.last_contrast_curve_path, arguments[0]]
            result = selected_function(*arguments)

            if function_name == "get_contrast_from_contrast_curve_file":
                plot_function_name = "Contrast"
                if result is None:
                    self.files_box.append(f"Distance {argument} mas is outside the saved contrast curve range.")
                else:
                    self.files_box.append(f"{plot_function_name}({argument}) = {result}")
        except Exception as e:
            self.files_box.append(f"Function failed ({function_name}): {e}")

    def choose_save_root_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose ADI save folder")
        if not folder:
            return
        self.save_root = Path(folder)
        self.save_folder_label.setText(str(self.save_root))


    def get_parameter_values(self, validate=False, for_paths=False):
        """Read the controls once; calculations use numbers, not display labels."""
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
            "noise": self.noise_checkbox.currentText() == "Yes",
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
                raise ValueError("A star magnitude is required for noise.")
            if params["planet"] == "With_Planet" and params["delta_magnitude"] is None:
                raise ValueError("A delta magnitude is required for the planet.")
        return params

    def build_params_slug(self, planet_folder=None):
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

    def write_ds9_fits(self, path, data, product=None):
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
            header["PRODTYPE"] = (product, "Final exported product")
            header["PROCSTAT"] = ("FINAL", "Not a raw ADI input for reprocessing")
            header["NOISEADD"] = (bool(noise), "Random noise added by this processing run")
            header["NOISEPOS"] = (params["noise_stage"] if noise else "NONE", "Noise before/after ADI, or NONE")
            header["CONVOLVD"] = (params["convolved"], "Circular aperture convolution applied")
            header["NORMALIZ"] = (normalized, "Divided by the stellar PSF maximum")
            header["PIXSCALE"] = (params["sampling"], "Angular pixel scale [mas/pixel]")
            header["WAVEUM"] = (params["wavelength"], "Wavelength [micrometre]")
            if normalized:
                header["NORMREF"] = "STAR_PSF_MAX"
                maximum = normalization[labels[0]]
                if np.isfinite(maximum):
                    header["NORMMAX"] = (float(maximum), "PSF maximum used as divisor")
            if params["convolved"]:
                header["CONVKERN"] = "CIRCLE_LAMBDA/2D"
                header["TELDIAM"] = (38.542, "Diameter used for convolution [m]")
        fits.PrimaryHDU(image, header=header).writeto(path, overwrite=True)

    def build_simulation_output_dir(self, save_root):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(save_root) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        self.last_no_planet_output_dir = None
        return output_dir

    def save_noise_inputs(self, output_dir, original_cube, processed_cube, original_psf, processed_psf):
        noise_dir = Path(output_dir) / "No_Planet" / "noisy_images"
        noise_dir.mkdir(parents=True, exist_ok=True)
        params_slug = self.build_params_slug(planet_folder="No_Planet")

        paths = [
            noise_dir / f"{params_slug}_image_cube_original.fits",
            noise_dir / f"{params_slug}_image_cube_noise.fits",
            noise_dir / f"{params_slug}_psf_cube_original.fits",
            noise_dir / f"{params_slug}_psf_cube_noise.fits",
        ]

        self.write_ds9_fits(paths[0], original_cube)
        self.write_ds9_fits(paths[1], processed_cube)
        self.write_ds9_fits(paths[2], original_psf)
        self.write_ds9_fits(paths[3], processed_psf)

        self.files_box.append(
            "Noise inputs saved:\n"
            f"- {paths[0]}\n"
            f"- {paths[1]}\n"
            f"- {paths[2]}\n"
            f"- {paths[3]}"
        )

    def read_config_values_from_path(self, path, section_name, keys):
        """
        Lit un fichier *_misthic_config_test.ini dans le dossier donné
        et récupère une ou plusieurs valeurs dans une section donnée.

        Parameters
        ----------
        path : str or Path
            Dossier contenant le fichier .ini.
        section_name : str
            Nom de la section, par exemple "simuconfig".
        keys : str or list[str]
            Clé ou liste de clés à récupérer.

        Returns
        -------
        value or dict or None
            - Si keys est une string : retourne directement la valeur.
            - Si keys est une liste : retourne un dictionnaire {clé: valeur}.
            - Retourne None en cas d'erreur.
        """

        path = Path(path)

        config_files = list(path.glob("*_misthic_config_test.ini"))

        if not config_files:
            self.files_box.append(
                "Warning: Config file (*_misthic_config_test.ini) not found in path."
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

            # Cas où on demande une seule clé
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

            # Cas où on demande plusieurs clés
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

    

    def process_planet_adi_with_noisy_cubes(
            self,
            output_dir=None,
            apply_photon_noise=False,
            delta_t=None,
            zenith_distance=None, params=None):
        """Legacy planet pipeline: add noise to each cube frame before ADI."""
        from micado_misthic.imgproc import micado_adi
        from astropy.io import fits

        params = params if params is not None else self.get_parameter_values()
        planet_path = self.get_with_planet_path(params)
        if planet_path is None:
            self.files_box.append("Warning: With_Planet path is not available. Skipping planet ADI.")
            return None

        input_files = find_simulation_fits(planet_path)
        cube_file = input_files["image_cube"]
        psf_cube_file = input_files["psf_cube"]
        perf_psf_file = input_files["perfect_psf"]
        if cube_file is None or psf_cube_file is None:
            self.files_box.append(
                "Warning: Could not find image_cube or psf_cube in "
                f"{planet_path}. Skipping planet ADI."
            )
            return None

        if apply_photon_noise and perf_psf_file is None:
            self.files_box.append(
                "Warning: Could not find perfect_psf in "
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
            perf_psf_data = np.asarray(fits.getdata(str(perf_psf_file)), dtype=np.float32)
            frame_exp_time = np.float32(delta_t)
            planet_photon_flux, planet_emission_per_pix, _ = self.get_planet_micado_flux(
                path=planet_path,
                delta_t=delta_t,
                zenith_distance=zenith_distance,
                params=params,
            )

            cube_data, _, _ = scale_to_photons(
                cube_data,
                perf_psf_data,
                planet_photon_flux,
                planet_emission_per_pix,
                frame_exp_time,
                sig_ron=0.,
                no_noise=False,
                silent=True,
            )
            psf_cube_data, _, _ = scale_to_photons(
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
            str(planet_path) + "\\",
        )
        planet_adi_sum = self.normalize_adi_by_psf(
            planet_adi_sum,
            self.last_psf_sum,
            "planet",
            "star",
        )

        if output_dir is not None:
            self.save_planet_adi_result(planet_adi_sum, output_dir)

        self.last_planet_contrast = float(get_planet_contrast(planet_adi_sum))
        self.files_box.append("Legacy planet ADI completed.")
        return planet_adi_sum

    def process_planet_adi_then_add_noise(
            self,
            output_dir=None,
            apply_photon_noise=False,
            delta_t=None,
            zenith_distance=None, params=None):
        from micado_misthic.imgproc import micado_adi
        from astropy.io import fits

        params = params if params is not None else self.get_parameter_values()
        planet_path = self.get_with_planet_path(params)
        if planet_path is None:
            self.files_box.append("Warning: With_Planet path is not available. Skipping planet ADI.")
            return None

        input_files = find_simulation_fits(planet_path)
        cube_file = input_files["image_cube"]
        psf_cube_file = input_files["psf_cube"]
        perf_psf_file = input_files["perfect_psf"]
        adi_file, psf_sum_file = input_files["adi_sum"], input_files["psf_sum"]
        cached_pair = adi_file is not None and psf_sum_file is not None
        mean_file = input_files["mean"] if cached_pair else None
        use_saved_mean = cached_pair and mean_file is not None
        need_cubes = not cached_pair or (apply_photon_noise and not use_saved_mean)
        if need_cubes and (cube_file is None or psf_cube_file is None):
            self.files_box.append(
                "Warning: Could not find image_cube or psf_cube in "
                f"{planet_path}. Provide a matching image_cube_mean with saved ADI/PSF for noise. Skipping planet ADI."
            )
            return None

        if apply_photon_noise and perf_psf_file is None:
            self.files_box.append(
                "Warning: Could not find perfect_psf in "
                f"{planet_path}. Skipping planet ADI."
            )
            return None

        self.files_box.append(f"Loading saved planet ADI: {adi_file.name}" if cached_pair else f"Starting planet ADI: {cube_file.name}")
        QApplication.processEvents()

        cube_data = np.asarray(fits.getdata(str(cube_file)), dtype=np.float32) if need_cubes else None
        psf_cube_data = np.asarray(fits.getdata(str(psf_cube_file)), dtype=np.float32) if need_cubes else None

        if apply_photon_noise:
            if delta_t is None or zenith_distance is None:
                raise ValueError("delta_t and zenith_distance are required for planet photon noise.")

            self.files_box.append(
                "Running planet ADI without noise, then adding photon and emission noise "
                "to the resulting 2D image..."
            )
            perf_psf_data = np.asarray(fits.getdata(str(perf_psf_file)), dtype=np.float32)
            frame_exp_time = np.float32(delta_t)
            planet_photon_flux, planet_emission_per_pix, _ = self.get_planet_micado_flux(
                path=planet_path,
                delta_t=delta_t,
                zenith_distance=zenith_distance,
                params=params,
            )
            if use_saved_mean:
                planet_adi_clean = fits.getdata(adi_file)
                planet_noise_info = mean_file
            else:
                planet_adi_clean, planet_noise_info = run_adi_without_noise(
                    cube_data, psf_cube_data, planet_path, adi_file=adi_file,
                )
            planet_adi_sum = add_noise_after_adi(
                planet_adi_clean,
                perf_psf_data,
                planet_noise_info,
                photon_flux=planet_photon_flux,
                sig_ron=0.,
                emission_flux=planet_emission_per_pix,
                frame_exp_time=frame_exp_time,
                silent=True,
            )
        elif cached_pair:
            planet_adi_sum = fits.getdata(adi_file)
        else:
            planet_adi_sum, _ = micado_adi(
                cube_data,
                psf_cube_data,
                str(planet_path) + "\\",
            )

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
        planet_adi_sum = self.normalize_adi_by_psf(planet_adi_sum, self.last_psf_sum, "planet", "star")

        if output_dir is not None:
            self.save_planet_adi_result(planet_adi_sum, output_dir)

        self.last_planet_contrast = float(get_planet_contrast(planet_adi_sum))
        self.files_box.append("Planet ADI completed.")
        return planet_adi_sum

    def perform_adi_with_noisy_cubes(
            self, path, save_root=None, apply_noise=False, output_dir=None, params=None):
        """Legacy pipeline: add noise to every cube frame before running ADI."""
        from micado_misthic.imgproc import micado_adi
        from astropy.io import fits

        params = dict(params if params is not None else self.get_parameter_values())
        params["noise"] = apply_noise
        params.update(noise_stage="BEFORE", convolved=False, normalization={})
        self.result_params = params
        self.last_adi_sum = self.last_psf_sum = None
        total_time_start = time.time()
        output_dir = Path(output_dir) if output_dir is not None else None
        self.last_no_planet_output_dir = None
        self.last_contrast_curve_path = None
        self.last_adi_source_dir = Path(path)
        self.last_reconstructed_adi_sum = None
        self.last_planet_contrast = None
        input_files = find_simulation_fits(path)
        cube_file = input_files["image_cube"]
        psf_cube_file = input_files["psf_cube"]
        perf_psf_file = input_files["perfect_psf"]

        if cube_file is None or psf_cube_file is None:
            QMessageBox.warning(
                self,
                "Incomplete ADI files",
                "Could not find image_cube or psf_cube files in the selected folder.",
            )
            return

        if apply_noise and perf_psf_file is None:
            QMessageBox.warning(
                self,
                "Incomplete noise files",
                "Could not find perfect_psf file in the selected folder. Noise cannot be applied.",
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
        perf_psf_data = (
            np.asarray(fits.getdata(str(perf_psf_file)), dtype=np.float32)
            if perf_psf_file is not None
            else None
        )
        noise_applied = False
        planet_noise_params = None

        if output_dir is None and save_root is None and params["save_adi"]:
            self.choose_save_root_folder()
            save_root = self.save_root

        if output_dir is None and params["save_adi"] and save_root is not None:
            output_dir = self.build_simulation_output_dir(save_root)

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
                    image_cube_noise, _, _ = scale_to_photons(
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
                    psf_cube_noise, _, _ = scale_to_photons(
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
        adi_sum, psf_sum = micado_adi(cube_data, psf_cube_data, str(path) + "\\")
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
        self.display_adi_results(adi_sum)

        if params["planet"] == "With_Planet":
            try:
                planet_kwargs = {"output_dir": output_dir, "params": params}
                if noise_applied and planet_noise_params is not None:
                    delta_t, zenith_distance = planet_noise_params
                    planet_kwargs.update({
                        "apply_photon_noise": True,
                        "delta_t": delta_t,
                        "zenith_distance": zenith_distance,
                    })
                planet_adi_sum = self.process_planet_adi_with_noisy_cubes(**planet_kwargs)

                if planet_adi_sum is not None:
                    self.reconstruct_adi_image(
                        adi_sum,
                        planet_adi_sum,
                        output_dir=output_dir,
                    )
            except Exception as e:
                self.files_box.append(f"Warning: Could not process planet ADI: {str(e)}")

        total_time_elapsed = time.time() - total_time_start
        self.files_box.append(
            f"Total legacy ADI processing function time: {total_time_elapsed:.1f} s."
        )

    def perform_adi_then_add_noise(
            self, path, save_root=None, apply_noise=False, output_dir=None, params=None):
        from micado_misthic.imgproc import micado_adi
        from astropy.io import fits

        params = dict(params if params is not None else self.get_parameter_values())
        params["noise"] = apply_noise
        params.update(noise_stage="AFTER", convolved=True, normalization={})
        self.result_params = params
        self.last_adi_sum = self.last_psf_sum = None
        total_time_start = time.time()
        output_dir = Path(output_dir) if output_dir is not None else None
        self.last_no_planet_output_dir = None
        self.last_contrast_curve_path = None
        self.last_adi_source_dir = Path(path)
        self.last_reconstructed_adi_sum = None
        self.last_planet_contrast = None
        input_files = find_simulation_fits(path)
        cube_file = input_files["image_cube"]
        psf_cube_file = input_files["psf_cube"]
        perf_psf_file = input_files["perfect_psf"]

        adi_file, psf_sum_file = input_files["adi_sum"], input_files["psf_sum"]
        cached_pair = adi_file is not None and psf_sum_file is not None
        mean_file = input_files["mean"] if cached_pair else None
        use_saved_mean = cached_pair and mean_file is not None
        if apply_noise and not use_saved_mean and (cube_file is None or psf_cube_file is None):
            QMessageBox.warning(
                self, "Incomplete noise files",
                f"Folder: {path}\nADI: {adi_file}\nPSF: {psf_sum_file}\n"
                "Could not find the matching *_image_cube_mean.fits in the selected folder. "
                "Provide this file with ADI_sum, PSF_sum and perfect_psf, or provide the original cubes.",
            )
            return

        if not cached_pair and (cube_file is None or psf_cube_file is None):
            QMessageBox.warning(
                self,
                "Incomplete ADI files",
                "Could not find a matching *_ADI_sum.fits / *_PSF_sum.fits pair or image_cube / psf_cube files.",
            )
            return

        if apply_noise and perf_psf_file is None:
            QMessageBox.warning(
                self,
                "Incomplete noise files",
                "Could not find perfect_psf file in the selected folder. Noise cannot be applied.",
            )
            return

        self.files_box.append(f"Loading saved ADI: {adi_file.name}" if cached_pair else f"Starting ADI processing on: {cube_file.name} and {psf_cube_file.name}...")
        QApplication.processEvents()
        start_time = time.time()


        need_cubes = not cached_pair or (apply_noise and not use_saved_mean)
        # Only load cubes when no saved products can supply the calculation.
        cube_data = np.asarray(fits.getdata(str(cube_file)), dtype=np.float32) if need_cubes and cube_file is not None else None
        psf_cube_data = np.asarray(fits.getdata(str(psf_cube_file)), dtype=np.float32) if need_cubes and psf_cube_file is not None else None
        perf_psf_data = (
            np.asarray(fits.getdata(str(perf_psf_file)), dtype=np.float32)
            if perf_psf_file is not None
            else None
        )
        noise_applied = False
        noise_parameters = None
        planet_noise_params = None

        if output_dir is None and save_root is None and params["save_adi"]:
            self.choose_save_root_folder()
            save_root = self.save_root

        if output_dir is None and params["save_adi"] and save_root is not None:
            output_dir = self.build_simulation_output_dir(save_root)

        # Prepare the noise parameters. The new ADI path applies this noise to
        # the final 2D ADI image instead of modifying every cube frame.
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
                    # Get photon flux and emission data
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
                        "Noise parameters ready: ADI will run on clean cubes, then "
                        f"2D noise will be added (delta_t={delta_t}s)."
                    )
                except Exception as e:
                    self.files_box.append(f"Warning: Could not prepare ADI noise: {str(e)}")
                    # Continue with unnoisy data.

        params["noise"] = noise_applied
        start_time_adi = time.time()
        if noise_applied:
            if use_saved_mean:
                adi_sum_clean = fits.getdata(adi_file)
                noise_info = mean_file
                self.files_box.append(f"Using saved cube mean: {mean_file.name}")
            else:
                adi_sum_clean, noise_info = run_adi_without_noise(
                    cube_data, psf_cube_data, path, adi_file=adi_file,
                )
            try:
                adi_sum = add_noise_after_adi(
                    adi_sum_clean,
                    perf_psf_data,
                    noise_info,
                    silent=True,
                    **noise_parameters,
                )
            except (OSError, ValueError, KeyError) as error:
                QMessageBox.warning(self, "Invalid noise inputs", str(error))
                return
            if use_saved_mean:
                psf_sum = np.asarray(fits.getdata(psf_sum_file), dtype=np.float64)
                psf_sum *= noise_parameters["photon_flux"] / np.sum(perf_psf_data)
            else:
                psf_sum = psf_sum_in_photons(
                    psf_cube_data, perf_psf_data, noise_parameters["photon_flux"],
                )
        elif cached_pair:
            adi_sum = fits.getdata(adi_file)
            psf_sum = fits.getdata(psf_sum_file)
        else:
            adi_sum, psf_sum = micado_adi(cube_data, psf_cube_data, str(path) + "\\")

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
                    "The clean input cubes were kept unchanged; noise was applied "
                    "after ADI on the 2D result."
                )
            self.save_adi_results(adi_sum, psf_sum, output_dir)

        self.last_adi_sum = adi_sum
        self.last_psf_sum = psf_sum
        self.display_adi_results(adi_sum)

        if params["planet"] == "With_Planet":
            try:
                planet_kwargs = {"output_dir": output_dir, "params": params}
                if noise_applied and planet_noise_params is not None:
                    delta_t, zenith_distance = planet_noise_params
                    planet_kwargs.update({
                        "apply_photon_noise": True,
                        "delta_t": delta_t,
                        "zenith_distance": zenith_distance,
                    })
                planet_adi_sum = self.process_planet_adi_then_add_noise(**planet_kwargs)

                if planet_adi_sum is not None:
                    self.reconstruct_adi_image(
                        adi_sum,
                        planet_adi_sum,
                        output_dir=output_dir,
                    )
            except Exception as e:
                self.files_box.append(f"Warning: Could not process planet ADI: {str(e)}")
        

        total_time_elapsed = time.time() - total_time_start
        self.files_box.append(f"Total ADI processing function time: {total_time_elapsed:.1f} s.")
        

    def normalize_adi_by_psf(self, adi_sum, psf_sum, label, psf_label=None):
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
    
    

    def save_adi_results(self, adi_sum, psf_sum, output_dir):
        output_dir = Path(output_dir) / "No_Planet"
        output_dir.mkdir(parents=True, exist_ok=True)
        params_slug = self.build_params_slug(planet_folder="No_Planet")
        self.last_no_planet_output_dir = output_dir

        adi_sum_path = output_dir / f"{params_slug}_adi.fits"
        psf_sum_path = output_dir / f"{params_slug}_psf.fits"

        self.write_ds9_fits(adi_sum_path, adi_sum, product="STAR_ADI")
        self.write_ds9_fits(psf_sum_path, psf_sum, product="PSF")

        self.files_box.append(
            f"No_Planet ADI results saved:\n- {adi_sum_path}\n- {psf_sum_path}"
        )
        return adi_sum_path, psf_sum_path

    def save_planet_adi_result(self, adi_sum, output_dir):
        output_dir = Path(output_dir) / "With_Planet"
        output_dir.mkdir(parents=True, exist_ok=True)
        params_slug = self.build_params_slug(planet_folder="With_Planet")

        adi_sum_path = output_dir / f"{params_slug}_planet_adi.fits"
        self.write_ds9_fits(adi_sum_path, adi_sum, product="PLANET_ADI")

        self.files_box.append(f"With_Planet ADI result saved:\n- {adi_sum_path}")
        return adi_sum_path

    def reconstruct_adi_image(self, star_adi_path, planet_adi_path, output_dir=None):
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
            self.write_ds9_fits(save_path, reconstructed, product="COMBINED")
            self.files_box.append(f"Reconstructed ADI result saved:\n- {save_path}")
        self.files_box.append("Reconstructed ADI image completed with perfCalculator.utils.")
        return reconstructed

    def update_adi_image_options(self):
        current_text = self.adi_image_combo.currentText()
        self.adi_image_combo.blockSignals(True)
        self.adi_image_combo.clear()

        if self.last_adi_sum is not None:
            self.adi_image_combo.addItem("ADI star")
        if self.last_reconstructed_adi_sum is not None:
            self.adi_image_combo.addItem("ADI star + planet")

        index = self.adi_image_combo.findText(current_text)
        if index >= 0:
            self.adi_image_combo.setCurrentIndex(index)
        self.adi_image_combo.blockSignals(False)

    def on_adi_image_changed(self, text):
        if text == "ADI star + planet" and self.last_reconstructed_adi_sum is not None:
            self.display_adi_image(self.last_reconstructed_adi_sum, "ADI star + planet")
        elif self.last_adi_sum is not None:
            self.display_adi_image(self.last_adi_sum, "ADI star")

    def display_adi_results(self, adi_sum):
        """Display ADI results in matplotlib figure"""
        self.update_adi_image_options()
        signals_blocked = self.adi_image_combo.blockSignals(True)
        self.adi_image_combo.setCurrentText("ADI star")
        self.adi_image_combo.blockSignals(signals_blocked)
        self.display_adi_image(adi_sum, "ADI star")

    def display_adi_image(self, image, title):
        self.adi_figure.clear()

        ax = self.adi_figure.add_subplot(111)
        vmin, vmax = np.nanpercentile(image, [1, 99.8])
        pixel_scale_mas = (self.result_params or self.get_parameter_values())["sampling"]
        height, width = image.shape
        # Pixel-edge bounds place the image centre at zero angular offset.
        extent = np.array([-width / 2, width / 2, -height / 2, height / 2]) * pixel_scale_mas
        im = ax.imshow(image, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Angular separation (mas)')
        ax.set_ylabel('Angular separation (mas)')
        self.adi_figure.colorbar(im, ax=ax, label='Intensity')
        self.adi_figure.tight_layout()
        self.adi_canvas.draw()

    def plot_contrast_curve(self):
        if self.last_adi_sum is None or self.last_psf_sum is None:
            QMessageBox.information(
                self,
                "Contrast curve unavailable",
                "No ADI image has been generated yet. Run ADI processing first, then click the plot button.",
            )
            return
        from micado_misthic.analysis import get_rms_contrast

        self.files_box.append("Starting contrast curve generation...")
        QApplication.processEvents()
        start_time = time.time()

        params = self.result_params or self.get_parameter_values()
        pxscale = params["sampling"]

        rms_contrast, x = get_rms_contrast(self.last_adi_sum)
        rms_contrast = 5.0 * rms_contrast

        x = x * pxscale
        planet_distance_mas = None
        planet_contrast = None

        if params["planet"] == "With_Planet" and self.last_planet_contrast is not None:
            planet_distance_mas = params["planet_distance"]
            planet_contrast = self.last_planet_contrast
            self.files_box.append(f"Planet contrast = {planet_contrast:.6e}")

        self.display_contrast_curve(
            x,
            rms_contrast,
            planet_distance_mas=planet_distance_mas,
            planet_contrast=planet_contrast,
        )
        self.last_contrast_curve_path = None
        contrast_output_dir = (
            self.last_no_planet_output_dir.parent
            if self.last_no_planet_output_dir is not None
            else self.last_adi_source_dir
        )
        if contrast_output_dir is not None:
            self.last_contrast_curve_path = self.save_contrast_curve(
                x,
                rms_contrast,
                contrast_output_dir / "Contrast_Curve",
                planet_distance_mas=planet_distance_mas,
                planet_contrast=planet_contrast,
            )
        else:
            self.files_box.append("Warning: contrast curve was not saved because no output folder is available.")
            
        self.tab_widget.setCurrentIndex(1)
        elapsed = time.time() - start_time
        self.files_box.append(f"Contrast curve generated in {elapsed:.1f} s.")

    def convolve_adi_and_psf(self, adi_sum, psf_sum, wavelength, pixel_scale, tel_diameter=38.542):
        """Convolve ADI and PSF with a lambda/(2D) circular aperture.

        wavelength is in microns, pixel_scale in mas/pixel and tel_diameter
        in metres. Return cropped images without normalization; psf_sum may
        be None when only the planet ADI image needs convolution.
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

    def save_contrast_curve(
            self,
            x,
            contrast,
            output_dir,
            planet_distance_mas=None,
            planet_contrast=None):
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

    def display_contrast_curve(self, x, contrast, planet_distance_mas=None, planet_contrast=None):
        self.contrast_figure.clear()

        ax = self.contrast_figure.add_subplot(111)
        ax.plot(x, contrast, color="#1a891a", linewidth=1.8)
        if planet_distance_mas is not None and planet_contrast is not None:
            ax.scatter(
                planet_distance_mas,
                planet_contrast,
                color="black",
                marker="x",
                s=80,
                zorder=5,
                label="Planet",
            )
            ax.legend()
        ax.set_title('5-sigma contrast curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Angular separation (mas)')
        ax.set_ylabel('Contrast, 5-sigma')
        ax.set_yscale('log')
        ax.grid(color='.9')
        ax.set_xlim(left=0)
        self.contrast_figure.tight_layout()
        self.contrast_canvas.draw()

    def build_input_path(self, planet_folder, params):
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
        try:
            params = self.get_parameter_values(for_paths=True)
        except ValueError:
            self.no_planet_path = None
            self.path_box.clear()
            return  # A numeric field can be incomplete while the user is typing.
        self.no_planet_path = self.build_input_path("No_Planet", params)
        self.path_box.setText(str(self.no_planet_path) + "\\")

    def get_clc_folder_candidates(self, clc):
        candidates = [clc]
        legacy_clc = self.clc_legacy_folder_map.get(clc)
        if legacy_clc is not None:
            candidates.append(legacy_clc)
        return candidates

    def resolve_clc_folder(self, parent_folder, clc):
        for candidate in self.get_clc_folder_candidates(clc):
            if (parent_folder / candidate).exists():
                return candidate
        return clc

    def run_processing(self):
        try:
            params = self.get_parameter_values(validate=True)
        except ValueError as error:
            QMessageBox.warning(self, "Invalid parameters", str(error))
            return
        print("========== PARAMETERS ==========", params)
        if self.adi_checkbox.currentText() != "Yes":
            return
        path = self.build_input_path("No_Planet", params)
        self.path_box.setText(str(path) + "\\")
        self.perform_adi_then_add_noise(
            path, save_root=self.save_root if params["save_adi"] else None,
            apply_noise=params["noise"], params=params,
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = InputParametersWindow()
    window.show()

    sys.exit(app.exec())





