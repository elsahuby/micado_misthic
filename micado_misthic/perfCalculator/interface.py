import sys
import ast
import inspect
from pathlib import Path
import numpy as np
import time
from datetime import datetime


from micado_misthic.utils.obsparams import *
from configobj import ConfigObj
import user_func as user_functions
from user_func import *

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
        self.last_adi_sum = None
        self.last_psf_sum = None
        self.last_output_dir = None
        self.last_params_slug = None
        self.last_contrast_curve_path = None
        self.last_adi_source_dir = None

        self.init_ui()
        self.update_paths()

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

        # Choix dossier de base (placé avant le formulaire/CLC)
        folder_layout = QHBoxLayout()

        self.folder_label = QLabel("Base folder: C:/...")
        self.folder_label.setStyleSheet("color: #444;")

        browse_button = QPushButton("Choose folder")
        browse_button.clicked.connect(self.choose_base_folder)

        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(browse_button)

        left_layout.addLayout(folder_layout)

        # Formulaire divisé en onglets : Instrument / Scene
        
        self.clc_legacy_folder_map = {
            "CLC15": "CLC0",
            "CLC25": "CLC1",
            "CLC50": "CLC2",
        }
        self.clc_combo = QComboBox()
        self.clc_combo.addItems(["CLC15", "CLC25", "CLC50"])

        self.ncpa_combo = QComboBox()
        self.ncpa_combo.addItems(["Yes", "No"])

        self.seeing_combo = QComboBox()
        self.seeing_combo.addItems(["Q1", "MED", "Q4"])

        # Scene-specific magnitude input (real number between 0 and 15)
        self.magnitude_combo = QLineEdit()
        self.magnitude_combo.setPlaceholderText("0.00 - 15.00")
        magnitude_validator = QDoubleValidator(0.0, 15.0, 3, self.magnitude_combo)
        magnitude_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        magnitude_validator.setLocale(QLocale(QLocale.Language.C))
        self.magnitude_combo.setValidator(magnitude_validator)

        # Planet-specific inputs 
        self.planet_on_checkbox = QCheckBox("Planet on")
        self.planet_on_checkbox.setChecked(False)

        #distance to the star in mas
        self.Distance_combo = QComboBox()
        self.Distance_combo.addItems([
            "15 mas", "30 mas", "50 mas", "75 mas", "100 mas", "125 mas",
            "150 mas", "200 mas", "250 mas", "300 mas", "500 mas", "700 mas",
            "1000 mas", "1400 mas",
        ])
        #delta magnitude between the star and the planet (real number between 0 and 10)
        self.delta_mag_combo = QLineEdit()
        self.delta_mag_combo.setPlaceholderText("0.00 - 10.00")
        delta_mag_validator = QDoubleValidator(0.0, 10.0, 3, self.delta_mag_combo)
        delta_mag_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        delta_mag_validator.setLocale(QLocale(QLocale.Language.C))
        self.delta_mag_combo.setValidator(delta_mag_validator)

        # Flux folder label (auto-detected)
        self.flux_folder_label = QLabel("Flux folder: not selected")
        self.flux_folder_label.setStyleSheet("color: #444;")

        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems(["1.5", "4.0"])

        # Filter + wavelength 
        self.filter_wavelength_map = {
            "J": ["1.19 µm", "1.245 µm", "1.270 µm"],
            "H": ["1.582 µm", "1.635 µm", "1.693 µm"],
            "K": ["2.1 µm", "2.145 µm", "2.235 µm"],
        }
        self.filter_wavelength_combo = QComboBox()
        for filt, wavelengths in self.filter_wavelength_map.items():
            for wavelength in wavelengths:
                self.filter_wavelength_combo.addItem(f"{filt} - {wavelength}", (filt, wavelength))

        # Detection noise 
        self.dark_checkbox = QComboBox()
        self.dark_checkbox.addItems(["No", "Yes"])

        # Création des onglets pour les input parameters
        self.input_tabs = QTabWidget()

        # Instrument tab 
        instrument_tab = QWidget()
        instrument_layout = QGridLayout(instrument_tab)
        instrument_layout.setHorizontalSpacing(12)
        instrument_layout.setVerticalSpacing(8)
        instrument_layout.addWidget(QLabel("Coronographe:"), 0, 0)
        instrument_layout.addWidget(self.clc_combo, 0, 1)
        instrument_layout.addWidget(QLabel("NCPA:"), 1, 0)
        instrument_layout.addWidget(self.ncpa_combo, 1, 1)
        instrument_layout.addWidget(QLabel("Sampling:"), 2, 0)
        instrument_layout.addWidget(self.sampling_combo, 2, 1)
        instrument_layout.addWidget(QLabel("Filter / Wavelength:"), 3, 0)
        instrument_layout.addWidget(self.filter_wavelength_combo, 3, 1)
        instrument_layout.addWidget(QLabel("Detection noise:"), 4, 0)
        instrument_layout.addWidget(self.dark_checkbox, 4, 1)

        # Scene tab 
        scene_tab = QWidget()
        scene_layout = QGridLayout(scene_tab)
        scene_layout.setHorizontalSpacing(12)
        scene_layout.setVerticalSpacing(8)
        scene_layout.addWidget(QLabel("Seeing:"), 0, 0)
        scene_layout.addWidget(self.seeing_combo, 0, 1)
        scene_layout.addWidget(QLabel("Magnitude:"), 1, 0)
        scene_layout.addWidget(self.magnitude_combo, 1, 1)
        scene_layout.addWidget(QLabel("Flux folder:"), 2, 0)
        scene_layout.addWidget(self.flux_folder_label, 2, 1)

        

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
        

        left_layout.addWidget(self.planet_on_checkbox)
        self.input_tabs.addTab(instrument_tab, "Instrument")
        self.input_tabs.addTab(scene_tab, "Star / Field")
        self.input_tabs.addTab(planet_tab, "Planet")

        left_layout.addWidget(self.input_tabs)

        # Affichage du chemin final
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

        # Ligne de séparation
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        left_layout.addWidget(line)

        # Post-processing
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
        self.filter_wavelength_combo.currentTextChanged.connect(self.update_paths)
        self.sampling_combo.currentTextChanged.connect(self.update_filter_wavelength_options)
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

        if not isinstance(parsed, tuple):
            parsed = (parsed,)

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
        planet_enabled = self.planet_on_checkbox.isChecked()
        self.Distance_combo.setEnabled(planet_enabled)
        self.delta_mag_combo.setEnabled(planet_enabled)
        self.distance_label.setEnabled(planet_enabled)
        self.delta_mag_label.setEnabled(planet_enabled)

    def auto_detect_flux_folder(self):
        """Automatically detect Flux_input folder inside base_folder."""
        flux_path = self.base_folder / "Flux_input"
        
        if flux_path.exists() and flux_path.is_dir():
            self.flux_folder = flux_path
            self.flux_folder_label.setText(f"Flux folder: {flux_path}")
        else:
            # If not found, set to None and show message
            self.flux_folder = None
            self.flux_folder_label.setText("Flux folder: not found")

    def get_selected_magnitude(self):
        magnitude = self.magnitude_combo.text().strip()
        return magnitude if magnitude else None

    def get_flux_folder(self):
        return self.flux_folder if self.flux_folder and self.flux_folder.exists() else None
    
    def get_star_spectrum(self, star_mag=None, wavelength=None, display_plot=False):
        """
        Get the star spectrum using the selected magnitude, wavelength, and flux folder.
        
        Parameters
        ----------
        star_mag : float or str, optional
            Star magnitude. If None, uses get_selected_magnitude().
        wavelength : float or str, optional
            Wavelength in micrometers. If None, uses the selected filter wavelength.
        display_plot : bool, optional
            Whether to display the spectrum plot. Default is False.
        
        Returns
        -------
        tuple
            (wavelengths, star_flux) - wavelengths in micrometers and star flux spectrum
        
        Raises
        ------
        FileNotFoundError
            If flux folder is not found.
        ValueError
            If star magnitude or wavelength is invalid or not provided.
        """
        if self.flux_folder is None:
            raise FileNotFoundError("Flux folder not found. Please select a valid base folder containing 'Flux_input'.")
        
        # Get magnitude if not provided
        if star_mag is None:
            mag_text = self.get_selected_magnitude()
            if mag_text is None:
                raise ValueError("No star magnitude selected.")
            try:
                star_mag = float(mag_text)
            except ValueError:
                raise ValueError(f"Invalid magnitude value: {mag_text}")
        else:
            try:
                star_mag = float(star_mag)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid magnitude value: {star_mag}")

        # Determine wavelength if not provided
        if wavelength is None:
            data = self.filter_wavelength_combo.currentData()
            if data is None and self.filter_wavelength_combo.count() > 0:
                self.filter_wavelength_combo.setCurrentIndex(0)
                data = self.filter_wavelength_combo.currentData()
            if data is None:
                raise ValueError("No wavelength selected for star spectrum.")
            _, wavelength = data

        try:
            wavelength_value = float(str(wavelength).replace('µm', '').replace('um', '').strip())
        except Exception:
            raise ValueError(f"Invalid wavelength value: {wavelength}")

        # Convert flux_folder Path to string with trailing slash for obsparams compatibility
        flux_dir = str(self.flux_folder) + "\\" if isinstance(self.flux_folder, Path) else self.flux_folder
        
        # Call the obsparams function to get the star spectrum
        try:
            wave_tr, star_flux = get_star_spectrum(flux_dir, star_mag, wavelength_value, display_plot=display_plot)
            return wave_tr, star_flux
        except Exception as e:
            raise RuntimeError(f"Error computing star spectrum: {e}")
        
    def get_aperture_surface(self, telescope_pupil_file=None):
        telescope_pupil_file = self.base_folder / "PUPIL/Pupil_ELT_v03.fits'"
        from micado_misthic.utils.obsparams import get_aperture_surface
        return get_aperture_surface(telescope_pupil_file)

    def get_micado_flux(self, delta_t, zenith_distance, flux_dir=None, star_flux=None, wavelength=None, path=None):
        """Wrapper around `micado_misthic.utils.obsparams.get_micado_flux`.

        Returns (photon_flux, emission_per_pix, trans_out).
        """
        # Determine flux_dir
        if flux_dir is None:
            if self.flux_folder is None:
                raise FileNotFoundError("Flux folder not found. Please select a valid base folder containing 'Flux_input'.")
            flux_dir = str(self.flux_folder) + "\\"

        # Read config parameters
        key_values = self.read_config_values_from_path(path, "simuconfig", ["delta_t", "zenith_distance"])
        

        # Prepare the spectrum: accept an array or call get_star_spectrum
        if star_flux is None:
            wave_tr, star_flux = self.get_star_spectrum(star_mag=None, wavelength=wavelength, display_plot=False)
        

        # Determine wavelength (um) if not provided
        if wavelength is None:
            data = self.filter_wavelength_combo.currentData()
            if data is None and self.filter_wavelength_combo.count() > 0:
                self.filter_wavelength_combo.setCurrentIndex(0)
                data = self.filter_wavelength_combo.currentData()
            if data is None:
                raise ValueError("No wavelength selected for get_micado_flux.")
            _, wavelength = data

        try:
            wavelength_value = float(str(wavelength).replace('µm', '').replace('um', '').strip())
        except Exception:
            raise ValueError(f"Invalid wavelength value: {wavelength}")

        filter_type = f"{wavelength_value:5.3f}"

        pixel_scale = float(self.sampling_combo.currentText())


        
        frame_exp_time = np.float32(delta_t)
       
        airmass = 1/np.cos(np.radians(zenith_distance))

        # Aperture surface: try reading pupil file under base_folder/PUPIL
        pupil_file = self.base_folder / "PUPIL" / "Pupil_ELT_v03.fits"
        
        aperture_surface = get_aperture_surface(str(pupil_file))
        

        # Call obsparams.get_micado_flux
        
        photon_flux, emission_per_pix, global_transmission = get_micado_flux(flux_dir, star_flux, filter_type, frame_exp_time, aperture_surface, airmass=airmass, pixel_scale=pixel_scale)

        return photon_flux, emission_per_pix, global_transmission
    
    def scale_to_photon(self, img_cube, perf_psf, photon_flux, emission_per_pix, frame_exp_time, sig_ron=15., no_noise=False):
        """
        Wrapper around obsparams.scale_to_photons to convert image levels to photons with noise.
        
        Parameters
        ----------
        img_cube : ndarray
            Image cube to scale (coronagraphic or target image)
        perf_psf : ndarray
            Performance PSF
        photon_flux : float
            Photon flux
        emission_per_pix : float
            Emission per pixel
        frame_exp_time : float
            Frame exposure time
        sig_ron : float, optional
            Readout noise sigma. Default is 15.
        no_noise : bool, optional
            If True, no noise is added. Default is False.
        
        Returns
        -------
        tuple
            (img_cube_noisy, perf_psf_noisy, flux_per_frame)
        """
        from micado_misthic.utils.obsparams import scale_to_photons
        
        img_cube_noisy, perf_psf_noisy, flux_per_frame = scale_to_photons(
            img_cube, perf_psf, photon_flux, emission_per_pix,
            frame_exp_time, sig_ron=sig_ron, no_noise=no_noise, silent=True
        )
        
        return img_cube_noisy, perf_psf_noisy, flux_per_frame

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
                self.files_box.append(f"{plot_function_name}({argument}) = {result}")
        except Exception as e:
            self.files_box.append(f"Function failed ({function_name}): {e}")

    def choose_save_root_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose ADI save folder")
        if not folder:
            return
        self.save_root = Path(folder)
        self.save_folder_label.setText(str(self.save_root))


    def get_parameter_values(self):
        data = self.filter_wavelength_combo.currentData()
        if data is None and self.filter_wavelength_combo.count() > 0:
            self.filter_wavelength_combo.setCurrentIndex(0)
            data = self.filter_wavelength_combo.currentData()

        filt, wavelength = data if data is not None else ("H", "1.582 Âµm")
        wavelength_value = str(wavelength).replace(" ", "").replace("Âµm", "um")

        return {
            "planet": "With_Planet" if self.planet_on_checkbox.isChecked() else "No_Planet",
            "planet_distance": self.Distance_combo.currentText().replace(" mas", ""),
            "clc": self.clc_combo.currentText(),
            "ncpa": "NCPA" if self.ncpa_combo.currentText() == "Yes" else "NoNCPA",
            "seeing": self.seeing_combo.currentText(),
            "filter": filt,
            "wavelength": wavelength_value,
            "sampling": self.sampling_combo.currentText(),
            "magnitude": self.get_selected_magnitude(),
            "noise": "noise" if self.dark_checkbox.currentText() == "Yes" else "no_noise",
        }

    def build_params_slug(self):
        params = self.get_parameter_values()
        slug_parts = [
            params["planet"],
            params["clc"],
            params["ncpa"],
            params["seeing"],
            f"{params['filter']}{params['wavelength']}",
            f"samp{params['sampling']}",
        ]
        if params["magnitude"]:
            slug_parts.append(f"mag{params['magnitude']}")
        if params["planet"] == "With_Planet":
            slug_parts.append(f"pdist{params['planet_distance']}mas")
        slug_parts.append(params["noise"])
        raw_slug = "_".join(slug_parts)
        return "".join(
            char if char.isascii() and (char.isalnum() or char in "._-") else "_"
            for char in raw_slug
        )

    def write_ds9_fits(self, path, data):
        from astropy.io import fits

        image = np.asarray(data, dtype=np.float32)
        fits.PrimaryHDU(image).writeto(path, overwrite=True)

    def build_simulation_output_dir(self, save_root):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        params_slug = self.build_params_slug()
        output_dir = Path(save_root) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        self.last_output_dir = output_dir
        self.last_params_slug = params_slug
        return output_dir

    def save_noise_inputs(self, output_dir, original_cube, processed_cube, original_psf, processed_psf):
        noise_dir = Path(output_dir) / "noisy_images"
        noise_dir.mkdir(parents=True, exist_ok=True)
        params_slug = self.last_params_slug or self.build_params_slug()

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

    def perform_adi_processing(self, path, save_root=None, apply_noise=False):
        from micado_misthic.imgproc import micado_adi
        from astropy.io import fits

        total_time_start = time.time()
        self.last_output_dir = None
        self.last_params_slug = None
        self.last_contrast_curve_path = None
        self.last_adi_source_dir = Path(path)
        cube_file = None
        psf_cube_file = None
        perf_psf_file = None

        for files in Path(path).glob("*.fits"):
            if "image_cube" in files.name:
                cube_file = files
            elif "psf_cube" in files.name:
                psf_cube_file = files
            elif "perfect_psf" in files.name:
                perf_psf_file = files

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

        self.files_box.append(f"Starting ADI processing on: {cube_file.name} and {psf_cube_file.name}...")
        QApplication.processEvents()
        start_time = time.time()


        # Load FITS data before calling micado_adi
        cube_data = fits.getdata(str(cube_file))
        psf_cube_data = fits.getdata(str(psf_cube_file))
        perf_psf_data = fits.getdata(str(perf_psf_file)) if perf_psf_file is not None else None
        original_cube_data = cube_data.copy()
        original_psf_cube_data = psf_cube_data.copy()
        noise_applied = False

        if save_root is None and self.save_adi_checkbox.currentText() == "Yes":
            self.choose_save_root_folder()
            save_root = self.save_root

        output_dir = None
        if self.save_adi_checkbox.currentText() == "Yes" and save_root is not None:
            output_dir = self.build_simulation_output_dir(save_root)

        # Apply noise if requested
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
                    photon_flux, emission_per_pix, _ = self.get_micado_flux(path=path, delta_t=delta_t, zenith_distance=zenith_distance)
                    ellapsed_flux = time.time() - start_time_flux
                    print(f"Photon flux and emission per pixel computed in {ellapsed_flux:.2f} s.")
                    frame_exp_time = np.float32(delta_t)
                    
                    # Scale image and PSF cubes to photons with noise.
                    self.files_box.append("Creating image and PSF cubes with photon noise...")
                    start_time_image_noise = time.time()
                    image_cube_noise, _, _ = self.scale_to_photon(
                        cube_data, perf_psf_data, photon_flux, emission_per_pix, 
                        frame_exp_time, sig_ron=15., no_noise=False
                    )
                    ellapsed_image_noise = time.time() - start_time_image_noise
                    print(f"Image cube with photon noise created in {ellapsed_image_noise:.2f} s.")

                    start_time_PSF_noise = time.time()
                    psf_cube_noise, _, _ = self.scale_to_photon(
                        psf_cube_data, perf_psf_data, photon_flux, emission_per_pix,
                        frame_exp_time, sig_ron=15., no_noise=False
                    )
                    ellapsed_cube_noise = time.time() - start_time_PSF_noise
                    print(f"PSF cube with photon noise created in {ellapsed_cube_noise:.2f} s.")
                    cube_data = image_cube_noise
                    psf_cube_data = psf_cube_noise
                    noise_applied = True
                    self.files_box.append(f"Applied photon noise scaling with delta_t={delta_t}s")
                except Exception as e:
                    self.files_box.append(f"Warning: Could not apply noise scaling: {str(e)}")
                    # Continue with unnoisy data
        start_time_adi = time.time()
        adi_sum, psf_sum = micado_adi(cube_data, psf_cube_data, str(path) + "\\")
        ellapsed_adi = time.time() - start_time_adi
        print(f"ADI processing completed in {ellapsed_adi:.2f} s.")
        elapsed = time.time() - start_time
        self.files_box.append(f"ADI processing completed in {elapsed:.1f} s.")

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
        self.display_adi_results(adi_sum, psf_sum)
        

        total_time_elapsed = time.time() - total_time_start
        self.files_box.append(f"Total ADI processing function time: {total_time_elapsed:.1f} s.")
        
        #sommes des differents processings
        # ellapsed_total = ellapsed_flux + ellapsed_image_noise + ellapsed_cube_noise + ellapsed_adi
        # print(f"Total elapsed time for flux, noise, and ADI processing: {ellapsed_total:.2f} s.")

    def save_adi_results(self, adi_sum, psf_sum, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        params_slug = self.last_params_slug or self.build_params_slug()

        adi_sum_path = output_dir / f"{params_slug}_adi_sum.fits"
        psf_sum_path = output_dir / f"{params_slug}_adi_psf_sum.fits"

        self.write_ds9_fits(adi_sum_path, adi_sum)
        self.write_ds9_fits(psf_sum_path, psf_sum)

        self.files_box.append(
            f"ADI results saved:\n- {adi_sum_path}\n- {psf_sum_path}"
        )

    def display_adi_results(self, adi_sum, psf_sum):
        """Display ADI results in matplotlib figure"""
        self.adi_figure.clear()

        ax = self.adi_figure.add_subplot(111)
        im = ax.imshow(adi_sum, origin='lower')
        ax.set_title('Image Cube ADI', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
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

        sampling = self.sampling_combo.currentText()
        pxscale = float(sampling)

        rms_contrast, x = get_rms_contrast(self.last_adi_sum)
        max_psf = np.max(self.last_psf_sum) if self.last_psf_sum is not None else 1.0
        if max_psf != 0:
            rms_contrast = 5.0 * rms_contrast / max_psf

        x = x * pxscale
        self.display_contrast_curve(x, rms_contrast)
        self.last_contrast_curve_path = None
        contrast_output_dir = self.last_output_dir or self.last_adi_source_dir
        if contrast_output_dir is not None:
            self.last_contrast_curve_path = self.save_contrast_curve(x, rms_contrast, contrast_output_dir)
        self.tab_widget.setCurrentIndex(1)
        elapsed = time.time() - start_time
        self.files_box.append(f"Contrast curve generated in {elapsed:.1f} s.")

    def save_contrast_curve(self, x, contrast, output_dir):
        from astropy.io import fits

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        params_slug = self.last_params_slug or self.build_params_slug()

        fits_path = output_dir / f"{params_slug}_contrast_curve.fits"

        contrast_table = fits.BinTableHDU.from_columns([
            fits.Column(name="separation_mas", array=np.asarray(x, dtype=np.float64), format="D"),
            fits.Column(name="contrast_5sigma", array=np.asarray(contrast, dtype=np.float64), format="D"),
        ])
        contrast_table.writeto(fits_path, overwrite=True)

        self.files_box.append(
            f"Contrast curve saved:\n- {fits_path}"
        )
        return fits_path

    def display_contrast_curve(self, x, contrast):
        self.contrast_figure.clear()

        ax = self.contrast_figure.add_subplot(111)
        ax.plot(x, contrast, color="#a24814", linewidth=1.8)
        ax.set_title('5-sigma contrast curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Angular separation (mas)')
        ax.set_ylabel('Contrast, 5-sigma')
        ax.set_yscale('log')
        ax.grid(color='.9')
        ax.set_xlim(left=0)
        self.contrast_figure.tight_layout()
        self.contrast_canvas.draw()

    def update_paths(self):
        planet_on = self.planet_on_checkbox.isChecked()
        planet_folder = "With_Planet" if planet_on else "No_Planet"
        clc = self.clc_combo.currentText()
        clc_folder = self.resolve_clc_folder(self.base_folder / planet_folder, clc)
        ncpa_value = self.ncpa_combo.currentText()
        ncpa = "NCPA" if ncpa_value == "Yes" else "NoNCPA"
        seeing = self.seeing_combo.currentText()
        data = self.filter_wavelength_combo.currentData()
        if data is None:
            # si pas d'élément sélectionné, tenter de sélectionner le premier
            if self.filter_wavelength_combo.count() > 0:
                self.filter_wavelength_combo.setCurrentIndex(0)
                data = self.filter_wavelength_combo.currentData()

        if data is None:
            # dernier recours: valeurs par défaut
            filt, wavelength = ("H", "1.582 µm")
        else:
            filt, wavelength = data

        wavelength_numeric = wavelength.replace(" ", "").replace("µm", "")
        wavelength_path = wavelength_numeric
        sampling = self.sampling_combo.currentText()

        final_path = (
            self.base_folder
            / planet_folder
            / clc_folder
            / f"{ncpa}"
            / f"{seeing}"
            / f"filter{wavelength_path}"
        )
        if planet_on:
            distance = self.Distance_combo.currentText().replace(" mas", "")
            final_path = final_path / f"p_dist_mas={distance}" / f"Samp{sampling}"
        else:
            final_path = final_path / f"samp{sampling}"

        self.path_box.setText(str(final_path)+ "\\")

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
        planet_folder = "With_Planet" if self.planet_on_checkbox.isChecked() else "No_Planet"
        clc = self.clc_combo.currentText()
        ncpa_value = self.ncpa_combo.currentText()
        ncpa = "NCPA" if ncpa_value == "Yes" else "NoNCPA"
        seeing = self.seeing_combo.currentText()
        data = self.filter_wavelength_combo.currentData()
        if data is None and self.filter_wavelength_combo.count() > 0:
            self.filter_wavelength_combo.setCurrentIndex(0)
            data = self.filter_wavelength_combo.currentData()

        if data is None:
            filt, wavelength = ("H", "1.582 µm")
        else:
            filt, wavelength = data
        sampling = self.sampling_combo.currentText()

        dark_noise = self.dark_checkbox.currentText()
        adi = self.adi_checkbox.currentText()

        print("========== PARAMETERS ==========")
        print(f"Planet: {planet_folder}")
        print(f"CL: {clc}")
        print(f"NCPA: {ncpa}")
        print(f"Seeing: {seeing}")
        print(f"Filter: {filt}")
        print(f"Wavelength: {wavelength}")
        print(f"Sampling: {sampling}")
        print()
        print("====== POST-PROCESSING ======")
        print(f"Detection noise: {dark_noise}")
        print(f"ADI: {adi}")
        print()
        print("Final path:")
        final_path = self.path_box.toPlainText()
        print(final_path)

        save_root = self.save_root if self.save_adi_checkbox.currentText() == "Yes" else None

        if adi == "Yes":
            apply_noise = dark_noise == "Yes"
            self.perform_adi_processing(final_path, save_root=save_root, apply_noise=apply_noise)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = InputParametersWindow()
    window.show()

    sys.exit(app.exec())







