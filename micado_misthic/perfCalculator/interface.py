"""
Author: Tristan Deseine
"""

import os
import sys
import ast
import inspect
from importlib.resources import files
from pathlib import Path
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
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFrame,
    QTextEdit,
    QTabWidget,
    QTabBar,
    QTextBrowser,
)
from PyQt6.QtCore import Qt, QLocale
from PyQt6.QtGui import QDoubleValidator

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from micado_misthic.perfCalculator.interface_modules.parameters import ParameterPathMethods
from micado_misthic.perfCalculator.interface_modules.output import OutputMethods
from micado_misthic.perfCalculator.interface_modules.processing import ProcessingMethods
from micado_misthic.perfCalculator.interface_modules.plotting import PlottingMethods

def scan_combinations(root, depth, skip=()):
    """Retourne l'ensemble des combinaisons valides.

    depth : nombre total de niveaux dans l'arborescence (niveaux ignorés inclus)
    skip  : indices (à partir de 0) des niveaux à ignorer
    """
    combos = set()

    def rec(path, level, parts):
        if level == depth:
            combos.add(parts)
            return
        last = level == depth - 1
        for name in os.listdir(path):
            if name.startswith("."):          # .DS_Store, etc.
                continue
            full = os.path.join(path, name)
            if os.path.isdir(full) or last:
                if level in skip:
                    rec(full, level + 1, parts)
                else:
                    value = os.path.splitext(name)[0] if (last and os.path.isfile(full)) else name
                    rec(full, level + 1, parts + (value,))

    rec(root, 0, ())
    return combos

def filter_folder(text):
    """'H - 1.582 µm' -> 'filter1.582'"""
    try:
        return "filter" + text.split(" - ")[1].split()[0]
    except IndexError:
        return text

class ParamAvailability:
    """Grise dans chaque combo les valeurs incompatibles avec les autres choix."""
    def __init__(self, combinations, widgets, converters=None):
        self.combinations = combinations
        self.widgets = widgets
        self.converters = converters or [None] * len(widgets)
        for w in widgets:
            w.currentTextChanged.connect(self.update)
        self.update()

    def _value(self, i, text):
        conv = self.converters[i]
        return conv(text) if conv else text
    
    def set_combinations(self, combinations):
        self.combinations = combinations
        self.update()

    def update(self):
        n = len(self.widgets)
        has_data = bool(self.combinations)
        for w in self.widgets:
            w.setEnabled(has_data)           # all grey until a valid folder is scanned
        if not has_data:
            return
        current = [w.currentText() for w in self.widgets]
        for i, w in enumerate(self.widgets):
            allowed = {
                c[i] for c in self.combinations
                if all(
                    c[j] == self._value(j, current[j])
                    for j in range(n) if j != i and current[j]
                )
            }
            model = w.model()
            for k in range(w.count()):
                model.item(k).setEnabled(self._value(i, w.itemText(k)) in allowed)

class InputParametersWindow(ParameterPathMethods, OutputMethods, ProcessingMethods, PlottingMethods, QWidget):
    """Qt interface for selecting, processing and displaying ADI simulations."""

    def __init__(self):
        """Initialise the performance calculator window and its runtime state.

        Output:
        None
        """
        super().__init__()

        self.setWindowTitle("Performance Calculator")
        self.setMinimumSize(1000, 640)
        self.resize(1080, 700)

        self.base_folder = Path()
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
        self.help_browser = None

        self.init_ui()

    def init_ui(self):
        """Build the controls, plots and Qt signal connections.

        Output:
        None
        """
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(16)

        # Left side : input and processing controls.
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.Shape.StyledPanel)
        left_frame.setMinimumWidth(420)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setSpacing(14)

        # Input section title.
        title = QLabel("Input parameters")
        title.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            text-decoration: underline;
        """)
        help_button = QPushButton("Help")
        help_button.setFixedSize(52, 24)
        help_button.clicked.connect(self.open_help)
        title_layout = QHBoxLayout()
        title_layout.addWidget(title)
        title_layout.addStretch(1)
        title_layout.addWidget(help_button)
        left_layout.addLayout(title_layout)

        ### Base folder
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("Base folder: not selected")
        self.folder_label.setStyleSheet("color: #444;")
        browse_button = QPushButton("Choose folder")
        browse_button.clicked.connect(self.choose_base_folder)
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(browse_button)
        left_layout.addLayout(folder_layout)
        ###

        ### Instrument parameters
        self.clc_combo = QComboBox()
        self.clc_combo.addItems(["CLC15", "CLC25", "CLC50"])
        self.clc_combo.setCurrentText("CLC25")

        self.ncpa_combo = QComboBox()
        self.ncpa_combo.addItems(["No","Yes"])

        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems(["1.5", "4.0"])

        # Available wavelengths must match obsparams.get_micado_flux.select_filter.
        self.filter_wavelength_map = {
            "J": ["1.190 µm", "1.247 µm", "1.270 µm"],
            "H": ["1.582 µm", "1.635 µm", "1.693 µm"],
            "K": ["2.100 µm", "2.150 µm", "2.235 µm"],
        }
        self.filter_wavelength_combo = QComboBox()
        #self.filter_wavelength_combo.setCurrentText("H:" "1.582 µm")

        # Detection noise.
        self.noise_checkbox = QComboBox()
        self.noise_checkbox.addItems(["Yes","No"])

        # Observation time.
        self.obs_time_combo = QComboBox()
        #self.obs_time_combo.addItem("Not specified", None)
        self.obs_time_combo.addItems(["15min", "30min", "1h", "1h30","2h"])
        self.obs_time_combo.setCurrentText("1h")
        ###

        ### Star parameters
        self.seeing_combo = QComboBox()
        self.seeing_combo.addItems(["Q1", "MED", "Q4"])
        self.seeing_combo.setCurrentText("MED")

        # Star magnitude input (real number between 0 and 15)
        self.magnitude_combo = QLineEdit()
        self.magnitude_combo.setPlaceholderText("0.00 - 20.00")
        magnitude_validator = QDoubleValidator(0.0, 15.0, 3, self.magnitude_combo)
        magnitude_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        magnitude_validator.setLocale(QLocale(QLocale.Language.C))
        self.magnitude_combo.setValidator(magnitude_validator)
        self.magnitude_combo.setText("0")

        self.declination_combo = QLineEdit()
        self.declination_combo.setPlaceholderText("-90° - 90°")
        declination_validator = QDoubleValidator(-90.0, 90.0, 3, self.declination_combo)
        declination_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        declination_validator.setLocale(QLocale(QLocale.Language.C))
        self.declination_combo.setValidator(declination_validator)
        self.declination_combo.setText("0.0")


        # Flux folder label (auto-detected from the base folder).
        self.flux_folder_label = QLabel("not selected")
        self.flux_folder_label.setStyleSheet("color: #444;")
        ###

        ### Planet parameters
        self.planet_on_checkbox = QCheckBox("Planet on")
        self.planet_on_checkbox.setChecked(False)

        # Distance to the star in mas.
        self.Distance_combo = QComboBox()
        self.Distance_combo.addItems([
            "15 mas", "30 mas", "50 mas", "75 mas", "100 mas", "125 mas",
            "150 mas", "200 mas", "250 mas", "300 mas", "500 mas", "700 mas",
            "1000 mas", "1400 mas",
        ])

        # Delta magnitude between the star and the planet.
        self.delta_mag_combo = QLineEdit()
        self.delta_mag_combo.setPlaceholderText("0.00 - 20.00")
        self.delta_mag_combo.setText("0")
        delta_mag_validator = QDoubleValidator(0.0, 20.1, 3, self.delta_mag_combo)
        delta_mag_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        delta_mag_validator.setLocale(QLocale(QLocale.Language.C))
        self.delta_mag_combo.setValidator(delta_mag_validator)
        ###

        ### Input parameter tabs
        self.input_tabs = QTabWidget()

        # Instrument tab.
        instrument_tab = QWidget()
        instrument_layout = QGridLayout(instrument_tab)
        instrument_layout.setHorizontalSpacing(12)
        instrument_layout.setVerticalSpacing(8)
        instrument_layout.addWidget(QLabel("Seeing:"), 0, 0)
        instrument_layout.addWidget(self.seeing_combo, 0, 1)
        instrument_layout.addWidget(QLabel("Coronograph:"), 1, 0)
        instrument_layout.addWidget(self.clc_combo, 1, 1)
        instrument_layout.addWidget(QLabel("NCPA:"), 2, 0)
        instrument_layout.addWidget(self.ncpa_combo, 2, 1)
        instrument_layout.addWidget(QLabel("Sampling:"), 3, 0)
        instrument_layout.addWidget(self.sampling_combo, 3, 1)
        instrument_layout.addWidget(QLabel("Filter / Wavelength:"), 4, 0)
        instrument_layout.addWidget(self.filter_wavelength_combo, 4, 1)
        instrument_layout.addWidget(QLabel("Observation time:"), 5, 0)
        instrument_layout.addWidget(self.obs_time_combo, 5, 1)


        # Star tab.
        star_tab = QWidget()
        star_layout = QGridLayout(star_tab)
        star_layout.setHorizontalSpacing(12)
        star_layout.setVerticalSpacing(8)
        star_layout.addWidget(QLabel("Magnitude:"), 1, 0)
        star_layout.addWidget(self.magnitude_combo, 1, 1)
        star_layout.addWidget(QLabel("Declination:"), 2, 0)
        star_layout.addWidget(self.declination_combo, 2, 1)
        star_layout.addWidget(QLabel("Flux folder:"), 3, 0)
        star_layout.addWidget(self.flux_folder_label, 3, 1)

        # Planet tab.
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
        ###

        # Preview of the selected input directory.
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

        ### Post-processing controls
        post_title = QLabel("Post-processing:")
        post_title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
        """)
        post_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(post_title)

        self.adi_checkbox = QComboBox()
        self.adi_checkbox.addItems(["Yes", "No"])
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

        # Run button.
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
        ###

        main_layout.addWidget(left_frame, 0)

        ### Results, plots and user tools
        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.Shape.StyledPanel)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setSpacing(12)

        # ADI image and contrast curve tabs.
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

        # Processing log and optional user functions.
        files_title = QLabel("Files / Tools")
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

        self.user_function_combo = QComboBox()
        self.user_function_map = self.get_user_function_map()
        self.user_function_combo.addItems(self.user_function_map.keys())

        self.user_function_argument = QLineEdit()
        self.user_function_argument.setPlaceholderText("Distance in mas")

        self.user_function_run_button = QPushButton("Run function")
        self.user_function_run_button.clicked.connect(self.run_user_function)

        user_function_layout = QHBoxLayout()
        user_function_layout.addWidget(self.user_function_combo, 2)
        user_function_layout.addWidget(self.user_function_argument, 3)
        user_function_layout.addWidget(self.user_function_run_button, 1)
        right_layout.addLayout(user_function_layout)
        ###

        main_layout.addWidget(right_frame, 1)
        self.setLayout(main_layout)

        # Keep the path preview synchronized with directory-related inputs.
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


        '''try:
            combos = scan_combinations(self.base_folder, depth=7,skip=(3,))
        except OSError as e:
            combos = set()
            print("Scan impossible :", e)

        if combos:'''
        self.availability = ParamAvailability(
            set(),
            [self.clc_combo, self.ncpa_combo, self.seeing_combo,
            self.filter_wavelength_combo, self.sampling_combo, self.obs_time_combo],
            converters=[
                None,
                {"No": "NoNCPA", "Yes": "NCPA"}.get,
                None,
                filter_folder,
                lambda t: f"Samp{t}",
                None,
            ],
        )

    def get_user_function_map(self):
        """
        Collect public functions exposed by ``user_func``.

        Output:
        functions: dict
            Function names mapped to callables.
        """
        return {
            name: func
            for name, func in inspect.getmembers(user_functions, inspect.isfunction)
            if func.__module__ == user_functions.__name__ and not name.startswith("_")
        }

    def refresh_availability(self):
        """Rescan the simulation tree and grey out impossible parameter values."""
        root = self.base_folder / "No_Planet"      # adjust, see note below
        try:
            combos = scan_combinations(str(root), depth=7, skip=(3,))
        except OSError as e:
            combos = set()
            self.files_box.append(f"Tree scan impossible: {e}")
        self.availability.set_combinations(combos)

    def parse_function_arguments(self, argument_text):

        """
        Parse the comma-separated arguments entered in the UI.
        Use to run user functions defined in user_func.py

        Input:
        argument_text: str
            Literal values or comma-separated text.

        Output:
        arguments: list
            Parsed Python values, with plain text kept as strings.
        """
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
        """
        Select the simulation root and refresh dependent paths.

        Output:
        None
        """
        folder = QFileDialog.getExistingDirectory(self, "Choose base folder")

        if folder:
            self.base_folder = Path(folder)
            self.folder_label.setText(f"Base folder: {folder}")
            self.auto_detect_flux_folder()
            self.update_paths()
            self.refresh_availability()

    def open_help(self):
        """Open or select the closable Markdown user-guide tab.

        Output:
        None
        """
        help_text = files(
            "micado_misthic.perfCalculator.interface_modules"
        ).joinpath("Help.md").read_text(encoding="utf-8")
        if self.help_browser is None:
            self.help_browser = QTextBrowser()
            self.help_browser.setOpenExternalLinks(True)
        self.help_browser.setMarkdown(help_text)

        index = self.tab_widget.indexOf(self.help_browser)
        if index == -1:
            index = self.tab_widget.addTab(self.help_browser, "Help")
            close_button = QPushButton("×")
            close_button.setFixedSize(20, 20)
            close_button.clicked.connect(self.close_help)
            self.tab_widget.tabBar().setTabButton(
                index, QTabBar.ButtonPosition.RightSide, close_button,
            )
        self.tab_widget.setCurrentIndex(index)

    def close_help(self):
        """Remove the Help tab from the result area.

        Output:
        None
        """
        index = self.tab_widget.indexOf(self.help_browser)
        if index != -1:
            self.tab_widget.removeTab(index)

    def update_planet_controls(self):
        """
        Enable planet inputs only when planet processing is selected.

        Output:
        None
        """
        planet_on = self.planet_on_checkbox.isChecked()
        self.Distance_combo.setEnabled(planet_on)
        self.delta_mag_combo.setEnabled(planet_on)
        self.distance_label.setEnabled(planet_on)
        self.delta_mag_label.setEnabled(planet_on)

    def on_adi_changed(self, text):
        """Update ADI-dependent controls after the ADI selection changes.

        Input:
        text: str
            Current ADI combo-box value.

        Output:
        None
        """
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
        """Enable output-folder selection only for saved ADI processing.

        Output:
        None
        """
        save_enabled = self.adi_enabled and self.save_adi_checkbox.currentText() == "Yes"
        self.save_folder_button.setEnabled(save_enabled)
        if not save_enabled:
            self.save_root = None
            self.save_folder_label.setText("No save folder selected")

    def run_user_function(self):
        """Run the user-selected function and report its result.
        functions are defined in micado_misthic/perfCalculator/user_func.py

        Output:
        None
        """
        function_name = self.user_function_combo.currentText()
        argument = self.user_function_argument.text().strip()
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
        """Select the root directory for saved processing results.

        Output:
        None
        """
        folder = QFileDialog.getExistingDirectory(self, "Choose ADI save folder")
        if not folder:
            return
        self.save_root = Path(folder)
        self.save_folder_label.setText(str(self.save_root))

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = InputParametersWindow()
    window.show()

    sys.exit(app.exec())
