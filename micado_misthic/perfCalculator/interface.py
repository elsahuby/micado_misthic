import sys
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QComboBox,
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
from PyQt6.QtCore import Qt

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
        self.last_adi_sum = None
        self.last_psf_sum = None

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

        # Formulaire
        form_layout = QGridLayout()
        form_layout.setHorizontalSpacing(15)
        form_layout.setVerticalSpacing(12)

        # CLC
        self.clc_combo = QComboBox()
        self.clc_combo.addItems(["CLC0", "CLC1", "CLC2"])
        form_layout.addWidget(QLabel("CLC:"), 0, 0)
        form_layout.addWidget(self.clc_combo, 0, 1)

        # NCPA
        self.ncpa_combo = QComboBox()
        self.ncpa_combo.addItems(["Yes", "No"])
        form_layout.addWidget(QLabel("NCPA:"), 1, 0)
        form_layout.addWidget(self.ncpa_combo, 1, 1)

        # Seeing
        self.seeing_combo = QComboBox()
        self.seeing_combo.addItems(["Q1", "MED", "Q4"])
        form_layout.addWidget(QLabel("Seeing:"), 2, 0)
        form_layout.addWidget(self.seeing_combo, 2, 1)

        # Sampling (affiché avant Filter/Wavelength)
        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems(["1.5", "4.0"])
        form_layout.addWidget(QLabel("Sampling:"), 3, 0)
        form_layout.addWidget(self.sampling_combo, 3, 1)

        # Filter + wavelength ensemble
        self.filter_wavelength_map = {
            "J": ["1.19 µm", "1.245 µm", "1.270 µm"],
            "H": ["1.582 µm", "1.635 µm", "1.693 µm"],
            "K": ["2.1 µm", "2.145 µm", "2.235 µm"],
        }
        self.filter_wavelength_combo = QComboBox()
        for filt, wavelengths in self.filter_wavelength_map.items():
            for wavelength in wavelengths:
                self.filter_wavelength_combo.addItem(f"{filt} - {wavelength}", (filt, wavelength))
        form_layout.addWidget(QLabel("Filter / Wavelength:"), 4, 0)
        form_layout.addWidget(self.filter_wavelength_combo, 4, 1)

        left_layout.addLayout(form_layout)

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

        self.dark_checkbox = QComboBox()
        self.dark_checkbox.addItems(["No", "Yes"])

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
        post_layout.addWidget(QLabel("Detection noise:"), 0, 0)
        post_layout.addWidget(self.dark_checkbox, 0, 1)
        post_layout.addWidget(QLabel("ADI:"), 1, 0)
        post_layout.addWidget(self.adi_checkbox, 1, 1)
        post_layout.addWidget(QLabel("Save ADI:"), 2, 0)
        post_layout.addWidget(self.save_adi_checkbox, 2, 1)
        post_layout.addWidget(QLabel("Save folder:"), 3, 0)
        post_layout.addWidget(self.save_folder_button, 3, 1)
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
        files_title = QLabel("Fichiers / Préprocessing")
        files_title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
        """)
        right_layout.addWidget(files_title)

        self.files_box = QTextEdit()
        self.files_box.setReadOnly(True)
        self.files_box.setPlaceholderText("Espace réservé pour les fichiers avant ou après le preprocessing.")
        self.files_box.setStyleSheet("""
            QTextEdit {
                background-color: #f6f6f6;
                border: 1px solid #bbb;
                font-family: Consolas;
                font-size: 13px;
            }
        """)
        self.files_box.setMaximumHeight(120)
        right_layout.addWidget(self.files_box)

        main_layout.addWidget(right_frame, 1)
        self.setLayout(main_layout)

        # Connexions pour mise à jour automatique du chemin
        self.clc_combo.currentTextChanged.connect(self.update_paths)
        self.ncpa_combo.currentTextChanged.connect(self.update_paths)
        self.seeing_combo.currentTextChanged.connect(self.update_paths)
        self.filter_wavelength_combo.currentTextChanged.connect(self.update_paths)
        self.sampling_combo.currentTextChanged.connect(self.update_filter_wavelength_options)

        self.update_filter_wavelength_options()
        self.update_save_controls()

    def choose_base_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose base folder")

        if folder:
            self.base_folder = Path(folder)
            self.folder_label.setText(f"Base folder: {folder}")
            self.update_paths()

    def update_filter_wavelength_options(self):
        sampling = self.sampling_combo.currentText()
        # si sampling == 4 (numérique), on n'autorise pas la bande J
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
                "ADI activé : le traitement ADI sera exécuté au Run sur le chemin sélectionné."
            )
        else:
            self.files_box.setPlaceholderText(
                "Espace réservé pour les fichiers avant ou après le preprocessing."
            )
        self.update_save_controls()

    def update_save_controls(self):
        save_enabled = self.adi_enabled and self.save_adi_checkbox.currentText() == "Yes"
        self.save_folder_button.setEnabled(save_enabled)
        if not save_enabled:
            self.save_root = None
            self.save_folder_label.setText("No save folder selected")

    def choose_save_root_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sauvegarde ADI")
        if not folder:
            return
        self.save_root = Path(folder)
        self.save_folder_label.setText(str(self.save_root))

    def perform_adi_processing(self, path, save_root=None):
        from micado_misthic.imgproc import micado_adi
        from astropy.io import fits

        cube_file = None
        psf_cube_file = None

        for files in Path(path).glob("*.fits"):
            if "image_cube" in files.name:
                cube_file = files
            elif "psf_cube" in files.name:
                psf_cube_file = files

        if cube_file is None or psf_cube_file is None:
            QMessageBox.warning(
                self,
                "Fichiers ADI incomplets",
                "Impossible de trouver les fichiers image_cube ou psf_cube dans le dossier sélectionné.",
            )
            return

        print(f"Processing ADI on {cube_file} and {psf_cube_file}")
        
        # Load FITS data before calling micado_adi
        cube_data = fits.getdata(str(cube_file))
        psf_cube_data = fits.getdata(str(psf_cube_file))
        
        adi_sum, psf_sum = micado_adi(cube_data, psf_cube_data, str(path) + "\\")

        if save_root is None and self.save_adi_checkbox.currentText() == "Yes":
            self.choose_save_root_folder()
            save_root = self.save_root

        if self.save_adi_checkbox.currentText() == "Yes" and save_root is not None:
            output_dir = self.build_adi_output_dir(save_root, path)
            self.save_adi_results(adi_sum, psf_sum, output_dir, cube_file, psf_cube_file)

        self.last_adi_sum = adi_sum
        self.last_psf_sum = psf_sum

    def build_adi_output_dir(self, save_root, selected_path):
        try:
            relative_path = Path(selected_path).relative_to(self.base_folder)
        except ValueError:
            relative_path = Path(selected_path).name

        output_dir = Path(save_root) / "ADI" / relative_path
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def save_adi_results(self, adi_sum, psf_sum, output_dir, cube_file=None, psf_cube_file=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        adi_sum_path = output_dir / f"adi_sum_{cube_file.stem if cube_file else 'result'}.fits"
        psf_sum_path = output_dir / f"adi_psf_{psf_cube_file.stem if psf_cube_file else 'result'}.fits"

        from astropy.io import fits

        fits.writeto(adi_sum_path, adi_sum, overwrite=True)
        fits.writeto(psf_sum_path, psf_sum, overwrite=True)

        self.files_box.append(
            f"ADI results saved:\n- {adi_sum_path}\n- {psf_sum_path}"
        )
        
        # Display the results in the matplotlib figure
        self.display_adi_results(adi_sum, psf_sum)

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
                "Aucune image ADI n'a encore été générée. Lancez d'abord le traitement ADI, puis cliquez sur le bouton de tracé.",
            )
            return

        from micado_misthic.analysis import get_rms_contrast

        sampling = self.sampling_combo.currentText()
        pxscale = float(sampling)

        rms_contrast, x = get_rms_contrast(self.last_adi_sum)
        max_psf = np.max(self.last_psf_sum) if self.last_psf_sum is not None else 1.0
        if max_psf != 0:
            rms_contrast = 5.0 * rms_contrast / max_psf

        x = x * pxscale
        self.display_contrast_curve(x, rms_contrast)
        self.tab_widget.setCurrentIndex(1)
        self.files_box.append("Courbe de contraste générée.")

    def display_contrast_curve(self, x, contrast):
        self.contrast_figure.clear()

        ax = self.contrast_figure.add_subplot(111)
        ax.plot(x, contrast, color="#a24814", linewidth=1.8)
        ax.set_title('Courbe de contraste 5-sigma', fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance du centre (mas)')
        ax.set_ylabel('Contraste, 5-sigma')
        ax.set_yscale('log')
        ax.grid(color='.9')
        ax.set_xlim(left=0)
        self.contrast_figure.tight_layout()
        self.contrast_canvas.draw()

    def update_paths(self):
        clc = self.clc_combo.currentText()
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
            / clc
            / f"{ncpa}"
            / f"{seeing}"
            / f"filter{wavelength_path}"
            / f"samp{sampling}"
        
        )

        self.path_box.setText(str(final_path)+ "\\")

    def run_processing(self):
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
            self.perform_adi_processing(final_path, save_root=save_root)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = InputParametersWindow()
    window.show()

    sys.exit(app.exec())







