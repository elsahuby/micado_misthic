"""
Author: Tristan Deseine, Pierre Baudoz
"""

import time

import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox


class PlottingMethods:
    """ADI-image and contrast-curve display helpers for the interface."""

    def update_adi_image_options(self):
        """Refresh the image selector from currently available results.

        Output:
        None
        """
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
        """Display the image selected in the ADI result combo box.

        Input:
        text: str
            Selected image label.

        Output:
        None
        """
        if text == "ADI star + planet" and self.last_reconstructed_adi_sum is not None:
            self.draw_adi_image(self.last_reconstructed_adi_sum, "ADI star + planet")
        elif self.last_adi_sum is not None:
            self.draw_adi_image(self.last_adi_sum, "ADI star")

    def show_stellar_adi_result(self, adi_sum):
        """Refresh the image choices, select the star and draw its new ADI result.

        Input:
        adi_sum: ndarray
            Final stellar ADI image.

        Output:
        None
        """
        self.update_adi_image_options()
        # Prevent the combo-box callback from drawing the same image a second time.
        signals_blocked = self.adi_image_combo.blockSignals(True)
        self.adi_image_combo.setCurrentText("ADI star")
        self.adi_image_combo.blockSignals(signals_blocked)
        self.draw_adi_image(adi_sum, "ADI star")

    def draw_adi_image(self, image, title):
        """Draw one ADI image on the Qt canvas with angular coordinates.

        Input:
        image: ndarray
            Two-dimensional image to display.
        title: str
            Plot title.

        Output:
        None
        """
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
        self.tab_widget.setCurrentWidget(self.image_tab)

    def plot_contrast_curve(self):
        """Compute, display and export the current 5-sigma contrast curve.

        Output:
        None
        """
        silent = self.silentGUI

        if self.last_adi_sum is None or self.last_psf_sum is None:
            QMessageBox.information(
                self,
                "Contrast curve unavailable",
                "No ADI image has been generated yet. Run ADI processing first, then click the plot button.",
            )
            return
        from micado_misthic.analysis import get_rms_contrast

        if not silent:
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
        planet_pa = None
        planet_name = None
        planet_delta_mag = None
        info_planets = self.create_info_planets(params)

        legend_conditions = [f'{params["clc"]} - seeing: {params["seeing"]} - {params["ncpa"]} - filter: {params["filter"]} - sampling: {params["sampling"]}mas',
                              f'Star mag: {params["magnitude"]} - Zenith dist: {params["z_dist"]}° - Observation time= {params["observation_time"]} - DIT: {params["exp_time_dit"]:.3f}s']                        

        self.display_contrast_curve(x, rms_contrast, info_planets, legend_conditions=legend_conditions)
        self.last_contrast_curve_path = None
        contrast_output_dir = (
            self.last_corostar_output_dir.parent
            if self.last_corostar_output_dir is not None
            else self.last_adi_source_dir
        )
        if contrast_output_dir is not None:
            self.last_contrast_curve_path = self.save_contrast_curveGUI(x, rms_contrast, contrast_output_dir / "Contrast_Curve",
                planet_distance_mas=planet_distance_mas, planet_contrast=planet_contrast, planet_pa=planet_pa, planet_name = planet_name, planet_delta_mag = planet_delta_mag)
        else:
            self.log_warning_once("Warning: contrast curve was not saved because no output folder is available.")
            
        self.tab_widget.setCurrentIndex(1)
        elapsed = time.time() - start_time
        if not silent:
            self.files_box.append(f"Contrast curve generated in {elapsed:.1f} s.")

    def display_contrast_curve(self, x, contrast, info_planets = None, legend_conditions = None):
        """Render the contrast curve and its optional planet marker.

        Input:
        x, contrast: ndarray
            Angular separations and 5-sigma contrast values.
        planet_distance_mas, planet_contrast: float, optional
            Planet marker coordinates.

        Output:
        None
        """
        self.contrast_figure.clear()
        markers = ["o", "s", "^", "v", "x","D", "*", "P", "X"]        
        ax = self.contrast_figure.add_subplot(111)
        ax.plot(x, contrast, color="#1a891a", linewidth=1.8)
        ax.scatter([],[], marker="none", label =legend_conditions[0])
        ax.scatter([],[], marker="none", label =legend_conditions[1])
        if info_planets is not None:
            if len(info_planets["distance_mas"]) >0:
                for p_dist, p_contrast,p_name,m_marker in zip(info_planets["distance_mas"],info_planets["contrast"],info_planets["legend_name"], markers):
                    ax.scatter(p_dist, p_contrast, marker=m_marker, color="red",s=80, zorder=5, label=p_name,)
            y_max = np.nanmax((np.max(np.array(info_planets["contrast"])),np.nanmax(contrast)))
        else:
            y_max = np.nanmax(contrast)
        ax.legend(framealpha=0.3)
        ax.set_title('5-sigma contrast curve', fontsize=10, fontweight='bold')
        ax.set_xlabel('Angular separation (mas)')
        ax.set_ylabel('Contrast, 5-sigma')
        ax.set_yscale('log')
        ax.grid(color='.9')
        ax.set_xlim(left=0)
        ax.set_ylim(top=10*y_max)
        self.contrast_figure.tight_layout()
        self.contrast_canvas.draw()
        self.tab_widget.setCurrentWidget(self.contrast_tab)

    def display_info_simul(self, exp_time_dit, params):
        
        fov = 4096. / 1.3 * exp_time_dit * params["sampling"] / 1000.
        if fov < 0.5:
            self.info_simul.setHtml(f"""
                <span style="color: red;">SATURATION RISK:</span><br>
                DIT is too low <span style="color: red;">({exp_time_dit:.3f} s) </span>to get a fov > 0.5 arcsec x 6 arcsec<br>
                Here FOV: <span style="color: red;">{fov:.2f} arcsec x 6 arcsec</span><br>
                <span style="color: red;">Choose fainter star or use ND filters</span>"""
            )
        if fov >= 0.5 and fov < 6:
            self.info_simul.setHtml(f"""
                <span style="color: orange;">SATURATION RISK:</span><br>
                DIT <span style="color: orange;">({exp_time_dit:.3f} s) </span>requires smaller FOV<br>
                Here FOV: <span style="color: orange;">({fov:.2f} arcsec x 6 arcsec)</span> <br>
                <span style="color: orange;">or using ND filters)</span>"""
            )
        if fov > 6 and exp_time_dit >10:
            self.info_simul.setHtml(f"""
                <span style="color: green;">NO SATURATION RISK:</span><br>
                Optimized DIT for FWC: <span style="color: green;">({exp_time_dit:.3f} s) </span><br>
                is replaced by DIT=10s for STARLOC"""
            )
        if fov > 6 and exp_time_dit <=10:
            self.info_simul.setHtml(f"""
                <span style="color: green;">NO SATURATION RISK:</span><br>
                Optimized DIT for FWC: <span style="color: green;">({exp_time_dit:.3f} s) </span><br>"""
            )