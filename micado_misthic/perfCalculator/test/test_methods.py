"""
This script was used to test and analyze the two different methods : ADI then noise AND Noise then ADI

Comments indicate which parameters can be changed.
Author: Tristan Deseine
"""

from pathlib import Path
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

from micado_misthic.analysis import get_rms_contrast
from micado_misthic.imgproc import get_circle_mask, micado_adi
from micado_misthic.utils import (
    get_aperture_surface,
    get_micado_flux,
    get_star_spectrum,
    scale_to_photons,
)

from micado_misthic.perfCalculator.utils import (
    add_noise_after_adi,
    convolve_adi_image,
    find_simulation_fits,
    save_contrast_curve,
    normalize_by_psf_max,
    psf_sum_in_photons,
    run_adi_without_noise,
)





def select_frames_and_angles(
    cube_data,
    psf_cube_data,
    test_dir,
    output_dir,
    n_frames,
    parangle_file=None,
):
    """Select the same frames from the image, PSF and angle table."""
    total_frames = cube_data.shape[0]
    if psf_cube_data.shape[0] != total_frames:
        raise ValueError(
            "Image and PSF cubes must contain the same number of frames: "
            f"{total_frames} != {psf_cube_data.shape[0]}."
        )

    if n_frames is None and parangle_file is None:
        return cube_data, psf_cube_data, test_dir

    selected_count = total_frames if n_frames is None else n_frames
    if not 1 <= selected_count <= total_frames:
        raise ValueError(
            f"n_frames must be between 1 and {total_frames}, got {selected_count}."
        )
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        selected_count,
        dtype=int,
    )

    if parangle_file is None:
        parangle_file = next(Path(test_dir).glob("*_parangle_tab.txt"), None)
        if parangle_file is None:
            raise FileNotFoundError(f"No parallactic-angle table found in {test_dir}")
    else:
        parangle_file = Path(parangle_file)
    if not parangle_file.is_file():
        raise FileNotFoundError(f"Parallactic-angle file not found: {parangle_file}")

    angles = np.atleast_1d(np.loadtxt(parangle_file, skiprows=1))
    if len(angles) < total_frames:
        raise ValueError(
            f"The angle table contains {len(angles)} values for "
            f"{total_frames} frames."
        )

    angle_dir = Path(output_dir) / "selected_parangles"
    angle_dir.mkdir(parents=True, exist_ok=True)
    angle_file = angle_dir / "selected_parangle_tab.txt"
    np.savetxt(
        angle_file,
        angles[frame_indices],
        header="parallactic_angle",
    )

    print(f"### Parallactic-angle source: {parangle_file}")
    print(f"### Selected {selected_count}/{total_frames} frames.")
    return cube_data[frame_indices], psf_cube_data[frame_indices], angle_dir


def find_config_file(path):
    path = Path(path)
    config_files = list(path.glob("*.ini"))
    if not config_files:
        raise FileNotFoundError(f"No .ini config file found in {path}")
    return config_files[0]


def get_sampling_from_path(path):
    for part in Path(path).parts:
        if part.lower().startswith("samp"):
            return float(part[4:])
    raise ValueError(f"Could not infer sampling from path: {path}")


def get_resampled_data_root(path):
    path = Path(path).resolve()

    for candidate in [path, *path.parents]:
        if (candidate / "Flux_input").exists() and (candidate / "PUPIL").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find resampled data root containing Flux_input and PUPIL "
        f"from {path}"
    )


def read_ini_value(config_file, section_name, key_name):
    current_section = None

    with open(config_file, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line.strip("[]").strip()
                continue

            if current_section != section_name or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.split("#", 1)[0].strip()

            if key == key_name:
                return value

    raise KeyError(f"Could not find [{section_name}] {key_name} in {config_file}")


def get_noise_parameters(path, star_mag, zenith_distance):
    config_file = find_config_file(path)

    delta_t = np.float32(read_ini_value(config_file, "simuconfig", "delta_t"))
    wavelength = float(read_ini_value(config_file, "waveconfig", "lbd0"))
    filter_type = f"{wavelength:5.3f}"
    pixel_scale = get_sampling_from_path(path)

    data_root = get_resampled_data_root(path)
    flux_dir = str(data_root / "Flux_input") + os.sep
    pupil_file = data_root / "PUPIL" / "Pupil_ELT_v03.fits"

    _, star_flux = get_star_spectrum(flux_dir, star_mag, wavelength)
    aperture_surface = get_aperture_surface(str(pupil_file))
    airmass = 1 / np.cos(np.radians(zenith_distance))

    photon_flux, emission_flux, _ = get_micado_flux(
        flux_dir,
        star_flux,
        filter_type,
        delta_t,
        aperture_surface,
        airmass=airmass,
        pixel_scale=pixel_scale,
    )

    print(f"### Config file: {config_file.name}")
    print(f"### delta_t={delta_t}, zenith_distance={zenith_distance}")
    print(f"### wavelength={wavelength}, pixel_scale={pixel_scale}, star_mag={star_mag}")
    print(f"### photon_flux={photon_flux:.3e}, emission_flux={emission_flux:.3e}")

    return photon_flux, emission_flux, delta_t


def run_before_after_adi_test(
    test_dir,
    output_dir=Path("output") / "adi_before_after_test",
    star_mag=12,
    sig_ron=15.0,
    n_frames=None,
    parangle_file=None,
    zenith_distance=40,
    tel_diameter=38.542,
    apply_convolution=False,
):
    test_dir = Path(test_dir)
    run_suffix = (
        f"mag{star_mag:g}_ron{sig_ron:g}_z{zenith_distance:g}"
        .replace(".", "p")
    )
    if n_frames is not None:
        run_suffix += f"_frames{n_frames}"
    output_dir = Path(output_dir) / run_suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n### Test folder: {test_dir}")

    photon_flux, emission_flux, frame_exp_time = get_noise_parameters(
        test_dir,
        star_mag=star_mag,
        zenith_distance=zenith_distance,
    )
    # Uncomment the next line to exclude background emission from the test.
    # emission_flux = 0.0
    input_files = find_simulation_fits(test_dir)
    cube_file = input_files["image_cube"]
    psf_cube_file = input_files["psf_cube"]
    perf_psf_file = input_files["perfect_psf"]
    if any(file is None for file in (cube_file, psf_cube_file, perf_psf_file)):
        raise FileNotFoundError(f"Missing image_cube, psf_cube or perfect_psf in {test_dir}")

    cube_data = np.asarray(fits.getdata(str(cube_file)), dtype=np.float32)
    psf_cube_data = np.asarray(fits.getdata(str(psf_cube_file)), dtype=np.float32)
    perf_psf_data = np.asarray(fits.getdata(str(perf_psf_file)), dtype=np.float32)
    cube_data, psf_cube_data, parallactic_dir = select_frames_and_angles(
        cube_data,
        psf_cube_data,
        test_dir,
        output_dir,
        n_frames,
        parangle_file=parangle_file,
    )
    psf_sum_photon = psf_sum_in_photons(
        psf_cube_data,
        perf_psf_data,
        photon_flux,
    )

    pixel_scale = get_sampling_from_path(test_dir)
    normalization_psf = psf_sum_photon
    if apply_convolution:
        wavelength = float(read_ini_value(find_config_file(test_dir), "waveconfig", "lbd0"))
        l_over_d_pix = wavelength * 1e-6 / tel_diameter * 180. / np.pi * 3600 * 1e3 / pixel_scale
        mask = get_circle_mask(psf_sum_photon.shape, 0.5 * l_over_d_pix)
        normalization_psf = convolve_adi_image(psf_sum_photon, mask)

    used_n_frames = cube_data.shape[0]
    frame_suffix = f"_{used_n_frames}frames" + ("_conv" if apply_convolution else "")
    psf_output = output_dir / f"psf_sum_photon{frame_suffix}.fits"
    fits.writeto(psf_output, normalization_psf.astype(np.float32), overwrite=True)
    avant_output = output_dir / f"avant_adi_sum{frame_suffix}.fits"
    apres_output = output_dir / f"apres_adi_noisy_normalized{frame_suffix}.fits"

    print("\n### AVANT : bruit sur le cube puis ADI")
    start_avant = time.time()
    cube_noise_avant, _, _ = scale_to_photons(
        cube_data,
        perf_psf_data,
        photon_flux,
        emission_flux,
        frame_exp_time,
        sig_ron=sig_ron,
        no_noise=False,
        silent=False,
    )
    psf_cube_noise_avant, _, _ = scale_to_photons(
        psf_cube_data,
        perf_psf_data,
        photon_flux,
        emission_flux,
        frame_exp_time,
        sig_ron=sig_ron,
        no_noise=False,
        silent=False,
    )
    avant_adi_sum, _ = micado_adi(
        cube_noise_avant,
        psf_cube_noise_avant,
        str(parallactic_dir) + os.sep,
    )
    if apply_convolution:
        avant_adi_sum = convolve_adi_image(avant_adi_sum, mask)
    avant_adi_sum = normalize_by_psf_max(avant_adi_sum, normalization_psf)
    fits.writeto(avant_output, avant_adi_sum.astype(np.float32), overwrite=True)

    elapsed_avant = time.time() - start_avant

    print("\n### APRES : ADI sans bruit puis bruit sur l'image ADI")
    start_apres = time.time()
    apres_adi_clean, noise_info = run_adi_without_noise(
        cube_data,
        psf_cube_data,
        parallactic_dir,
        adi_file=input_files["adi_sum"] if Path(parallactic_dir) == test_dir else None,
    )
    apres_adi_noisy = add_noise_after_adi(
        apres_adi_clean,
        perf_psf_data,
        noise_info,
        photon_flux=photon_flux,
        sig_ron=sig_ron,
        emission_flux=emission_flux,
        frame_exp_time=frame_exp_time,
        silent=False,
    )
    if apply_convolution:
        apres_adi_noisy = convolve_adi_image(apres_adi_noisy, mask)
    apres_adi_noisy = normalize_by_psf_max(apres_adi_noisy, normalization_psf)
    elapsed_apres = time.time() - start_apres

    fits.writeto(apres_output, apres_adi_noisy.astype(np.float32), overwrite=True)

    avant_contrast, avant_separation = get_rms_contrast(avant_adi_sum)
    apres_contrast, apres_separation = get_rms_contrast(apres_adi_noisy)
    avant_contrast = 5.0 * avant_contrast
    apres_contrast = 5.0 * apres_contrast
    avant_separation = avant_separation * pixel_scale
    apres_separation = apres_separation * pixel_scale
    contrast_difference_percent = 100.0 * (apres_contrast - avant_contrast) / avant_contrast

    avant_contrast_output = output_dir / f"avant_contrast_curve{frame_suffix}.fits"
    apres_contrast_output = output_dir / f"apres_contrast_curve{frame_suffix}.fits"
    save_contrast_curve(
        avant_contrast_output,
        avant_separation,
        avant_contrast,
    )
    save_contrast_curve(
        apres_contrast_output,
        apres_separation,
        apres_contrast,
    )
    print(f"### AVANT contrast curve saved: {avant_contrast_output}")
    print(f"### APRES contrast curve saved: {apres_contrast_output}")

    ##### plotting    

    fig, axes = plt.subplot_mosaic(
        [
            ["avant", "apres"],
            ["contrast", "contrast"],
            ["difference", "difference"],
        ],
        figsize=(11, 9),
        constrained_layout=True,
        height_ratios=[1.2, 1.0, 0.8],
    )
    avant_limits = np.nanpercentile(avant_adi_sum, [1, 99.8])
    fig.suptitle("Convolution: circular aperture of radius lambda / (2 D)" if apply_convolution else "Without convolution")
    apres_limits = np.nanpercentile(apres_adi_noisy, [1, 99.8])
    display_vmin = min(avant_limits[0], apres_limits[0])
    display_vmax = max(avant_limits[1], apres_limits[1])
    image_avant = axes["avant"].imshow(
        avant_adi_sum,
        origin="lower",
        vmin=display_vmin,
        vmax=display_vmax,
    )
    axes["avant"].set_title("ADI(cube + noise)")
    avant_cax = make_axes_locatable(axes["avant"]).append_axes(
        "right", size="4%", pad=0.05
    )
    fig.colorbar(image_avant, cax=avant_cax, label="Contrast")

    image_apres = axes["apres"].imshow(
        apres_adi_noisy,
        origin="lower",
        vmin=display_vmin,
        vmax=display_vmax,
    )
    axes["apres"].set_title("ADI(cube) + noise")
    apres_cax = make_axes_locatable(axes["apres"]).append_axes(
        "right", size="4%", pad=0.05
    )
    fig.colorbar(image_apres, cax=apres_cax, label="Contrast")

    for axis in (axes["avant"], axes["apres"]):
        axis.set_xlabel("X (pixels)")
        axis.set_ylabel("Y (pixels)")

    contrast_axis = axes["contrast"]
    contrast_axis.plot(
        avant_separation,
        avant_contrast,
        color="#a24814",
        linewidth=1.8,
        label="ADI(cube + noise)",
    )
    contrast_axis.plot(
        apres_separation,
        apres_contrast,
        color="#2563a6",
        linewidth=1.8,
        label="ADI(cube) + noise",
    )
    contrast_axis.set_title("5-sigma contrast curves")
    contrast_axis.set_xlabel("Angular separation (mas)")
    contrast_axis.set_ylabel("Contrast, 5-sigma")
    contrast_axis.set_yscale("log")
    contrast_axis.set_xlim(left=0)
    contrast_axis.grid(color=".9")
    contrast_axis.legend()

    difference_axis = axes["difference"]
    difference_axis.plot(
        apres_separation,
        contrast_difference_percent,
        color="#6f42a6",
        linewidth=1.8,
        label="Relative difference: after vs before",
        
    )
    difference_axis.axhline(0.0, color=".35", linewidth=0.9, linestyle="--")
    difference_axis.set_title("Relative contrast difference (after - before / before)")
    difference_axis.set_xlabel("Angular separation (mas)")
    difference_axis.set_ylabel("Relative difference (%)")
    difference_axis.set_xlim(left=0)
    difference_axis.set_ylim(-30, 30)
    difference_axis.grid(color=".9")
    difference_axis.legend()

    plot_output = output_dir / f"before_after_adi_comparison{frame_suffix}.png"
    fig.savefig(plot_output, dpi=300, bbox_inches="tight")
    print(f"### Comparison plot saved: {plot_output}")
    plt.show()


    print("\n### DONE")
    print(f"AVANT elapsed: {elapsed_avant:.2f} s -> {avant_output}")
    print(f"APRES elapsed: {elapsed_apres:.2f} s -> {apres_output}")
    print(f"AVANT CONTRAST -> {avant_contrast_output}")
    print(f"APRES CONTRAST -> {apres_contrast_output}")
    print(f"PLOT -> {plot_output}")

    return {
        "apply_convolution": apply_convolution,
        "normalization_psf": normalization_psf,
        "normalization_psf_output": psf_output,
        "convolved_psf": normalization_psf if apply_convolution else None,
        "convolved_psf_output": psf_output if apply_convolution else None,
        "avant_adi": avant_adi_sum,
        "apres_adi": apres_adi_noisy,
        "avant_output": avant_output,
        "apres_output": apres_output,
        "plot_output": plot_output,
        "avant_contrast": avant_contrast,
        "avant_separation_mas": avant_separation,
        "avant_contrast_output": avant_contrast_output,
        "apres_contrast": apres_contrast,
        "apres_separation_mas": apres_separation,
        "apres_contrast_output": apres_contrast_output,
        "contrast_difference_percent": contrast_difference_percent,
        "elapsed_avant": elapsed_avant,
        "elapsed_apres": elapsed_apres,
    }


if __name__ == "__main__":
    # User parameters: update these values before running the comparison.
    test_dir = r"D:\resampled_data\No_Planet\CLC1\NoNCPA\Q1\filter1.190\Samp1.5"

    # Use None to load the parallactic-angle table from test_dir automatically.
    parangle_file = r"C:\Users\tdeseine\Desktop\MISTHIC\output\adi_before_after_test\paralactic_angle\z=40\Parangle_tab.txt"
    output_dir = Path("output") / "adi_before_after_test"

    star_mag = 12
    sig_ron = 15.0
    n_frames = None  # Use None to process every available frame.
    zenith_distance = 40
    tel_diameter = 38.542
    apply_convolution = True

    run_before_after_adi_test(
        test_dir,
        output_dir=output_dir,
        star_mag=star_mag,
        sig_ron=sig_ron,
        n_frames=n_frames,
        parangle_file=parangle_file,
        zenith_distance=zenith_distance,
        tel_diameter=tel_diameter,
        apply_convolution=apply_convolution,
    )
