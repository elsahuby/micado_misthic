
from scipy.signal import fftconvolve
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path
from micado_misthic.imgproc import micado_adi


def convolve_adi_image(image, mask):
    """Convolve and crop using the centring convention of comb_sim_eh_adapted."""
    convolved = fftconvolve(image, mask, mode="full")
    height, width = image.shape
    y_start = (convolved.shape[0] - height + 1) // 2
    x_start = (convolved.shape[1] - width + 1) // 2
    return convolved[y_start:y_start + height, x_start:x_start + width]


def find_simulation_fits(path):
    """Find all input FITS products in one directory scan."""
    files = {file.name.lower(): file for file in sorted(Path(path).glob("*"))
             if file.is_file() and file.suffix.lower() == ".fits"}
    result = dict.fromkeys(("image_cube", "psf_cube", "perfect_psf", "adi_sum", "psf_sum", "mean"))
    for name, file in files.items():
        if "image_cube" in name and not name.endswith("_mean.fits"):
            result["image_cube"] = file
        elif "psf_cube" in name:
            result["psf_cube"] = file
        elif "perfect_psf" in name:
            result["perfect_psf"] = file

    for name, file in files.items():
        if name.endswith("adi_sum.fits"):
            prefix = name[:-len("adi_sum.fits")]
            psf = files.get(prefix + "psf_sum.fits")
            if psf is not None:
                result["adi_sum"] = file
                result["psf_sum"] = psf
                break
    if result["adi_sum"] is None:
        result["adi_sum"] = files.get("adi_sum.fits")

    adi = result["adi_sum"]
    prefix = adi.name.lower()[:-len("adi_sum.fits")] if adi else ""
    if prefix:
        result["mean"] = files.get(prefix + "image_cube_mean.fits")
    else:
        means = [file for name, file in files.items() if name.endswith("image_cube_mean.fits")]
        if len(means) == 1:
            result["mean"] = means[0]
    return result


def save_contrast_curve(output_path, separation_mas, contrast,
                        planet_distance_mas=None, planet_contrast=None):
    """Write a 5-sigma contrast FITS table with optional planet metadata."""
    columns = [
        fits.Column(name="separation_mas", array=np.asarray(separation_mas, dtype=np.float64), format="D"),
        fits.Column(name="contrast_5sigma", array=np.asarray(contrast, dtype=np.float64), format="D"),
    ]
    table = fits.BinTableHDU.from_columns(columns)
    if planet_distance_mas is not None and planet_contrast is not None:
        table.header["PLDIST"] = (float(planet_distance_mas), "Planet distance in mas")
        table.header["PLCONT"] = (float(planet_contrast), "Planet contrast")
    table.writeto(output_path, overwrite=True)


def normalize_by_psf_max(img, psf):
    """Normalize an image by the PSF maximum; leave it unchanged if zero."""
    max_psf = np.nanmax(psf)
    if max_psf == 0:
        return img
    return np.asarray(img, dtype=np.float64) / max_psf


def get_planet_flux_with_deltaMag(star_flux, star_mag, delta_Mag):

    """
    Return the flux for a planet as a function of the wavelength vector in the 
    MICADO transmission and flux files. If several magnitude values are 
    given for different wavelength, the magnitude is interpolated to match the
    wavelength array. The magnitude used is the star magnitude + the delta magnitude to fix
    a magnitude for the planet.

    Parameters
    ----------
    star_flux : from get_star_spectrum
       
    star_mag : float
       
    delta_Mag : float
       
    lbd : float

    Returns
    -------
    planet_flux 
       
    """     
    planet_mag = delta_Mag+star_mag
    planet_flux = star_flux * 10 ** (- (planet_mag-star_mag)/2.5)

    return planet_flux


def reconstruct_coronographic_image(
        star_adi_path,
        planet_adi_path,
        save_path=None,
        display=False):
    """Sum star and planet ADI images supplied as arrays or FITS paths."""

    star_image = fits.getdata(star_adi_path) if isinstance(star_adi_path, (str, Path)) else np.asarray(star_adi_path)
    planet_image = fits.getdata(planet_adi_path) if isinstance(planet_adi_path, (str, Path)) else np.asarray(planet_adi_path)
    reconstructed = star_image + planet_image

    if save_path is not None:
        fits.writeto(save_path, reconstructed.astype(np.float32), overwrite=True)

    if display:
        vmin, vmax = np.nanpercentile(reconstructed, [1, 99.8])
        plt.figure("Reconstructed star + planet")
        plt.imshow(reconstructed, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return reconstructed


def get_planet_contrast(
        planet_adi_sum_fits):
    """Return the peak contrast from a final planet image or its FITS path."""
    data = fits.getdata(planet_adi_sum_fits) if isinstance(planet_adi_sum_fits, (str, Path)) else np.asarray(planet_adi_sum_fits)
    contrast = np.nanmax(data)

    return contrast



def check_fits_NaN(parent_dir=r"E:\2026_ADI"):
    """
    Check if there is any NaN in a fits. Usefull when simulation seems corrupted.
    """

    for fichier in Path(parent_dir).rglob("*.fits"):
        try:
            image = fits.getdata(fichier)
            if np.isnan(image).all():
                print(f"NaN trouvé : {fichier}")
        except Exception as erreur:
            print(f"Erreur avec {fichier} : {erreur}")


########################################################################
########################################################################
#### OPTIMIZED FUNCTION FROM MISTHIC FOR THE PERFORMANCE CALCULATOR ####
########################################################################
########################################################################
  
   
def run_adi_without_noise(cube, psf_cube, parallactic_dir, adi_file=None):
    """Reuse or save the clean ADI sum and retain the data needed for 2D noise."""
    adi_file = Path(adi_file) if adi_file is not None else Path(parallactic_dir) / "adi_sum.fits"
    if adi_file.is_file():
        print(f"### Reusing clean ADI: {adi_file}")
        adi_sum = fits.getdata(adi_file)
    else:
        adi_sum, _ = micado_adi(cube, psf_cube, str(parallactic_dir) + "\\")
        fits.writeto(adi_file, adi_sum, overwrite=False)
    noise_info = {
        "n_frames": cube.shape[0],
        "mean_abs_cube": np.mean(np.abs(cube), axis=0),
    }
    return adi_sum, noise_info


def add_noise_after_adi(
    adi_sum,
    perf_psf,
    noise_info,
    photon_flux,
    sig_ron,
    emission_flux,
    frame_exp_time,
    set_ron_equivalent=True,
    silent=True,
):
    """Add noise using a noise-info dict or the path to an image_cube_mean FITS."""
    if isinstance(noise_info, (str, Path)):
        with fits.open(noise_info) as hdul:
            mean_hdu = hdul["MEAN_ABS"] if "MEAN_ABS" in hdul else hdul[0]
            noise_info = {
                "n_frames": hdul[0].header["NFRAMES"],
                "mean_abs_cube": np.asarray(mean_hdu.data, dtype=np.float64),
            }
    n_frames = noise_info["n_frames"]
    mean_abs_cube = noise_info["mean_abs_cube"]
    if np.shape(mean_abs_cube) != np.shape(adi_sum):
        raise ValueError("Mean image and ADI image must have the same dimensions.")
    if not np.isfinite(n_frames) or n_frames <= 0 or int(n_frames) != n_frames:
        raise ValueError("NFRAMES must be a positive integer.")
    if not np.all(np.isfinite(mean_abs_cube)) or np.any(mean_abs_cube < 0):
        raise ValueError("MEAN_ABS must contain finite, non-negative values.")
    height, width = adi_sum.shape
    perf_psf_sum = np.sum(perf_psf)
    if perf_psf_sum == 0:
        raise ValueError("Cannot scale ADI data: perfect PSF sum is zero.")

    conv_factor = photon_flux / perf_psf_sum
    adi_sum = np.asarray(adi_sum, dtype=np.float64) * conv_factor
    rng = np.random.default_rng()
    np.random.seed(198717161)

    mean_abs_cube_photons = mean_abs_cube * conv_factor
    noise_var = mean_abs_cube_photons * n_frames
    photon_noise = rng.poisson(noise_var) - noise_var

    
    if sig_ron != 0:
        if set_ron_equivalent:
            full_well_capacity = 5e4
            exp_time = full_well_capacity / (
                np.max(mean_abs_cube_photons) / frame_exp_time
            )
            if exp_time <= 1.3 or exp_time >= frame_exp_time:
                if not silent:
                    print(f"### Warning: exposure time outside desired range: {exp_time}")
                if exp_time >= frame_exp_time:
                    exp_time = frame_exp_time
            if not silent:
                print(f"### Exposure time: {exp_time}")
            sig_ron = sig_ron * np.sqrt(n_frames * frame_exp_time / exp_time)
        else:
            if not silent:
                print("### No optimized-exposure scaling of the RON.")
            sig_ron = sig_ron * np.sqrt(n_frames)

    ron_noise = rng.normal(0.0, sig_ron, size=(height, width))

    emission_mean = emission_flux * frame_exp_time * n_frames
    if emission_mean > 0:
        emission_noise = rng.poisson(emission_mean, size=(height, width)) - emission_mean
    else:
        emission_noise = np.zeros((height, width), dtype=np.float64)

    return adi_sum + photon_noise + ron_noise + emission_noise


def psf_sum_in_photons(psf_cube, perf_psf, photon_flux):
    """Return the noiseless PSF cube sum with the photon scaling applied."""
    perf_psf_sum = np.sum(perf_psf)
    if perf_psf_sum == 0:
        raise ValueError("Cannot scale the PSF cube: perfect PSF sum is zero.")

    conv_factor = photon_flux / perf_psf_sum
    return np.sum(np.abs(psf_cube) * conv_factor, axis=0)
