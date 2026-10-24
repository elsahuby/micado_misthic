"""
Author: Tristan Deseine
"""

import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from astropy.io import fits
from configobj import ConfigObj
from micado_misthic.imgproc import micado_adi
from micado_misthic.perfCalculator.simu_preprocessing.file_utils import (
    copy_aux_files,
    find_input_file,
    format_observation_time,
    select_observation_time,
)

from micado_misthic.perfCalculator.simu_preprocessing.resampling import resampling as run_resampling


# Select the processing steps.
resampling = False
ADI = True

simulation_data_dir = r'/Volumes/MICADO_Back/2026/Forgot'#/Volumes/MICADO_Back/Test_atraiter'#/Volumes/MICADO_Back/2026/CLC50_noNCPA_5Hz_Med/z=20'
resampled_output_dir = r'/Volumes/TristanMIC2/Resampled2026_2'#/Volumes/TristanMIC2/Resampled_Planet'#/Volumes/MICADO_Back/TMP'  
ADI_output_dir = r'/Volumes/TristanMIC2/ADI_oct26_all'
planet_on = 1
sampling_in_mas = [1.5, 4.0]

planet_directory = 'Planet' if planet_on == 1 else 'CoroStar'
ADI_input_path = Path(resampled_output_dir) / planet_directory
ADI_output_path = Path(ADI_output_dir) / planet_directory
OBSERVATION_TIMES = (15, 30, 60, 90, 120)  #in minutes


def main():
    if resampling:
        run_resampling(
            output_main_directory=resampled_output_dir,
            input_directory=simulation_data_dir,
            sampling_in_mas=sampling_in_mas,
            planet_on=planet_on,
            corono=None,
            ncpa=None,
            seeing=None,
        )

    if ADI:
        if not ADI_input_path.is_dir():
            raise FileNotFoundError(f"Dossier d'entree ADI introuvable : {ADI_input_path}")
        cube_paths = sorted(
            path for path in ADI_input_path.rglob("*_image_cube.fits")
            if not path.name.startswith("._")
        )
        if not cube_paths:
            raise FileNotFoundError(f"Aucun fichier *_image_cube.fits dans : {ADI_input_path}")
        print(f"ADI : {len(cube_paths)} cubes trouves dans {ADI_input_path}")
        for cube_path in cube_paths:
            psf_path = cube_path.with_name(cube_path.name.replace("_image_cube", "_psf_cube"))
            parangle_path = find_input_file(cube_path.parent, "*_parangle_tab.txt")
            prefix = cube_path.name.removesuffix("_image_cube.fits")
            perfect_psf_path = cube_path.with_name(f"{prefix}_perfect_psf.fits")
            config_path = find_input_file(cube_path.parent, "*.ini")
            delta_t = float(ConfigObj(str(config_path))["simuconfig"]["delta_t"])
            full_cube = np.asarray(fits.getdata(cube_path))
            full_angles = np.atleast_1d(np.loadtxt(parangle_path, skiprows=1))
            full_psf_cube = None

            for minutes in OBSERVATION_TIMES:
                cube, angles,first_frame = select_observation_time(minutes, full_cube, full_angles, delta_t)

                effective_seconds = len(cube) * delta_t
                duration_label = format_observation_time(effective_seconds)
                if effective_seconds < minutes * 60:
                    print(
                        f"Duree demandee: {minutes} min; duree disponible: "
                        f"{effective_seconds / 60:g} min. Sortie renommee {duration_label}."
                    )

                output_dir = ADI_output_path / cube_path.parent.relative_to(ADI_input_path) / duration_label
                adi_output_path = output_dir / f"{prefix}_ADI_sum.fits"
                psf_output_path = output_dir / f"{prefix}_PSF_sum.fits"
                mean_output_path = output_dir / f"{prefix}_image_cube_mean.fits"

                output_dir.mkdir(parents=True, exist_ok=True)
                perfect_psf = np.asarray(fits.getdata(perfect_psf_path))
                max_perf_psf= np.max(perfect_psf)
                sum_perf_psf= np.sum(perfect_psf)

                if all(path.is_file() for path in (adi_output_path, psf_output_path, mean_output_path)):
                    print(f"ADI, PSF et moyenne {duration_label} deja crees, traitement ignore : {output_dir}")
                    continue

                copy_aux_files(cube_path.parent, output_dir)
                # Replace the copied angle table with the selected observation angles.
                np.savetxt(output_dir / parangle_path.name, angles, header="a")

                if not mean_output_path.is_file():
                    header = fits.Header()
                    header["NFRAMES"] = (len(cube), "number of frames in the cube")
                    header["FRAME0"] = (first_frame, "first frame")
                    fits.writeto(mean_output_path, (np.mean(np.abs(cube), axis=0)/max_perf_psf).astype(np.float32),
                                 header=header)
                    print(f"Moyenne enregistree : {mean_output_path}")

                if not adi_output_path.is_file() or not psf_output_path.is_file():
                    if full_psf_cube is None:
                        full_psf_cube = np.asarray(fits.getdata(psf_path))/max_perf_psf
                    psf_cube, _, _ = select_observation_time(minutes, full_psf_cube, full_angles, delta_t)
                    with TemporaryDirectory() as parangle_dir:
                        np.savetxt(Path(parangle_dir) / "selected_parangle_tab.txt", angles, header="a")
                        adi_sum, psf_sum = micado_adi(cube, psf_cube, parangle_dir + os.sep)
                    for path, data in ((adi_output_path, adi_sum), (psf_output_path, psf_sum)):
                        if not path.is_file():
                            header = fits.Header()
                            header["S_PERPSF"] = sum_perf_psf
                            fits.writeto(path, data.astype(np.float32), header=header)
                print(f"Produits {duration_label} completes dans : {output_dir}")

            # Keeping a cropped value of the psf cubes for simulation of planet at distances >80 mas
            if planet_on == 0:
                ADIplanet_output_path = Path(ADI_output_dir) / "Planet"
                outputplanet_dir = ADIplanet_output_path / cube_path.parent.relative_to(ADI_input_path)
                outputplanet_dir.mkdir(parents=True, exist_ok=True)
            
                parts = Path(cube_path.parent).parts
                z = float(next(p for p in parts if p.startswith("z=")).split("=")[1])        # 20.0
                lbd_filter = float(next(p for p in parts if p.startswith("filter")).removeprefix("filter"))  # 1.247
                sampling = float(parts[-1].removeprefix("Samp"))   
                dim_small = round(lbd_filter / 38.542 * 0.2062648 *1000./sampling *20.)*2 # keeping only a field of 20x20 lbd/D for planet simulation
                print(z,lbd_filter,sampling,dim_small) 
                full_psf_cube = np.asarray(fits.getdata(psf_path))/max_perf_psf
                dim_psf = len(full_psf_cube[0])
                small_fov_psf_cube_path = outputplanet_dir / "small_FOV_PSF_cube.fits"
                print(dim_psf,dim_small,small_fov_psf_cube_path)
                small_fov_psf_cube = full_psf_cube[:,dim_psf//2-dim_small//2:dim_psf//2+dim_small//2,dim_psf//2-dim_small//2:dim_psf//2+dim_small//2]
                print(dim_psf,small_fov_psf_cube_path)
                header = fits.Header()
                header["S_PERPSF"] = sum_perf_psf
                fits.writeto(small_fov_psf_cube_path, small_fov_psf_cube.astype(np.float32), header=header)

if __name__ == "__main__":
    main()
