# Performance Calculator Help

This guide describes the main steps required to use the performance calculator.

## Select the simulation data

Click **Choose folder** and select the directory containing the preprocessed simulations and the following resources:

```text
base_folder/
├── No_Planet/
├── With_Planet/
├── Flux_input/
└── PUPIL/
```

The selected instrument and stellar parameters determine the simulation directory displayed below the tabs.
The base folder should absolutely be composed of, at least, the ``"No_Planet"`` folder with simulation in it, the ``"Flux_input"`` folder and the ``"PUPIL"``. If you want to rename those 3 folders, you have to adapt the software, where they are used. Finally, the ``"With_Planet"`` folder is essential if you want to add a planet in the simulation.

**Note 01/10/2026:**  For now, we have two different files corresponding to the 2021 and 2026 simulations. The 2026 file does not contain any planet simulations. They are named ``2021_ADI`` and ``2026_ADI``, and are stored on the 4-terabyte ``TristanMIC2`` hard drive.

## Instrument parameters

Select the coronagraph, NCPA configuration, spatial sampling, filter wavelength and observation time.

The available wavelengths depend on the sampling. A sampling of 4 mas/pixel does not provide the J-band filters.

*These parameters are necessary to operate the interface because they are all part of the simulation names.*

## Stellar parameters

* ***Seeing:** atmospheric seeing used by the simulation.
* **Magnitude:** required when detection noise is enabled.
* **Declination:** value between -90° and 90°. It selects the nearest available zenith distance.
* **Flux folder:** detected automatically inside the selected base folder.



*The seeing is necessary. The magnitude is not part of the simulation names but are essential to compute stellar flux that is then used to compute the noise. The declination is essential but only for 2026 simulations. In fact, this parameter permit to compute a zenital distance, and the 2021 simulations have only one zenithal distance compared to 2026 one that are defined according to two differents zenithal distances (20° and 40°).*

## Planet parameters

Enable **Planet on** to process and reconstruct a simulated planet. Select its angular distance and enter a delta magnitude between 0 and 20.

The delta magnitude is required when the planet is enabled.

## Run the processing

1. Set **ADI** to **Yes**.
2. Choose whether the final products should be saved.
3. Select an output folder when saving is enabled.
4. Click **Run**.

The interface loads the ADI and PSF products created during preprocessing. When noise is enabled, it also requires the matching cube-mean image, perfect PSF and simulation configuration file.

The processed stellar image is displayed in the **ADI Image** tab. When a planet is enabled, the image selector can switch between the stellar result and the reconstructed star-plus-planet result that are include in the preprocessed simulations.

## Contrast curve

Click **Plot contrast curve** after a successful run to calculate the 5-sigma contrast curve. The curve is displayed in its own tab and exported as FITS and PNG files when an output directory is available.

The **Files / Tools** area reports the loaded files, processing steps, saved products and any missing input.

Below the **Files / Tools** area, you can select a feature that allows you, among other things, to analyze the results. For now, there is only one function that allow to give the value of the contrast for a given separation.