# Documentation of the data preprocessing

This Markdown document aims to explain the preprocessing of simulation data. This preprocessing helps developers of the performance calculator or collaborators prepare new data to be used by it.


**Author:** Tristan Deseine <br>
**Creation date:** 21/09/2026 <br>
**Last update:** 22/09/2026

## Overview of the directory

The preprocessing is stored in the ***simu_preprocessing*** directory, composed of three different python files.

* `file_utils.py`
* `resampling.py`
* `pipeline.py`



The first two files are used in the third one, `pipeline.py` which is the main file to prepocess the data.
`file_utils.py` is regrouping functions which are manipulating files and which are necessary for the resampling for example. This last process is implemanted in `resampling.py`.


## Preprocessing pipeline overview (pipeline.py)

The `pipeline.py` script runs two main processing steps, enabled independently with the `resampling` and `ADI` flags:

1. **Resampling:** adapts the simulation images and PSFs to the requested pixel scales and saves them in the resampled data directory.
2. **ADI (Angular Differential Imaging):** selects the frames for each requested observation duration and applies `micado_adi()` to reduce the stellar contribution and help reveal planets. It saves the ADI result, the associated PSF result and the mean image in a folder named after the effective observation duration.

The input and output paths, pixel scales and observation durations are configured at the beginning of the script. ADI uses the resampled data, which must already exist if the resampling step is disabled.


To enable or disable each processing step, set its corresponding variable (`resampling` or `ADI`) to ***True*** or ***False***, respectively.



## file_utils.py overview

This section will present the methods of `file_utils.py`.

### **find_input_file()**

This function find an input file according to a specific pattern. It can be used for find the .ini config file of a simulation.
<p style="margin-bottom: 0;"><u>Input:</u></p>
<ul style="margin-top: 0;">
  <li><em>directory</em>: Directory where we are searching</li>
  <li><em>pattern</em>: Pattern for the filename to search for (e.g. "*.ini")</li>
</ul>

<p style="margin-bottom: 0;"><u>Output:</u></p>
<ul style="margin-top: 0;">
  <li><em>path</em>: Path to the first file matching the pattern.</li>
</ul>  

### **copy_aux_files()**

This function copies configuration files ending in .ini and files whose names contain "parangle" or "parall" into the output directory. It ignores macOS auxiliary files starting with "._". If the source directory does not exist, it returns without copying anything. Copy failures are printed.
<p style="margin-bottom: 0;"><u>Input:</u></p>
<ul style="margin-top: 0;">
  <li><em>source_dir</em>: Directory containing the auxiliary files to copy.</li>
  <li><em>dest_dir</em>: Existing destination directory where the files should be copied.</li>
</ul>

### **select_observation_time()**

This function selects consecutive frames from an already loaded cube. The number of frames is calculated by rounding the desired time multiplied by 60 and divided by the exposure time. The selection is centred on the frame whose parallactic angle is closest to zero (passage through the meridian), and shifted if necessary to stay within the available frames. If the requested duration exceeds the available data, the whole cube is returned.

<p style="margin-bottom: 0;"><u>Input:</u></p>
<ul style="margin-top: 0;">
  <li><em>desired_time</em>: Requested observation time in minutes.</li>
  <li><em>cube</em>: NumPy array containing the images, with frames along the first axis.</li>
  <li><em>angles</em>: Array of parallactic angles corresponding to the cube frames.</li>
  <li><em>delta_t</em>: Exposure time, in seconds.</li>
</ul>

<p style="margin-bottom: 0;"><u>Output:</u></p>
<ul style="margin-top: 0;">
  <li><em>selected_cube</em>: Slice of the cube containing the selected frames.</li>
  <li><em>selected_angles</em>: Slice of the angle array corresponding to these frames.</li>
</ul>



### **format_observation_time()**

This function converts an effective observation duration into a readable folder label, after rounding to the nearest second. It is used for naming ADI output directories.

<p style="margin-bottom: 0;"><u>Input:</u></p>
<ul style="margin-top: 0;">
  <li><em>duration_seconds</em>: Effective observation duration in seconds.</li>
</ul>

<p style="margin-bottom: 0;"><u>Output:</u></p>
<ul style="margin-top: 0;">
  <li><em>label</em>: String used to name the ADI output folder.</li>
</ul>

### **parse_simulation_metadata()**

This function extracts the simulation parameters from directory names. It accepts both CLC_seeing_ncpa and CLC_ncpa_seeing ordering. The seeing value is converted to uppercase, and the NCPA value is normalised to "NCPA" or "NoNCPA".
<p style="margin-bottom: 0;"><u>Input:</u></p>
<ul style="margin-top: 0;">
  <li><em>folder</em>: Simulation folder.</li>
  <li><em>filter_dir</em>: Directory name containing the wavelength as "Lambda=..." (e.g. "Lambda=2.2").</li>
</ul>

<p style="margin-bottom: 0;"><u>Output:</u></p>
<ul style="margin-top: 0;">
  <li><em>corono</em>: Coronagraph identifier extracted from the first part of the folder name.</li>
  <li><em>seeing</em>: Seeing value in uppercase.</li>
  <li><em>ncpa</em>: "NCPA" or "NoNCPA".</li>
  <li><em>filter_str</em>: Filter name built from the wavelength (e.g. "filter2.2").</li>
  <li><em>wavelength</em>: Wavelength in microns, as a NumPy float64.</li>
</ul>

<span style="color: red;"><strong>Warning:</strong> This function may need to be adapted if directory naming conventions change or if additional simulation parameters must be extracted from directory names.</span>
### **discover_simulations()**

This function recursively searches the input directory for simulation folders containing FITS filenames matching each required name: image_cube, perfect_psf and psf_cube. It ignores files starting with "._" and skips incomplete simulations or simulations with invalid metadata. Each branch must contain a Lambda=... directory, and its first directory below the input directory must describe the coronagraph, seeing and NCPA parameters. Optional z=... and p_dist_mas=... directories can appear at any position along the branch.
<p style="margin-bottom: 0;"><u>Input:</u></p>
<ul style="margin-top: 0;">
  <li><em>input_directory</em>: Root directory containing the simulation branches.</li>
  <li><em>planet_on</em>: Select simulations with a planet (1) or without a planet (0). Simulations with a planet are identified by a p_dist_mas=... directory in their path.</li>
</ul>

<p style="margin-bottom: 0;"><u>Output:</u></p>
<ul style="margin-top: 0;">
  <li><em>simulations</em>: List of dictionaries, one per matching simulation. Each dictionary contains source_dir, corono, seeing, ncpa, z, filter, wavelength and distance. The z and distance values retain their directory names, or are None when absent. The list is empty if no simulation matches.</li>
</ul>

<span style="color: red;"><strong>Warning:</strong> This function may need to be adapted if you add new simulation parameters to extract, change directory naming conventions, or introduce an unsupported directory structure.</span>

### **get_initial_sampling_factor()**

This function uses `find_input_file()` to locate the single .ini configuration file in a simulation directory. It reads detector_sampling from the [detectorconfig] section and converts it to a float.
<p style="margin-bottom: 0;"><u>Input:</u></p>
<ul style="margin-top: 0;">
  <li><em>simulation_dir</em>: Directory containing the single simulation .ini configuration file.</li>
</ul>

<p style="margin-bottom: 0;"><u>Output:</u></p>
<ul style="margin-top: 0;">
  <li><em>detector_sampling</em>: Initial detector sampling factor, as a float.</li>
</ul>


### **matches_selection()**

This function checks whether a simulation matches the requested coronagraph, NCPA and seeing values. Comparisons are case-insensitive. A selection set to None is ignored, so that parameter does not restrict the selection. This function is used when we don't want to resample all the simulations.
<p style="margin-bottom: 0;"><u>Input:</u></p>
<ul style="margin-top: 0;">
  <li><em>simulation</em>: Dictionary containing the simulation metadata.</li>
  <li><em>corono</em>: Requested coronagraph identifier, or None to accept any coronagraph.</li>
  <li><em>ncpa</em>: Requested NCPA value, or None to accept either value.</li>
  <li><em>seeing</em>: Requested seeing value, or None to accept any seeing.</li>
</ul>

<p style="margin-bottom: 0;"><u>Output:</u></p>
<ul style="margin-top: 0;">
  <li>A boolean: True if all specified criteria match, otherwise False.</li>
</ul>

### **get_output_directory()**

This function builds the output directory path from the simulation metadata. The path follows the order: output_main_directory / With_Planet or No_Planet / corono / ncpa / seeing / z / filter / distance. The z, filter and distance components are omitted when their values are None.
<p style="margin-bottom: 0;"><u>Input:</u></p>
<ul style="margin-top: 0;">
  <li><em>output_main_directory</em>: Root directory for the preprocessed data.</li>
  <li><em>simulation</em>: Simulation dictionary containing corono, ncpa, seeing, z, filter and distance.</li>
  <li><em>planet_on</em>: Set to 1 to use the With_Planet directory, or 0 to use No_Planet.</li>
</ul>

<p style="margin-bottom: 0;"><u>Output:</u></p>
<ul style="margin-top: 0;">
  <li><em>output_directory</em>: Constructed output path as a string. The function does not create the directory.</li>
</ul>

<span style="color: red;"><strong>Warning:</strong> This function may need to be adapted if you add new simulation parameters to extract, or change the order of the tree configuration.</span>

## resampling.py overview

This file contains the `resampling()` function, which prepares simulation images at the requested spatial sampling. It uses the functions from `file_utils.py` to discover simulations, select them and organise the output directories, and `fft_resize()` from micado_misthic.imgproc to resize the images.

### **resampling()**

This function first builds the list of available simulations with `discover_simulations()`, then processes those matching the requested coronagraph, NCPA and seeing values. For each selected simulation, it reads the initial detector sampling from the .ini configuration file and resamples the image_cube, perfect_psf and psf_cube FITS products for each requested sampling value. Other FITS products and macOS auxiliary files starting with "._" are ignored.

The function checks each requested sampling value against the wavelength and the telescope diameter D, set to 38.542 metres. A value is skipped when twice the requested pixel scale is greater than the angular scale lambda/D, converted to milliarcseconds.

<p style="margin-bottom: 0;"><u>Input:</u></p>
<ul style="margin-top: 0;">
  <li><em>output_main_directory</em>: Root directory where the resampled simulations will be saved.</li>
  <li><em>input_directory</em>: Root directory containing the input simulation branches.</li>
  <li><em>sampling_factor</em>: List of target pixel scales in milliarcseconds per pixel ([1.5, 4.0]).</li>
  <li><em>planet_on</em>: Select simulations with a planet (1) or without a planet (0). Simulations with a planet are identified by a p_dist_mas=... directory in their path. This parameter also determines whether outputs are stored under With_Planet or No_Planet.</li>
  <li><em>corono</em>: Requested coronagraph identifier (e.g. "CLC1"), or None to accept any coronagraph. (OPTIONAL)</li>
  <li><em>ncpa</em>: Requested NCPA value ("NCPA" or "NoNCPA"), or None to accept either value. (OPTIONAL)</li>
  <li><em>seeing</em>: Requested seeing value, or None to accept any seeing. (OPTIONAL)</li>
</ul>


Existing output FITS files are skipped. Simulations with a missing or invalid detector_sampling value are skipped, as are FITS files that cannot be read or contain no data.
