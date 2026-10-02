# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits
from scipy import interpolate
import matplotlib.pyplot as plt


def get_parallactic_angle(ha_hours, dec_deg, latitude_deg):
    """
    This function has been adapted from the IDL PARANGLE function.
    Returns the parallactic angle for a given hour angle (in hours),
    declination (in degrees) and latitude (in degrees).
    """
    ha_rad       = ha_hours / 24. * 2.*np.pi
    dec_rad      = dec_deg  * np.pi / 180.
    latitude_rad = latitude_deg * np.pi / 180.

    parangle  = np.arctan2(-np.sin(ha_rad), np.cos(dec_rad)*np.tan(latitude_rad)-np.sin(dec_rad)*np.cos(ha_rad))
    parangle *= 180. / np.pi

    return parangle

def get_zenith_distance(ha_hours, dec_deg, latitude_deg):
    """
    Returns the zenith distance for a given hour angle (in hours),
    declination (in degress) and latitude (in degress).
    """

    ha_rad       = ha_hours / 24. * 2.*np.pi
    dec_rad      = dec_deg  * np.pi / 180.
    latitude_rad = latitude_deg * np.pi / 180.

    zendist = np.arccos(np.sin(latitude_rad)*np.sin(dec_rad) +
                        np.cos(latitude_rad)*np.cos(dec_rad)*np.cos(ha_rad))
    zendist *= 180./np.pi

    return zendist
    
def get_air_index(wavelength, pressure=537., temperature=10., rel_humidity=20.):
    """
    This routine is an adaptation from the IDL airindex function by Marc W. Buie, STSci, 2/28/91.

    ; This function is based on the formulas in Filippenko, PASP, v. 94,
    ; pp. 715-721 for the index of refraction of air.  The conversion from
    ; relative humidity to vapor pressure is from the Handbook of Chemistry
    ; and Physics.

    """
    n = np.float64(64.328) + 29498.1/(146.0 - (1.0/wavelength)**2) + 255.4/(41.0 - (1.0/wavelength)**2)
#    print(n)
    pfac = np.float64(pressure) * (1.0+(1.049-0.0157*temperature)*1.0e-6*pressure)/(720.883*(1.0 + 0.003661*temperature))
#    print(pfac)
    dt = 100.0 - temperature
#    print(dt)
    logp = 2.8808 - 5.67*dt/(274.1+temperature-0.15*dt)
#    print(logp)
    f = rel_humidity * logp**10.0
#    print(f)
    water = (np.float64(0.0624) - 0.000680/wavelength**2)*f/(1.0 + 0.003661*temperature)
#    print(water)
    n = ( n - water ) * pfac
#    print(n)
    n = (1.0 + n * np.float64(1.0e-6))
#    print(n)
    return n

def scale_to_photons(img_cube, perf_psf, photon_flux, emission_flux, 
                     frame_exp_time, sig_ron = 15., no_noise=False,
                     set_ron_equivalent=True, silent=True):
    """
    Convert the image levels into photons and include the photon noise and read out noise.
    """
    if len(img_cube.shape) == 2:
        img_cube = img_cube[np.newaxis,:,:]
        nocube = True
    else :
        nocube = False
        
    n_img, n_y, n_x = img_cube.shape
    # obs_time = frame_exp_time * n_img
    ones_array = np.ones_like(img_cube)
    
    ### convert into mean photon numbers
    perf_psf_sum = np.sum(perf_psf)
    conv_factor = photon_flux / perf_psf_sum # convert from ADU to photons !         

    # psf_photon_flux = np.sum(psf_cube) * conv_factor
    # img_photon_flux = np.sum(coro_cube)* conv_factor
    
    if len(img_cube.shape) == 2:
        img_cube = img_cube[np.newaxis,:,:]
        
    # if len(img_cube.shape) == 2:
    #     mean_psf = np.sum(img_cube)
    #     flux_per_frame = photon_flux/mean_psf
    # elif len(img_cube.shape) == 3:
    #     mean_psf         = np.sum(img_cube, axis=(1,2))
    #     # flux_per_frame = np.expand_dims(photon_flux/mean_psf, (1,2))
    
    img_cube_sum = np.sum(img_cube, axis=(1,2)).mean()
    
    # print('conv_factor = ', conv_factor)
    img_cube = np.abs(img_cube) * conv_factor
    # psf_cube = np.abs(psf_cube) * conv_factor
    perf_psf = np.abs(perf_psf) * conv_factor

    flux_per_frame = np.sum(img_cube, axis=(1,2)).mean()
    
    if not silent:
        print(f'\n\n### obsparams.scale_to_photon'+
              f'\n\tperf_psf_sum=  \t{perf_psf_sum:.2e}'+
              f'\n\timg_cube_sum=  \t{img_cube_sum:.2e}'+
              f'\n\tphoton_flux=   \t{photon_flux:.2e}'+
              f'\n\tflux_per_frame=\t{flux_per_frame:.2e}')
    
    if no_noise is False: ### MAKE NOISE
        rng = np.random.default_rng()
        np.random.seed(198717161)
        
        # Apply photon noise 
        # coro_cube_noisy = np.zeros_like(coro_cube)
        if not silent:
            print(f"### Total Flux (within scale_to_photon func): {photon_flux:.3e}")
        img_cube_noisy = rng.poisson(img_cube)
        # psf_cube_noisy  = rng.poisson(psf_cube)
        perf_psf_noisy  = rng.poisson(perf_psf)

        # Readout noise (assuming exp 1.3s<time<30s)
        FWC = 5e4 # Full Well Capacity of the detector
        # cube_coronog = np.zeros((len(frames),x_dim,y_dim))
        if sig_ron != 0:
            if set_ron_equivalent is True:
                # Mean coronographic img
                mean_coro = np.mean(img_cube_noisy,axis=0)
                exp_time = FWC / (np.max(mean_coro) / frame_exp_time)
                
                if exp_time <= 1.3 or exp_time >= frame_exp_time:
                    if not silent :
                        print("### Warning! Exposure time outside desired range ! ): {}".format(exp_time))
                    if exp_time >= frame_exp_time:
                        exp_time = frame_exp_time
                if not silent:
                    print("### Exposure time:", exp_time)
                
                ron_noise = sig_ron*np.sqrt(frame_exp_time / exp_time)
                
                # val_exp = np.zeros(2)
                # val_exp[0] = FWC / (np.max(mean_coro) / frame_exp_time)
                # val_exp[1] = FWC / (np.max(mean_psf) / frame_exp_time)
            else :
                if not silent:
                    print('### No scaling of the RON for optimized exp time.')
                ron_noise = sig_ron
            img_cube_noisy = img_cube_noisy + rng.standard_normal(size=(n_img,n_y,n_x)) * ron_noise    # RON is a gaussian noise 
            # psf_cube_noisy  = psf_cube_noisy + rng.standard_normal(size=(n_img,n_y,n_x)) * noise_eq
            perf_psf_noisy  = perf_psf_noisy + rng.standard_normal(size=(n_y,n_x)) * ron_noise
        
        # Emission noise ---------------------------------------
        if emission_flux != 0:
            img_cube_noisy = img_cube_noisy + rng.poisson(ones_array * emission_flux * frame_exp_time) # also poisson noise
            # psf_cube_noisy = psf_cube_noisy + rng.poisson(ones_array * emission_flux * frame_exp_time)
            perf_psf_noisy = perf_psf_noisy + rng.poisson(ones_array[0] * emission_flux * frame_exp_time)
    else:
        img_cube_noisy = img_cube + ones_array * emission_flux * frame_exp_time
        # psf_cube_noisy = psf_cube + ones_array * emission_flux * frame_exp_time
        perf_psf_noisy = perf_psf + ones_array[0] * emission_flux * frame_exp_time

    if nocube:
        img_cube_noisy = img_cube_noisy[0]

    return img_cube_noisy, perf_psf_noisy, flux_per_frame

def get_aperture_surface(aperture_filename, diam_pix=1015, diam_m = 38.542):
    
    aperture = fits.getdata(aperture_filename)
    pixel_surface = (diam_m/diam_pix)**2
    aperture_surface = np.sum(aperture)*pixel_surface
    
    return aperture_surface

def get_micado_flux(flux_dir, spectrum, filter_type, obs_time, aperture_surface_m2, 
                    airmass=1, pixel_scale=4, tel_diam=38.542, display_plot=False):
    '''
    Return the photon flux for a given target and instrument configuration.
    Inputs:
        directory   : Input directory containing transmission files
        mag         : Magnitude of the star at wavelength lbd (can be a vector of 
                      magnitude if lbd is the vector of the wavelength corresponding
                      to these magnitude) OR planet flux from EXOREM spectra 
                      (photons*m^-2*um^-1*s^-1); accessible if planet=True
        lbd         : Wavelength for the magnitude asked (value of vector, see above)
                      Example: mag=[magV,magJ,magH,magK] and lbd=[0.55,1.25,1.65,2.2]
                      For a given star with a specific spectral type.
                      For a flux in Vega mag: magV=magJ=magH=magK
        filter_type : Filter name; string of format '1.190' (for example)
        obs_time    : Exposure time in s
        airmass     : Airmass; must be between 1 and 3. Set to 1 by default
        pixel_scale    : Pixel scale in mas/px; either 1.5 or 4 here. Set to 4 by default
        planet      : If True, calculate planet flux. If False, calulate star flux.
    ---------------------------------------------------------------------------
    Outputs:
        flux_psf_final : Output flux in Lyot stop (with 5 missing segments)
        emission_tot   : Output total emission per frame (keyword)
        trans_out      : Optional output of the total transmission for the chosen filter

    Authors: Helen (Nellie) Baran, Pierre Baudoz
    Revision: Elsa Huby
    '''
    # airmass is set to 1 by default
    # pixel_scale is set to 4 mas by default
    
    tr_filter0  = fits.getdata(flux_dir+'Transmission_filter_J1J2H1H2K1K2_v2020.fits')
    tr_filter1  = fits.getdata(flux_dir+'Transmission_filter_JHK.fits')
    tr_atmo0    = fits.getdata(flux_dir+'Transmission_atmo_airmass=1_to_3.fits')
    x_airmass   = fits.getdata(flux_dir+'Airmass=1_to_3_for_trans_atmo.fits')
    tr_inst_tel = fits.getdata(flux_dir+'Transmission_tel+instr_noatmo_no_filter.fits')
    wave_tr     = fits.getdata(flux_dir+'Wavelength_for_emission_transmission_in_micrometer.fits')
    emission    = fits.getdata(flux_dir+'Emission_sky+tel+instr_phot_per_s_per_micrometer_in_lsurdxlsurd.fits')
    
    wave_step   = np.mean(np.diff(wave_tr))
    
    ### Select Filter:
    def select_filter(filter_type):
        switcher = {
            '1.190':tr_filter0[0,:],
            '1.270':tr_filter0[1,:],
            '1.582':tr_filter0[2,:],
            '1.693':tr_filter0[3,:],
            '2.100':tr_filter0[4,:],
            '2.220':tr_filter0[5,:], #update Elsa 22.09.16
            '1.245':tr_filter1[0,:],
            '1.635':tr_filter1[1,:],
            '2.145':tr_filter1[2,:],
            '0':np.ones_like(tr_filter1[2,:])
            } 
        return switcher.get(filter_type, lambda: "No filter defined")
    
    tr_filter = select_filter(filter_type)
    
    # # Telescope surface  
    # # pup_dir       = '/Users/pierre/Desktop/MICADO/Simul_MICADO/Simul_2022/Python/INPUT/PUPIL/'
    # pup_dir       = '/home/ehuby/WORK/SIMU/SIMU_MICADO/INPUT/PUPIL/' 
    # HDU_pup       = fits.open(pup_dir+'Pupil_ELT_v02_5missing_segments_v1.fits')
    # pup_lyot      = HDU_pup[0].data
    # surf_pixel    = (3.8e-2)**2  # surface pixel in m^2
    # surf_tel_lyot = np.sum(pup_lyot*surf_pixel) # equivalent surface transmitted by pupil in m^2

    flux = spectrum * aperture_surface_m2
    # test = flux.copy()

    # Calculating transmission of the atmosphere via position index of minimum of x_airmass-airmass
    # airmass = 1 / np.cos(zenith_distance)
    # diff = np.abs(np.array(x_airmass-airmass))
    # index_min = diff.index(np.min(diff))
    index_min = np.argmin(np.abs(x_airmass-airmass))
    tr_atmo = tr_atmo0[index_min,:]
    # print('airmass index in get_micado_flux ',index_min)
    
    # Total emission in the filter
    l_over_d = wave_tr*1e-6/tel_diam*180./np.pi*3600.*1000. # mas
    print(tr_filter)
    emission_per_pix = np.sum(emission * tr_filter * (pixel_scale/l_over_d)**2) * wave_step * obs_time # in photons

    # Total flux in the Final Pupil plane w/o coronagraph
    tr_inst = tr_filter * tr_inst_tel
    final_photon_flux = np.sum(tr_inst*tr_atmo*flux) * wave_step * obs_time # in photons

    # Output transmission
    # print(f'wave_step {wave_step}, obs_time {obs_time}, surf_tel_lyot {surf_tel_lyot}, tr_atmo index {index_min}')
    global_transmission = tr_inst * tr_atmo #* obs_time * aperture_surface_m2
    # print("params:", wave_step, obs_time)
    # print("Flux psf final:", flux_psf_final)
    
    ### Optionnal plot
    if display_plot is True:
        plt.figure('get_micado_flux')
        plt.clf()
        plt.plot(wave_tr, tr_inst_tel, ':', label='Instr + tel', alpha=.5)
        plt.plot(wave_tr, tr_atmo, '--', label='Atmospheric', alpha=.5)
        plt.plot(wave_tr, tr_filter, '-.', label=f'Filter ({filter_type}um)', alpha=.5)
        plt.plot(wave_tr, global_transmission, label='Total')
        plt.yscale('log')
        plt.ylim(.08, 1.)
        plt.legend()
        plt.xlabel('Wavelength [um]')
        # plt.ylabel('Transmission')
        plt.title('Transmission')
        plt.grid(color='.95', which='minor')
        plt.grid(color='.85', which='major')
        
    return final_photon_flux, emission_per_pix, [wave_tr, global_transmission]

def get_planet_spectrum(spec_dir, temp, logg, met, CO, dist_pc, r_planet,
                        flux_dir=None):
    """
    Return the planet photon flux spectrum, with the same wavelength sampling
    as the MICADO transmission files.

    Parameters
    ----------
    spec_dir : string
        Directory with planet EXOREM spectra.
    temp : float
        Planet temperature in Kelvin.
    logg : float
        Planet log(g).
    met : float
        Planet metallicity.
    CO : float
        Planet CO.
    r_planet : float
        Planet radius in Jupiter radii
    dist_pc : float
        Planet distance to system in parsecs
    flux_dir : string
        Directory with MICADO transmission and flux files.

    Returns
    -------
    spec_interp : nd.array
        Planet spectrum interpolated to correspond to the wavelength vector
        of the MICADO transmission and flux files.

    """
    spec_file_name  = spec_dir+f'spectra_YGP_{temp}K_logg{logg}_met{met:.2f}_CO{CO:3.2f}.dat'
    spec_file       = np.loadtxt(spec_file_name, unpack=True)
    spec_flux       = spec_file[1,:] # W*m^-2 / cm^-1
    spec_wave_num   = spec_file[0,:] # cm^-1
    # k_step = np.mean(np.diff(spec_wave_num))
    spec_wave       = 10000. / spec_wave_num  # um
    lam_spec        = spec_flux * spec_wave_num**2 / 10000
    
    if flux_dir is not None:
        wave_tr         = fits.getdata(flux_dir+'Wavelength_for_emission_transmission_in_micrometer.fits')
        # wave_step = np.mean(np.diff(wave_tr))

        f = interpolate.interp1d(spec_wave,lam_spec, kind='linear',fill_value="extrapolate")
        spec_interp = f(wave_tr)
        spec_wave = wave_tr
    else:
        spec_interp = lam_spec
        
    spec_interp[spec_interp < 0] = 0
    
    # flux conversion into photons
    h = 6.62607015*10e-34       # J*s
    c = 299792458               # m/s
    lbd_in_m = spec_wave*10e-6    # m
    photon_per_joule = lbd_in_m / h / c
    dist_m = dist_pc * 3.086e+16 # convert to meters from parsecs
    r_planet_m = r_planet * 7.149e+7 #convert to meters from r_jup
    
    planet_flux = spec_interp * photon_per_joule * (r_planet_m**2) / (dist_m**2)
    
    return planet_flux, spec_wave

def get_star_spectrum(flux_dir, star_mag, lbd, display_plot=False):
    """
    Return the flux for a star as a function of the wavelength vector in the 
    MICADO transmission and flux files. If several magnitude values are 
    given for different wavelength, the magnitude is interpolated to match the
    wavelength array.

    Parameters
    ----------
    flux_dir : string
        DESCRIPTION.
    star_mag : float or list or array
        DESCRIPTION.
    lbd : float or list or array
        DESCRIPTION.

    Returns
    -------
    star_flux : TYPE
        DESCRIPTION.

    """
    
    flux_zero   = fits.getdata(flux_dir+'Zero_mag_in_phot_per_m2_per_um_per_s.fits')
    wave_tr     = fits.getdata(flux_dir+'Wavelength_for_emission_transmission_in_micrometer.fits')
    
    if isinstance(star_mag, list):
        # flux = flux_zero * 10**(-interpol(mag,lbd,wave_tr)/2.5) * surf_tel_lyot 3 IDL
        f = interpolate.interp1d(lbd, star_mag, kind='linear')
        mag_interp = f(wave_tr)
        star_flux = flux_zero * 10**(-mag_interp/2.5)
    else:
        # star_flux = np.zeros((len(star_mag), len(flux_zero)))
        # for i in range(len(mag)):
        #     star_flux[i] = flux_zero * 10**(-mag[i]/2.5)
        star_flux = flux_zero * 10**(-star_mag/2.5)
    
    if display_plot is True:
        plt.figure('get_star_spectrum')
        plt.clf()
        plt.plot(wave_tr, star_flux)
        plt.title('Star photon flux')
    
    return wave_tr, star_flux