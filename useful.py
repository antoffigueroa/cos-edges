#!/usr/bin/env python
#####################################
# Useful functions
#####################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy import constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import vorbin
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from astropy.wcs import WCS
import pyregion
from pyregion.region_to_filter import as_region_filter

# Define useful constants
c = const.c.to('km/s').value


def read_cube(data_path, var_path, ret_hdr=True):
    """
    Reads the data from a fits file

    Parameters
    ----------
    data_path : str
        path to data file

    var_path : str
        path to variance file

    ret_hdr : boolean
        whether to return header or not. Default is True.

    Returns
    -------
    cube1 : :obj:'~numpy.ndarray'
        data cube

    vcube : :obj:'~numpy.ndarray'
        variance cube

    wave : :obj:'~numpy.ndarray'
        wavelength array

    hdr1 : FITS header object
        fits header. Only if ret_hdr is True.
    """
    # open cube and get the data
    incube1 = fits.open(data_path)
    cube1 = incube1[0].data
    hdr1 = incube1[0].header
    incube1.close()
    # open variance cube and get the variances
    varcube = fits.open(var_path)
    vcube = varcube[0].data
    vhdr = varcube[0].header
    varcube.close()
    # read header to get info about the wavelength dimension
    lamstart = hdr1['CRVAL3']
    deltalam = hdr1['CDELT3']
    lamlength = hdr1['NAXIS3']
    # get wavelength array
    wave = np.linspace(lamstart, lamstart + deltalam*lamlength - 1,
                       num=lamlength)
    if ret_hdr:
        return cube1, vcube, wave, hdr1
    else:
        return cube1, vcube, wave


def save_cube(data, hdr, var=None, cubeid=None, desc=None):
    """
    Saves data cube as a fits file

    Parameters
    ----------
    data : :obj:'~numpy.ndarray'
        data cube

    hdr : FITS header object
        header of the data

    var : :obj:'~numpy.ndarray' or None
        variance cube. If given, it will save the variance in a separate .fits
        file. Defaul is None.

    cubeid : str or None
        name of the object contained in the cube. Default is None.

    desc : str or None
        description of the cube. Default is None.

    Returns
    -------
    Saves the data as a fits file
    """
    # creates the fits object
    cube_fits = fits.PrimaryHDU(data, hdr)
    # defines these keyword that will be in the final name of the file
    if cubeid is None:
        cubeid = ''
    if desc is None:
        desc = ''
    # writes the fits file
    cube_fits.writeto(desc+'_'+cubeid+'data.fits', overwrite=True)
    # writes the variance file, if there is one
    if var is not None:
        var_fits = fits.PrimaryHDU(var, hdr)
        var_fits.writeto(desc+'_'+cubeid+'var.fits', overwrite=True)


def vel(wav, wavobs):
    """
    Calculates the velocity difference of a object, given its observed
    wavelength

    Parameters
    ----------
    wav : float
        rest-frame wavelegnth to use as reference

    wavobs : float or :obj:'~numpy.ndarray'
        observed wavelength. Can be an array.

    Returns
    -------
    delta_v : float or :obj:'~numpy.ndarray'
        difference in velocity
    """
    delta_wav = wavobs - wav
    delta_v = delta_wav/wav * c
    return delta_v


def wav_obs(wav, delta_v):
    """
    Calculates the observed wavelength of an object, given its velocity

    Parameters
    ----------
    wav : float
        rest-frame wavelength to use as reference

    delta_v: float or :obj:'~numpy.ndarray'
        difference in velocity. Can be an array.

    Returns
    -------
    wavobs : float or :obj:'~numpy.ndarray'
        observed wavelength
    """
    delta_wav = delta_v/c * wav
    wavobs = wav + delta_wav
    return wavobs


def sub_cont(wav, spec, wobs):
    """
    Substracts continuum from spectrum by doing a simple 3rd degree polynomial
    fit around an emission line

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelegnth array of the spectrum

    spec : :obj:'~numpy.ndarray'
        flux of the spectrum

    wobs : float
        observed wavelength of the emission line

    Returns
    -------
    spec_clean : :obj:'~numpy.ndarray'
        flux with the continuum subtracted
    """
    # find the wavelength cut
    lcut_1 = np.round(wobs - 7)
    lcut_2 = np.round(wobs + 7)
    # find the index of the cut
    p_1 = (np.abs(wav - lcut_1)).argmin()
    p_2 = (np.abs(wav - lcut_2)).argmin()
    # cut to the left and to the right
    wav_cont_left = wav[0:p_1]
    wav_cont_right = wav[p_2:-1]
    spec_cont_left = spec[0:p_1]
    spec_cont_right = spec[p_2:-1]
    # concatenate left and right
    wav_cont = np.concatenate((wav_cont_left, wav_cont_right))
    spec_cont = np.concatenate((spec_cont_left, spec_cont_right))
    # make fit
    z = np.polyfit(wav_cont, spec_cont, 3)
    p = np.poly1d(z)
    cont = p(wav)
    # subtract continuum
    spec_clean = spec - cont
    return spec_clean


def cut_spec(wav, spec, var, delta, wobs, full=True, cont=False):
    """
    Cuts spectrum in the specified window.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the spectrum

    spec : :obj:'~numpy.ndarray'
        flux array of the spectrum

    var : :obj:'~numpy.ndarray'
        variance array of the spectrum

    delta : float
        wavelength width of the window to cut the spectrum. The spectrum will
        be cut in the window [wobs - delta, wobs + delta]

    wobs : float
        central wavelength of the window to cut the spectrum. The spectrum will
        be cut in the window [wobs - delta, wobs + delta]

    full : boolean
        if True, will return a full output. This includes initial guesses for
        the amplitude, central wavelength and velocity dispersion of a Gaussian
        fit, along with its lower and upper bounds. Default is True.

    cont : boolean
        if True, will perform a simple continuum subtraction around wobs.
        Default is False.

    Returns
    -------
    wav_cut : :obj:'~numpy.ndarray'
        cut wavelength array

    spec_cut : :obj:'~numpy.ndarray'
        cut flux array

    var_cut : :obj:'~numpy.ndarray'
        cut variance array

    p0 : :obj:'~numpy.ndarray'
        initial guesses for the amplitude, central wavelength and velocity
        dispersion of a Gaussian fit. Only if full is True.

    bounds : 2-tuple of :obj:'~numpy.ndarray'
        lower and upper bounds for a gaussian fit. It includes values that
        would be physically possible for an emission line. Only if full is
        True.
    """
    # find the wavelengths to cut
    lcut_1 = np.round(wobs - delta)
    lcut_2 = np.round(wobs + delta)
    # fins the indexes of those wavelengths to cut
    p_1 = (np.abs(wav - lcut_1)).argmin()
    p_2 = (np.abs(wav - lcut_2)).argmin()
    # cut the input arrays
    wav_cut = wav[p_1:p_2]
    spec_cut = spec[p_1:p_2]
    var_cut = var[p_1:p_2]
    if cont:
        # remove continuum
        spec_cut = sub_cont(wav_cut, spec_cut, wobs)
    if full:
        # find the initial guesses and bounds for a gaussian fit
        if max(spec_cut) < 0.:
            amplitude = -1 * max(spec_cut)
        else:
            amplitude = max(spec_cut)
        p0 = np.array([amplitude, np.mean(wav_cut), np.std(wav_cut)])
        low_bounds = np.array([0, wobs-7., 1.0])
        up_bounds = np.array([np.inf, wobs+7., 15])
        bounds = np.array([low_bounds, up_bounds])
        return wav_cut, spec_cut, var_cut, p0, bounds
    else:
        return wav_cut, spec_cut, var_cut


def cut_spec_n(wav, spec, var, delta, wobs_array, n, cont=False, four=False,
               bpt=False):
    """
    Cuts n spectra in the specified windows. It concatenates those cuts, so
    that it returns a single spectrum in the end.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the spectrum

    spec : :obj:'~numpy.ndarray'
        flux array of the spectrum

    var : :obj:'~numpy.ndarray'
        variance array of the spectrum

    delta : float
        wavelength width of the window to cut the spectrum. The spectrum will
        be cut in the windows [wobs[i] - delta, wobs[i] + delta].

    wobs_array : :obj:'~numpy.ndarray'
        array with the central wavelengths of the windows to cut the spectrum.
        The spectrum will be cut in the windows
        [wobs[i] - delta, wobs[i] + delta].

    n : int
        number of windows to perform on the spectra

    cont : boolean
        if True, will perform a simple continuum subtraction around wobs.
        Default is False.

    four : boolean
        if False, will assume the first two lines are a doublet and calculate
        the initial guesses and upper and lower bounds accordingly. The default
        is False.

    bpt : boolean
        if False, will assume the first two lines are a doublet and calculate
        the initial guesses and upper and lower bounds accordingly. The default
        is False.

    Returns
    -------
    wav_cut : :obj:'~numpy.ndarray'
        cut wavelength array

    spec_cut : :obj:'~numpy.ndarray'
        cut flux array

    var_cut : :obj:'~numpy.ndarray'
        cut variance array

    p0 : :obj:'~numpy.ndarray'
        initial guesses for the amplitude, central wavelength and velocity
        dispersion of a Gaussian fit.

    bounds : 2-tuple of :obj:'~numpy.ndarray'
        lower and upper bounds for a gaussian fit. It includes values that
        would be physically possible for an emission line.
    """
    low_bounds = np.array([])
    up_bounds = np.array([])
    for i in range(n):
        wav_cut_i, spec_cut_i, var_cut_i = cut_spec(wav, spec, var, delta,
                                                    wobs_array[i], full=False,
                                                    cont=cont)
        if i == 0:
            wav_cut = wav_cut_i
            spec_cut = spec_cut_i
            var_cut = var_cut_i
            if max(spec_cut_i) < 0.:
                amplitude = -1 * max(spec_cut_i)
            else:
                amplitude = max(spec_cut_i)
            if four:
                p0 = np.array([amplitude, np.mean(wav_cut_i), 5, 1.0])
                low_bounds = np.array([0, wobs_array[i]-7, 1.5, 0.5])
                up_bounds = np.array([np.inf, wobs_array[i]+7, 15, 2.0])
            elif bpt:
                p0 = np.array([amplitude, np.mean(wav_cut_i), 5])
                low_bounds = np.array([0, wobs_array[i]-7, 1.5])
                up_bounds = np.array([np.inf, wobs_array[i]+7, 15])
            else:
                p0 = np.array([amplitude, np.mean(wav_cut_i), 5])
                low_bounds = np.array([0, wobs_array[i]-7, 1.5])
                up_bounds = np.array([np.inf, wobs_array[i]+7, 15])
        else:
            wav_cut = np.append(wav_cut, wav_cut_i)
            spec_cut = np.append(spec_cut, spec_cut_i)
            var_cut = np.append(var_cut, var_cut_i)
            if max(spec_cut_i) < 0.:
                amplitude = -1 * max(spec_cut_i)
            else:
                amplitude = max(spec_cut_i)
            if four or bpt:
                p0 = np.append(p0, np.array([amplitude, 5]))
                low_bounds = np.append(low_bounds, np.array([0, 1.5]))
                up_bounds = np.append(up_bounds, np.array([np.inf, 15]))
            else:
                p0 = np.append(p0, np.array([amplitude]))
                low_bounds = np.append(low_bounds, np.array([0]))
                up_bounds = np.append(up_bounds, np.array([np.inf]))
    bounds = np.array([low_bounds, up_bounds])
    return wav_cut, spec_cut, var_cut, p0, bounds


def snr_continuum(wav, spec, var, window, full=False):
    """
    Calculates continuum SNR of a spectrum in a given wavelength window.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the spectrum.

    spec : :obj:'~numpy.ndarray'
        flux array of the spectrum.

    var : :obj:'~numpy.ndarray'
        variance array of the spectrum.

    window : 2-tuple of float
        wavelength window to perform calculation of SNR.

    full : boolean
        if True, apart from returning the SNR value, it returns the median
        signal and medial noise values. Default is False.

    Returns
    -------
    median_snr : float
        median SNR value.

    median_signal : float
        median signal value. Only if full is True.

    noise_median : float
        median noise value. Only if full is True.
    """
    middle = (window[1] + window[0])/2
    delta = np.abs(window[1] - middle)
    wav_cut, spec_cut, var_cut = cut_spec(wav, spec, var, delta, middle,
                                          full=False)
    snr = spec_cut/np.sqrt(var_cut)
    median_snr = np.nanmedian(snr)
    if full:
        median_signal = spec_cut[np.argsort(spec_cut)[len(spec_cut)//2]]
        noise_median = np.sqrt(var_cut[np.argsort(var_cut)[len(var_cut)//2]])
        return median_snr, median_signal, noise_median
    else:
        return median_snr


def snr_continuum_cube(wav, data, var_cube, window, full=False):
    """
    Applies snr_continuum to a data cube.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the spectra.

    data : :obj:'~numpy.ndarray'
        data cube with flux.

    var_cube : :obj:'~numpy.ndarray'
        variance cube of the data.

    window : 2-tuple of float
        wavelength window to perform calculation of SNR.

    full : boolean
        if True, apart from returning the SNR value, it returns the median
        signal and medial noise values. Default is False.

    Returns
    -------
    snr_map : :obj:'~numpy.ndarray'
        SNR map of the data cube.

    signal_map : :obj:'~numpy.ndarray'
        map of median signal of the data cube.

    noise_map :obj:'~numpy.ndarray'
        map of median noise of the data cube.
    """
    image = data[0, :, :]
    y_total, x_total = image.shape
    snr_map = np.full_like(image, 0, dtype=np.double)
    if full:
        signal_map = np.full_like(image, 0, dtype=np.double)
        noise_map = np.full_like(image, 0, dtype=np.double)
    for y in range(y_total):
        for x in range(x_total):
            spectrum = data[:, y, x]
            variance = var_cube[:, y, x]
            if full:
                snr, signal, noise = snr_continuum(wav, spectrum, variance,
                                                   window, full=True)
                snr_map[y, x] = snr
                signal_map[y, x] = signal
                noise_map[y, x] = noise
            else:
                snr = snr_continuum(wav, spectrum, variance, window)
                snr_map[y, x] = snr
    if full:
        return snr_map, signal_map, noise_map
    else:
        return snr_map


def normalise_spectra(wav_cut, spec_cut, var_cut, window, wobs, deg=2):
    """
    normalises the continuum  around an emission line by fitting a polynomial.

    Parameters
    ----------
    wav_cut : :obj:'~numpy.ndarray'
        cut wavelength array. Must be cut around the absorption line.

    spec_cut : :obj:'~numpy.ndarray'
        cut flux array. Must be cut around the absorption line.

    var_cut : :obj:'~numpy.ndarray'
        variance array. Must be cut around the absorption line.

    window : 4-tuple of float
        window to perform the cut on the continuum.

    wobs : float
        observed wavelength of the absorption line.

    deg : int
        degree of the polynomial to be used to fit the continuum around the
        absorption line.

    Returns
    -------
    spec_norm : :obj:'~numpy.ndarray'
        normalised flux array.
    """
    left_far_window = window[0]
    left_near_window = window[1]
    right_near_window = window[2]
    right_far_window = window[3]
    p_1 = (np.abs(wav_cut - (wobs - left_far_window))).argmin()
    p_2 = (np.abs(wav_cut - (wobs - left_near_window))).argmin()
    cont_left_wav = wav_cut[p_1:p_2]
    cont_left_spec = spec_cut[p_1:p_2]
    cont_left_var = var_cut[p_1:p_2]
    p_1 = (np.abs(wav_cut - (wobs + right_near_window))).argmin()
    p_2 = (np.abs(wav_cut - (wobs + right_far_window))).argmin()
    cont_right_wav = wav_cut[p_1:p_2]
    cont_right_spec = spec_cut[p_1:p_2]
    cont_right_var = var_cut[p_1:p_2]
    cont_wav = np.concatenate((cont_left_wav, cont_right_wav))
    cont_spec = np.concatenate((cont_left_spec, cont_right_spec))
    cont_var = np.concatenate((cont_left_var, cont_right_var))
    z_cont = np.polyfit(cont_wav, cont_spec, deg, w=1 / cont_var)
    p_cont = np.poly1d(z_cont)
    spec_norm = spec_cut / p_cont(wav_cut)
    return spec_norm


def create_mask(snr_map, threshold):
    """
    Creates a mask of spaxels with SNR higher than a certain threshold.

    Parameters
    ----------
    snr_map : :obj:'~numpy.ndarray'
        array with SNR map of the data cube.

    threshold : float
        minimum SNR allowed in the mask.

    Returns
    -------
    mask : :obj:'~numpy.ndarray'
        bool mask marking the spaxels that have a SNR greater or equal than the
        threshold.
    """
    mask = snr_map >= threshold
    return mask


def snr_line(wav, spec, var, line, full=False):
    """
    Calculates line SNR of a spectrum around a certain emission line.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the spectrum.

    spec : :obj:'~numpy.ndarray'
        flux array of the spectrum.

    var : :obj:'~numpy.ndarray'
        variance array of the spectrum.

    line : float
        observed wavelength of the emission line.

    full : boolean
        if True, apart from returning the SNR value, it returns the median
        signal and medial noise values. Default is False.

    Returns
    -------
    median_snr : float
        integrated SNR value.

    median_signal : float
        integrated signal value. Only if full is True.

    noise_median : float
        integrated noise value. Only if full is True.
    """
    delta_lambda = wav[1] - wav[0]
    wav_cut, spec_cut, var_cut = cut_spec(wav, spec, var, 20., line,
                                          full=False, cont=True)
    signal = np.sum(spec_cut * delta_lambda)
    noise = np.sqrt(np.sum(var_cut * delta_lambda**2.0))
    snr = signal/noise
    if full:
        return snr, signal, noise
    else:
        return snr


def snr_line_cube(wav, data, var_cube, line, full=False):
    """
    Applies snr_line to a data cube.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the spectra.

    data : :obj:'~numpy.ndarray'
        data cube with flux.

    var_cube : :obj:'~numpy.ndarray'
        variance cube of the data.

    window : 2-tuple of float
        wavelength window to perform calculation of SNR.

    full : boolean
        if True, apart from returning the SNR value, it returns the median
        signal and medial noise values. Default is False.

    Returns
    -------
    snr_map : :obj:'~numpy.ndarray'
        SNR map of the data cube.

    signal_map : :obj:'~numpy.ndarray'
        map of integrated signal of the data cube.

    noise_map :obj:'~numpy.ndarray'
        map of integrated noise of the data cube.
    """
    image = data[0, :, :]
    y_total, x_total = image.shape
    snr_map = np.full_like(image, 0, dtype=np.double)
    if full:
        signal_map = np.full_like(image, 0, dtype=np.double)
        noise_map = np.full_like(image, 0, dtype=np.double)
    for y in range(y_total):
        for x in range(x_total):
            spectrum = data[:, y, x]
            variance = var_cube[:, y, x]
            if full:
                snr, signal, noise = snr_line(wav, spectrum, variance, line,
                                              full=True)
                snr_map[y, x] = snr
                signal_map[y, x] = signal
                noise_map[y, x] = noise
            else:
                snr = snr_line(wav, spectrum, variance, line)
                snr_map[y, x] = snr
    if full:
        return snr_map, signal_map, noise_map
    else:
        return snr_map


def d_lumin_cm(redshift):
    """
    Calculates luminosity distance at a given redshift on cm.

    Parameters
    ----------
    redshift : float
        redshift to calculate the luminosity distance.

    Returns
    -------
    Dlumincm : float
        luminosity distance on cm.
    """
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    Dlumin = cosmo.luminosity_distance(redshift)  # Mpc
    Dlumincm = Dlumin.to(u.cm).value
    return Dlumincm


def scale_kpc_arcsec(redshift):
    """
    Calculates how many kpc are in an arcsec at a given redshift. Assumes a
    flat Lambda CDM cosmology with H0=70 and Om0=0.3.

    Parameters
    ----------
    redshift : float
        redshift to calculate the scale.

    Returns
    -------
    scalekpc : float
        kpc scale of one arcsec.
    """
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    scale = cosmo.kpc_comoving_per_arcmin(redshift)  # kpc/arcmin
    scalekpc = scale.to(u.kpc/u.arcsec).value
    return scalekpc


def get_coord(hdr, spaxel):
    """
    Gets the WCS coordinates of a given spaxel.

    Parameters
    ----------
    hdr : FITS header object
        data cube header.

    spaxel : 2-tuple of int
        coordinates of the spaxel.

    Returns
    -------
    coord[0] : ????
        WCS coordinates in (RA, DEC).
    """
    w = WCS(hdr)
    coord = w.pixel_to_world(spaxel[0], spaxel[1], 0)
    return coord[0]


def dist_kpc(coord1, coord2, redshift):
    """
    Calculates the projected distance in kpc of two objects at a given
    redshift.

    Parameters
    ----------
    coord1 : ????
        WCS coordinates of the first object in (RA, DEC).

    coord2 : ????
        WCS coordinates of the second object in (RA, DEC).

    redshift : float
        redsihft of the objects.

    Returns
    -------
    dis_kpc : float
        projected distance between the two objects in kpc.
    """
    dis = coord1.separation(coord2)
    dis_kpc = dis.to(u.arcsec).value * scale_kpc_arcsec(redshift)
    return dis_kpc


def closest_key(dictionary, number):
    """
    Finds the key associated to the closest value to a given number.

    Parameters
    ----------
    dictionary : dict
        dictionary with the keys and values to search.

    number : float
        number to find the closest value to.

    Returns
    -------
    closest_key_solution : str
        closest dictionary key to the number.
    """
    key_list = list(dictionary.keys())
    dist = np.inf
    closest_index = np.nan
    for i in range(len(key_list)):
        if np.abs(number-float(key_list[i])) < dist:
            dist = number-float(key_list[i])
            closest_index = i
    closest_key_solution = key_list[closest_index]
    return closest_key_solution


def abundance2metallicity(abundance):
    """
    Transforms oxygen abundance into metallicity. LOOK UP FOR CITATION

    Parameters
    ----------
    abundance : float
        original oxygen abundance to transform into metallicity.

    Returns
    -------
    log_metallicity : float
        base 10 logarithm of the metallicity given the inputed oxygen
        abundance.
    """
    metallicity = 10**abundance/10**8.75
    log_metallicity = np.log10(metallicity)
    return log_metallicity


def abundance2metallicity_KK(abundance):
    """
    Transforms oxygen abundance into metallicity. LOOK UP FOR CITATION

    Parameters
    ----------
    abundance : float
        original oxygen abundance to transform into metallicity.

    Returns
    -------
    metallicity : float
        metallicity given the inputed oxygen abundance.
    """
    metallicity = np.log10(29/0.015) + abundance - 12
    return metallicity


def create_vorbin_input(signal_map, noise_map, cut=0.):
    """
    Creates an input file that the vorbin software can take as input and saves
    it as a .txt file.

    Parameters
    ----------
    signal_map : :obj:'~numpy.ndarray'
        signal map of the data cube.

    noise_map : :obj:'~numpy.ndarray'
        noise map of the data cube.

    cut : float
        minimum SNR that each bin should have. Default is 0.

    Returns
    -------
    vorbin_input : :obj:'~numpy.ndarray'
        table with coordinates of each spaxels and their signal and noise
        values.
    """
    y_total, x_total = signal_map.shape
    total_spaxels = y_total * x_total
    vorbin_input = np.zeros((total_spaxels, 4))
    signal_map[np.isnan(signal_map)] = np.nanmedian(signal_map)
    noise_map[np.isnan(noise_map)] = np.nanmedian(noise_map)
    counter = 0
    for y in range(y_total):
        for x in range(x_total):
            signal = signal_map[y, x]
            noise = noise_map[y, x]
            vorbin_input[counter] = np.array([int(x), int(y), signal, noise])
            counter = counter + 1
    np.savetxt('vorbin/vorbin_input.txt', vorbin_input)
    return vorbin_input


def voronoi_binning_example(file_dir, target_sn, cubeid=None):
    """
    Usage of the vorbin software. Creates and saves a .txt file with the
    coordinates of each spaxel and the number of their assigned bin.

    Parameters
    ----------
    file_dir : str
        directory with the location of the input file. This input file must
        contain four columns: x coordinate, y coordinate, signal and noise of
        each spaxel. It can be created using create_vorbin_input.

    target_sn : float
        minimum SNR that each bin must have.

    cubeid : str
        name of the object. If given, the output file will include it in its
        name. Default is None.
    """
    """
    Usage example for the procedure VORONOI_2D_BINNING.

    It is assumed below that the file voronoi_2d_binning_example.txt
    resides in the current directory. Here columns 1-4 of the text file
    contain respectively the x, y coordinates of each SAURON lens
    and the corresponding Signal and Noise.

    """
    x, y, signal, noise = np.loadtxt(file_dir).T

    # Perform the actual computation. The vectors
    # (binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale)
    # are all generated in *output*
    #
    binNum, aa, bb, cc, dd, ee, ff, gg = voronoi_2d_binning(x, y, signal,
                                                            noise, target_sn,
                                                            plot=1, quiet=0)

    # Save to a text file the initial coordinates of each pixel together
    # with the corresponding bin number computed by this procedure.
    # binNum uniquely specifies the bins and for this reason it is the only
    # number required for any subsequent calculation on the bins.
    #
    np.savetxt('vorbin/voronoi_2d_binning_example_output_'+cubeid+'.txt',
               np.column_stack([x, y, binNum]), fmt=b'%10.6f %10.6f %8i')
    plt.tight_layout()
    plt.pause(1)


def calculate_ew(wav, spec, err=None, window=None, linear=True):
    """
    calculates the equivalent width of a given absorption line.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        normalised spectrum with the absorption line.

    err : :obj:'~numpy.ndarray'
        error array of the spectrum. If given it will calculate an error and
        return an error value. Default is None.

    window : 2-tuple of float
        wavelength window to calculate the equivalent width. If given, it will
        cut the spectrum to only include that window. Default is None.

    linear : boolean
        if True, it will assume that the wavelength is linearly sampled.
        Default is True.

    Returns
    -------
    ew : float
        calculated equivalent width value.

    ew_error : float
        calculate equivalent width error. Only if err is not None.
    """
    if window is not None:
        wobs = (window[1] + window[0])/2.
        delta = window[1] - wobs
        wav, spec, aa = cut_spec(wav, spec, wav, delta, wobs, full=False,
                                 cont=False)
    if linear:
        delta_lambda = wav[1] - wav[0]
        ew = np.sum((1 - spec)*delta_lamda)
    else:
        ew = 0
        for i in range(len(wav) - 1):
            delta_lambda = wav[i + 1] - wav[i]
            ew += (1 - spec[i]) * delta_lambda
    if err is not None:
        wav_err = err[0]
        error = err[1]
        ew_error = 0
        for i in range(len(wav_err) - 1):
            delta_lambda = wav_err[i + 1] - wav_err[i]
            ew_error += (error[i]*delta_lambda)**2
        ew_error = np.sqrt(ew_error)
        return ew, ew_error
    else:
        return ew


def wav_space(vel, spec, ion, zabs):
    """
    Transforms velocity array into wavelength fora given transition at a given
    redshift.

    Parameters
    ----------
    vel : :obj:'~numpy.ndarray'
        velocity array.

    spec : :obj:'~numpy.ndarray'
        spectrum.

    ion : float
        rest-frame equivalent width of the relevant line.

    zabs : float
        relevant redshift.

    Returns
    -------
    wav_obs : :obj:'~numpy.ndarray'
        observed wavelengths corresponding to the inputed velocities.
    """
    wav_z = ion*(1 + zabs)
    wav_obs = vel * wav_z / c + wav_z
    return wav_obs


def nearest_value_index(array, value):
    """
    find the array index to the nearest array element to a certain value.

    Parameters
    ----------
    array : :obj:'~numpy.ndarray'
        array to look for the nearest value.

    value : float
        given value.

    Returns
    -------
    index : int
        index to the closest element in the array to the value.
    """
    index = np.nanargmin(np.abs(array-value))
    return index


def bpt_curve(redshift):
    """
    calculates the BPT curve at a given redshift. LOOK FOR CITATION.

    Parameters
    ----------
    redshift : float
        given redshift of the relevant objects.

    Returns
    -------
    log_nii_alpha : :obj:'~numpy.ndarray'
        array with the log10 of values of NII/Halpha.

    log_oiii_hbeta : :obj:'~numpy.ndarray'
        array with the log10 of the corresponding values of OIII/Hbeta.
    """
    log_nii_halpha = np.linspace(-2.0, 0, num=10000)
    log_oiii_hbeta = (0.61 / (log_nii_halpha - 0.02 - 0.1833 * redshift)
                      + 1.2 + 0.03 * redshift)
    return log_nii_halpha, log_oiii_hbeta


def mask_from_regfile(regfile, image):
    """
    creates a mask from a DS9 .reg file.

    Parameters
    ----------
    regfile : str
        directory with the location of the .reg file.

    image : :obj:'~numpy.ndarray'
        array with the image the .reg file comes from.

    Returns
    -------
    mask2d : :obj:'~numpy.ndarray'
        2D array with the information from the .reg file.
    """
    if len(image.shape) == 3:
        image = image[:, 0, 0]
    r = pyregion.open(regfile)
    r_0 = pyregion.ShapeList([r[0]])
    shape = image.shape
    region_filter = as_region_filter(r, origin=1)
    mask_new = region_filter.mask(shape)
    mask_new_inverse = np.where(~mask_new, True, False)
    mask2d = mask_new_inverse
    return mask2d


def create_spectrum_list(vorbin_path, data_path, var_path):
    """
    creates spectra list from the vorbin software output. Each element of the
    list corresponds to the integrated spectrum with the number corresponding
    to the index of said element. It also generates a variance list and outputs
    the vorbin output in the form of a pandas DataFrame.

    Parameters
    ----------
    vorbin_path : str
        directory with the location of the vorbin output file.

    data_path : str
        directory with the location of the data cube file.

    var_path : str
        directory with the location of the variance cube file.

    Returns
    -------
    df : pandas DataFrame
        pandas DataFrame with the information from the vorbin output file.

    spectrum_list : :obj:'~numpy.ndarray'
        array with the integrated spectra. Each element of the array is the
        integrated spectrum that corresponds to the bin with the same number as
        the index.

    var_list : :obj:'~numpy.ndarray'
        array with the integrated variances. Each element of the array is the
        integrated variance that corresponds to the bin with the same number as
        the index.
    """
    cube, var, wav, hdr = read_cube(data_path, var_path)
    df = pd.read_csv(vorbin_path, names=['x', 'y', 'number'], sep=r"\s+")
    df = df.sort_values(by=['number'], ignore_index=True)
    number_of_bins = df['number'][df.shape[0] - 1] + 1
    spectrum_list = np.zeros((number_of_bins, len(wav)))
    var_list = np.zeros((number_of_bins, len(wav)))
    counter = 0
    for i in range(number_of_bins):
        while df['number'][counter] == i:
            spectrum_list[i] = spectrum_list[i] + cube[:,
                                                       int(df['y'][counter]),
                                                       int(df['x'][counter])]
            var_list[i] = var_list[i] + var[:,
                                            int(df['y'][counter]),
                                            int(df['x'][counter])]
            counter += 1
            if counter == df.shape[0]:
                break
        if counter == df.shape[0]:
            break
    return df, spectrum_list, var_list


def create_cube_vorbin(vorbin_path, spectrum_list, var_list, data_path,
                       var_path):
    """
    creates a voronoi binned cube.

    Parameters
    ----------
    vorbin_path : str
        directory with the location of the vorbin output file.

    spectrum_list : :obj:'~numpy.ndarray'
        array with spectra, where each spectra corresponds to the bin of its
        index.

    var_list : :obj:'~numpy.ndarray'
        array with variance, where each variance corresponds to the bin of its
        index.

    data_path : str
        directory with the location of the original data cube.

    var_path : str
        directory with the location of the original variance cube.

    Returns
    -------
    vorbin_cube : :obj:'~numpy.ndarray'
        voronoi binned cube. It will also be saved as a FITS file.
    """
    cube, var, wav, hdr = read_cube(data_path, var_path)
    df = pd.read_csv(vorbin_path, names=['x', 'y', 'number'], sep=r"\s+")
    vorbin_cube = np.zeros((spectrum_list.shape[1],
                            int(df['y'].iloc[-1]) + 1,
                            int(df['x'].iloc[-1]) + 1))
    vorbin_var = np.zeros((spectrum_list.shape[1],
                           int(df['y'].iloc[-1]) + 1,
                           int(df['x'].iloc[-1]) + 1))
    for i in range(df.shape[0]):
        y_cube = int(df['y'][i])
        x_cube = int(df['x'][i])
        vorbin_cube[:, y_cube, x_cube] = spectrum_list[int(df['number'][i])]
        vorbin_var[:, y_cube, x_cube] = var_list[int(df['number'][i])]
    save_cube(vorbin_cube, hdr, var=vorbin_var, cubeid='00047',
              desc='vorbin_cont_11')
    return vorbin_cube

def calc_mag(flux, band):
    sun_abs_mag = {"r": 8, "i": 5.21}
    sun_app_mag = {"r": -26.93, "i": -26.37}
    L_bol_sun = 3.828 * 10**33 * u.erg / u.s
    L_sun = L_bol_sun
    #L_sun = L_bol_sun * 10**((4.74 - sun_abs_mag[band])/2.5)
    #print(f"The luminosity of the sun is {L_sun}")
    d_sun = 1.495978707 * 10**11 * u.m
    f_sun = L_sun / (4 * np.pi * d_sun**2)
    f_sun = f_sun.to(u.erg / u.s / u.cm**2).value
    mag = -2.5 * np.log10(flux / f_sun) + sun_app_mag[band]
    return mag






