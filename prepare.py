#!/usr/bin/env python

#####################################
# Functions that prepare the cubes
#####################################

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric
from astropy.constants import c
from astropy.time import Time
import astropy.units as u
import astropy.constants as consts
import pandas as pd
from scipy import integrate

import analysis
import useful
from constants_wavelengths import *

# Bin the data


def binning(data, var, wav, hdr, bin_size=2, save=True, path=None,
            cubeid=None):
    """
    Bins the data to the desired bin size.

    Parameters
    ----------
    data : :obj:'~numpy.ndarray'
        data cube

    var : :obj:'~numpy.ndarray'
        variance cube

    wav : :obj:'~numpy.ndarray'
        wavelength array

    hdr : FITS header object
        fits header.

    bin_size : int
        desired final bin size. Defaul is 2.

    save : boolean
        if True, it will save the final data cube into a fits file. Default is
        True.

    path : str
        Data path to saved the final data cube. Used only if save is True. If
        None, the binned cubed will be saved to the working directory. Default
        is None.

    cubeid : str
        Name of the data cube. Used only if save is True. If given, it will be
        included in the name of the saved data cube. Default is None.

    Returns
    -------
    rebinned : :obj:'~numpy.ndarray'
        Final rebinned data cube.

    rebinned_var : :obj:'~numpy.ndarray'
        Final rebinned variance cube.
    """
    crpix1 = hdr['CRPIX1']
    crpix2 = hdr['CRPIX2']
    pixscale = hdr['CDELT2']*3600.0
    slicescale = hdr['CDELT2']*3600.0
    # Rebin the Montage'd cube 2x2 to better account for the oversampling
    lwave, lpix, lslice = np.shape(data)
    """
    Given that the Large slicer has spaxel dimensions of 1.35"x0.294"
      - There are 24 slices and 63 pixels
      - Conversion to kps is 0.387 kpc/" (This is for a z=0.02 galaxy)
      - The slices have width ~520 pc  (z=0.02)
      - The pixels have width ~115pc  (z=0.02)
    Montage rebins the spaxels to be square: 0.294"x0.294" or 115x115 pc
      - There are 4.59 new spaxels in a single slice
      - Rebin to bigger square spaxels because of oversampling
      - The seeing is also worse than the ccd resolution
      - Rebinning to at least 2x2
    """
    # Number of spaxels in a bin
    pixinbin = bin_size
    sliceinbin = bin_size
    area = (pixinbin*pixscale)*(sliceinbin*slicescale)
    # Number of pixel/slice bins
    pixbins = int(lpix/pixinbin)
    slicebins = int(lslice/sliceinbin)
    cube_rebin = np.zeros([lwave, pixbins, slicebins])
    vcube_rebin = np.zeros([lwave, pixbins, slicebins])
    i = 0
    for ni in range(slicebins):
        j = 0
        for nj in range(pixbins):
            dum = np.nansum(data[:, j:j+pixinbin, i:i+sliceinbin], axis=(1, 2))
            cube_rebin[:, nj, ni] = dum
            vdum = np.nansum(var[:, j:j+pixinbin, i:i+sliceinbin], axis=(1, 2))
            vcube_rebin[:, nj, ni] = vdum
            j += pixinbin
        i += sliceinbin
    rebinshape = np.shape(cube_rebin)
    slicescale *= float(sliceinbin)
    pixscale *= float(pixinbin)
    # usage:
    rebinned = fits.PrimaryHDU(cube_rebin, hdr)
    rhdr = rebinned.header
    rhdr['NAXIS1'] = rebinshape[2]
    rhdr['NAXIS2'] = rebinshape[1]
    rhdr['CRPIX1'] = (crpix1-0.5)*(slicebins/lslice) + 0.5
    rhdr['CRPIX2'] = (crpix2-0.5)*(pixbins/lpix) + 0.5
    # rhdr['CD1_1']  = rhdr['CD1_1'] / (slicebins/lslice)
    # rhdr['CD2_1']  = rhdr['CD2_1'] / (slicebins/lslice)
    # rhdr['CD1_2']  = rhdr['CD1_2'] / (pixbins/lpix)
    # rhdr['CD2_2']  = rhdr['CD2_2'] / (pixbins/lpix)
    rhdr['CDELT1'] = rhdr['CDELT1'] / (slicebins/lslice)
    rhdr['CDELT2'] = rhdr['CDELT2'] / (pixbins/lpix)
    rhdr['NAXIS3'] = len(wav)
    rhdr['CRVAL3'] = wav[0]
    rebinned_var = fits.PrimaryHDU(vcube_rebin)
    if save:
        rebinned.writeto(path+'r'+str(bin_size)+'_kb220304_'+cubeid
                         + '_icubes_proj.fits', overwrite=True)
        rebinned_var.writeto(path+'r'+str(bin_size)+'_kb220304_'+cubeid
                             + '_vcubes_proj.fits', overwrite=True)
    return rebinned, rebinned_var

# MW extinction correction


def milky_way_extinction_correction(lamdas, data, var, Av_NED=0.109):
    """
    Corrects for the extinction caused by light travelling through the
    dust and gas of the Milky Way, as described in Cardelli et
    al. 1989.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        wavelength vector

    data : :obj:'~numpy.ndarray'
        3D cube of data

    var : :obj:'~numpy.ndarray'
        3D cube of variances

    Av_NED : float
        Av value at the RA and DEC of the object. Default is 0.109.

    Returns
    -------
    data : :obj:'~numpy.ndarray'
        the data corrected for extinction
    """
    # convert lamdas from Angstroms into micrometers
    lamdas = 10000.0/lamdas

    # define the equations from the paper
    y = lamdas - 1.82
    a_x = 1.0 + 0.17699*y - 0.50447*(y**2.0) - 0.02427*(y**3.0) \
        + 0.72085*(y**4.0) + 0.01979*(y**5.0) - 0.77530*(y**6.0) \
        + 0.32999*(y**7.0)
    b_x = 1.41338*y + 2.28305*(y**2.0) + 1.07233*(y**3.0) - 5.38434*(y**4.0) \
        - 0.62251*(y**5.0) + 5.30260*(y**6.0) - 2.09002*(y**7.0)

    # define the constants
    Rv = 3.1  # this is the usual adopted value
    Av = Av_NED  # This comes from NED Extinction Calculator

    # find A(lambda)
    A_lam = (a_x + b_x/Rv)*Av

    # apply to the data
    data = (10.0**(0.4*A_lam[:, None, None])) * data
    var = (10.0**(0.4*A_lam[:, None, None]))**2.0 * var

    return data, var


def calculate_ebv(hbeta_flux, hgamma_flux):
    """
    Calculates E(B - V) given the Hbeta and Hgamma measured flux.

    Parameters
    ----------
    hbeta_flux : float
        Hbeta flux measured in the spectrum. The units are not important, as
        long as they are the same as the units of hgamma_flux.

    hgamma_flux : float
        Hgamma flux measured in the spectrum. The units are not important, as
        long as they are the same as the units of hbeta_flux.

    Returns
    -------
    ebv : float
        E(B - V) value measured in the spectrum. Will return 0.01 if the flux
        ratio is lower than 2.5.

    flux_ratio : float
        flux ratio defined as Hbeta/Hgamma.
    """
    diff_ext = 0.465
    actual_ratio = 2.15
    flux_ratio = hbeta_flux / hgamma_flux
    if flux_ratio >= 2.15:
        ebv = 2.5 / diff_ext * np.log10(flux_ratio/actual_ratio)
    else:
        ebv = 0.01
    return ebv, flux_ratio


def ebv_cube(wav, data, var, redshift, mask=None, plot=False, wav_type='air',
             n=1):
    """
    Applies calculate_ebv to a data cube.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data

    data : :obj:'~numpy.ndarray'
        data cube

    var : :obj:'~numpy.ndarray'
        variance cube

    redhsift : float
        the redshift of the object

    mask : :obj:'~numpy.ndarray'
        array of boolean. If given the calculation will only be performed in
        the spaxels with True value. Default is None.

    plot : boolean
        if True, it will plot the Hbeta and Hgamma fits. Default is False.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    n : int
        number of velocity components. Default is 1.

    Returns
    -------
    ebv : :obj:'~numpy.ndarray'
        2D array with the calculated E(B - V) of each spaxel.

    flux_ratio : :obj:'~numpy.ndarray'
        2D array with the calculated Hbeta/Hgamma of each spaxel.
    """
    image = data[0, :, :]
    y_total, x_total = image.shape
    ebv = np.full_like(image, 0, dtype=np.double)
    flux_ratio = np.full_like(image, np.nan, dtype=np.double)
    # ebv = np.full_like(image, 0, dtype=np.double)
    for y in range(y_total):
        for x in range(x_total):
            if mask is not None:
                if ~ mask[y, x]:
                    continue
            spectrum = data[:, y, x]
            variance = var[:, y, x]
            popt, pcov = analysis.hbeta_hgamma_fitting(wav, spectrum, redshift,
                                                       var=variance, plot=True,
                                                       coord=(x, y),
                                                       wav_type=wav_type, n=n)
            if type(popt) == int:
                ebv[y, x] = 0.01
                flux_ratio[y, x] = 2.15
                continue
            else:
                hbeta_flux, hgamma_flux = analysis.hbeta_hgamma_fluxes(popt,
                                                                       n=n)
                ebv[y, x], flux_ratio[y, x] = calculate_ebv(hbeta_flux,
                                                            hgamma_flux)
    return ebv, flux_ratio


def dust_extinction_correction(wav, data, var, redshift, mask=None,
                               ebv_map=None, wav_type='air', n=1):
    """
    Performs dust extinction correction on a data cube.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    data : :obj:'~numpy.ndarray'
        data cube to perform the dust extinction correction on.

    var : :obj:'~numpy.ndarray'
        variance cube of the data.

    redshift : float
        redshift of the object.

    mask : :obj:'~numpy.ndarray'
        array of boolean. If given the calculation will only be performed in
        the spaxels with True value. Default is None.

    ebv_map : :obj:'~numpy.ndarray'
        E(B - V) array. If None, it is calculated using ebv_cube. Default is
        None.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    n : int
        number of velocity components. Default is 1.

    Returns
    -------
    data : :obj:'~numpy.ndarray'
        final dust extinction corrected data cube.

    var : :obj:'~numpy.ndarray'
        final dust extinction corrected variance cube.
    """
    if ebv_map is None:
        ebv, flux_ratio = ebv_cube(wav, data, var, redshift, mask=mask,
                                   wav_type=wav_type, n=n)
    else:
        ebv = ebv_map
    # convert lamdas from Angstroms into micrometers
    wav = 10000.0/wav

    # define the equations from the paper
    y = wav - 1.82
    a_x = 1.0 + 0.17699*y - 0.50447*(y**2.0) - 0.02427*(y**3.0) \
        + 0.72085*(y**4.0) + 0.01979*(y**5.0) - 0.77530*(y**6.0) \
        + 0.32999*(y**7.0)
    b_x = 1.41338*y + 2.28305*(y**2.0) + 1.07233*(y**3.0) - 5.38434*(y**4.0) \
        - 0.62251*(y**5.0) + 5.30260*(y**6.0) - 2.09002*(y**7.0)

    # define the constants
    Rv = 3.1  # this is the usual adopted value
    Av = ebv * Rv
    print(f"Av value is {Av[0][0]}")

    # tile a_x and b_x so that they're the right array shape
    a_x = np.tile(a_x, [data.shape[2], data.shape[1], 1]).T
    b_x = np.tile(b_x, [data.shape[2], data.shape[1], 1]).T

    # find A(lambda)
    A_lam = (a_x + b_x/Rv)*Av

    data = (10.0**(0.4*A_lam)) * data
    var = (10.0**(0.4*A_lam))**2.0 * var

    return data, var


def remove_emission_lines(wav, spec, var, redshift, wav_type='air',
                          only_balmer=False):
    """
    Removes [O II], [O III] and Balmer series emission lines from a spectrum.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the spectrum.

    spec : :obj:'~numpy.ndarray'
        flux array of the spectrum.

    var : :obj:'~numpy.ndarray'
        variance array of the spectrum.

    redshift : float
        redshift of the object.

    wav_type : str
        wavelength type of the wavelength array. The options are "air" for air
        wavelengths and "vac" for vacuum wavelengths. Default is "air".

    only_balmer : boolean
        if True, only removes the balmer series lines and ignores the rest.
        Default is False.

    Returns
    -------
    new_spec : :obj:'~numpy.ndarray'
        flux array with the emission lines removed.
    """
    try:
        if wav_type == 'air':
            dict_wav = dict_wav_air
            four_gaussian_model = analysis.four_gaussian_model_air
            hgamma_hbeta_model = analysis.hgamma_hbeta_model_air
        elif wav_type == 'vac':
            dict_wav = dict_wav_vac
            four_gaussian_model = analysis.four_gaussian_model_vac
            hgamma_hbeta_model = analysis.hgamma_hbeta_model_vac
    except RuntimeError:
        print('No valid wave_type provided')
    if only_balmer:
        popt, pcov = analysis.hbeta_hgamma_fitting(wav, spec, redshift,
                                                   var=var, wav_type=wav_type)
        if type(popt) == int:
            return spec
        A1, mu, sigma3, A3 = popt
        found_redshift = mu/dict_wav['Hgamma'] - 1
        emission_lines = hgamma_hbeta_model(wav, *popt)
    else:
        popt, pcov = analysis.four_gaussian_fitting(wav, spec, redshift,
                                                    var=var, wav_type=wav_type)
        if type(popt) == int:
            return spec
        A1, mu, sigma1, alpha, A3, sigma3, A4, sigma4 = popt
        found_redshift = mu/dict_wav['OII_l'] - 1
        emission_lines = four_gaussian_model(wav, *popt)
        # Add missing lines
        # # [O III] weaker
        oiii_params = (A4/3., dict_wav['OIII_l'] * (1 + found_redshift),
                       sigma4)
        oiii_line = analysis.gaussian_model(wav, *oiii_params)
        emission_lines = emission_lines + oiii_line
        # # Hgamma
        hgamma_params = (A3 * 0.466,
                         dict_wav['Hgamma'] * (1 + found_redshift), sigma3)
        hgamma_line = analysis.gaussian_model(wav, *hgamma_params)
        emission_lines = emission_lines + hgamma_line
    # # Hdelta
    hdelta_params = (A3 * 0.256,
                     dict_wav['Hdelta'] * (1 + found_redshift), sigma3)
    hdelta_line = analysis.gaussian_model(wav, *hdelta_params)
    emission_lines = emission_lines + hdelta_line
    # # Hepsilon
    hepsilon_params = (A3 * 0.158,
                       dict_wav['Hepsilon'] * (1 + found_redshift), sigma3)
    hepsilon_line = analysis.gaussian_model(wav, *hepsilon_params)
    emission_lines = emission_lines + hepsilon_line
    # # H8
    h8_params = (A3 * 0.105, dict_wav['H8'] * (1 + found_redshift), sigma3)
    h8_line = analysis.gaussian_model(wav, *h8_params)
    emission_lines = emission_lines + h8_line
    # # H9
    h9_params = (A3 * 0.0703, dict_wav['H9'] * (1 + found_redshift), sigma3)
    h9_line = analysis.gaussian_model(wav, *h9_params)
    emission_lines = emission_lines + h9_line
    # # H10
    h10_params = (A3 * 0.0529, dict_wav['H10'] * (1 + found_redshift), sigma3)
    h10_line = analysis.gaussian_model(wav, *h10_params)
    emission_lines = emission_lines + h10_line
    new_spec = spec - emission_lines
    return new_spec


def remove_emission_lines_cube(wav, data, var, redshift, mask=None,
                               wav_type='air'):
    """
    Applies remove_emission_lines to a data cube.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data cube.

    data : :obj:'~numpy.ndarray'
        flux data cube.

    var : :obj:'~numpy.ndarray'
        variance cube.

    redshift : float
        redshift of the object.

    mask : :obj:'~numpy.ndarray'
        array of boolean. If given the removal will only be performed in the
        spaxels with True value. Default is None.

    wav_type : str
        wavelength type of the wavelength array. The options are "air" for air
        wavelengths and "vac" for vacuum wavelengths. Default is "air".

    Returns
    -------
    new_data : :obj:'~numpy.ndarray'
        flux data cube with the emission lines removed.
    """
    new_data = np.full_like(data, np.nan, dtype=np.double)
    z_total, y_total, x_total = data.shape
    for y in range(y_total):
        for x in range(x_total):
            if ~ mask[y, x]:
                new_data[:, y, x] = data[:, y, x]
                continue
            spec = data[:, y, x]
            variance = var[:, y, x]
            new_data[:, y, x] = remove_emission_lines(wav, spec, variance,
                                                      redshift,
                                                      wav_type=wav_type)
    return new_data


def bary_correction(wav, header):
    """
    Applies barycenter correction to the wavelength data. It assumes the data
    is captured from Keck.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array.

    header : FITS header object
        fits header of the data.

    Returns
    -------
    barywave : :obj:'~numpy.ndarray'
        barycenter corrected wavelength array.
    """
    racenter = header['CRVAL1']
    decenter = header['CRVAL2']
    expstart = header['DATE-BEG']
    keck = EarthLocation.from_geodetic(lat=19.8283*u.deg, lon=-155.4783*u.deg,
                                       height=4160*u.m)
    sc = SkyCoord(ra=racenter*u.deg, dec=decenter*u.deg)
    barycorr = sc.radial_velocity_correction(obstime=Time(expstart),
                                             location=keck)
    bcorr = barycorr.to(u.km/u.s)
    cval = consts.c.to('km/s').value
    barywave = wav * (1.0 + (bcorr.value/cval))
    return barywave


def flux_calib(wav, spec, target_mag, band='r'):
    """
    Performs flux calibrations using SDSS photometry. DOESN'T WORK.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array.

    spec : :obj:'~numpy.ndarray'
        flux array.

    target_mag : float
        SDSS magnitude of the relevant object.

    band : str
        SDSS used to flux calibrate.

    Returns
    -------
    new_wav : :obj:'~numpy.ndarray'
        wavelength array cut in the aforementioned band.

    new_spec : :obj:'~numpy.ndarray'
        flux array cut in the aforementioned band.

    new_band_trans : :obj:'~numpy.ndarray'
        band transmision interpolated to have the same sampling as the
        spectrum.

    filtered_spec : :obj:'~numpy.ndarray'
        spectrum after the filter is applied to it.
    """
    dict_bands = {'r': "/Users/antoniafernandezfigueroa/phd/ultrastrong/" +
                  "flux_calib/SDSS_r/SLOAN_SDSS.r.dat",
                  "i": "/Users/antoniafernandezfigueroa/phd/ultrastrong/" +
                  "flux_calib/SDSS_i/SLOAN_SDSS.i.dat"}
    band_transmission = pd.read_csv(dict_bands[band], sep=' ',
                                    names=['wavelength', 'transmission'])
    max_trans = np.max(band_transmission['transmission'])
    band_transmission['transmission'] *= 1. / max_trans
    band_trans = band_transmission['transmission'].to_numpy()
    band_wav = band_transmission['wavelength'].to_numpy()
    if band_wav[0] < wav[0]:
        # create new wavelength array that extends until band_wav[0]
        wav_step = wav[1] - wav[0]
        new_wav = np.arange(band_wav[0], wav[0] - wav_step, wav_step)
        new_wav = np.append(new_wav, wav)
        # create new spectrum array that extends until band_wav[0]
        new_spec = np.zeros(new_wav.shape)
        new_spec[-1 * (len(spec) + 1):-1] = spec
        new_spec[new_spec == 0] = np.nanmedian(spec)
        # cut them, so they have the same range as band_wav
        max_wav = band_wav[-1]
        p_1 = (np.abs(new_wav - max_wav)).argmin()
        new_spec = new_spec[0:p_1]
        new_wav = new_wav[0:p_1]
    elif band_wav[-1] > wav[-1]:
        # create new wavelength array that extends until band_wav[-1]
        wav_step = wav[-1] - wav[-2]
        new_wav = np.arange(wav[-1] + wav_step, band_wav[-1], wav_step)
        new_wav = np.append(wav, new_wav)
        # create new spectrum array that extends until band_wav[-1]
        new_spec = np.zeros(new_wav.shape)
        new_spec[0:wav.shape[0]] = spec
        new_spec[new_spec == 0] = np.nanmedian(spec)
        # cut them, so they have the same range as band_wav
        max_wav = band_wav[0]
        p_1 = (np.abs(new_wav - max_wav)).argmin()
        new_spec = new_spec[p_1:-1]
        new_wav = new_wav[p_1:-1]
    else:
        return
    # interpolate the band_trans array so it has the same sampling as wav
    new_band_trans = np.interp(new_wav, band_wav, band_trans)
    filtered_spec = new_spec * new_band_trans
    flux_band = integrate.trapezoid(filtered_spec, x=new_wav)
    print(flux_band)
    mag = useful.calc_mag(flux_band * 10**-17, band)
    print(mag)
    return new_wav, new_spec, new_band_trans, filtered_spec


def apply_barycentric_correction_and_save(input_fits_path, output_fits_path):
    """
    opens a fits file, applies barycentric correction to the wavelength and
    saves the corrected file.

    Parameters
    ----------
    input_fits_path : str
        name of the original fits file.

    output_fits_path : str
        desired name of the outputed fits file.
    """
    # Load the FITS data and header
    with fits.open(input_fits_path) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data
    # Extract required header information
    racenter = hdr['CRVAL1']
    decenter = hdr['CRVAL2']
    expstart = hdr['DATE-BEG']
    # Location of Keck
    keck = EarthLocation.from_geodetic(lat=19.8283 * u.deg,
                                       lon=-155.4783 * u.deg,
                                       height=4160 * u.m)
    # Sky coordinates of the observation
    sc = SkyCoord(ra=racenter * u.deg, dec=decenter * u.deg)
    # Time of the observation
    time = Time(expstart, format='isot', scale='utc')
    # Calculate the barycentric velocity correction
    barycorr = sc.radial_velocity_correction(obstime=time, location=keck)
    # Convert the correction from velocity to wavelength shift
    barycorr_value = barycorr.to(u.km / u.s).value
    barycorr_factor = (1 + barycorr_value / c.to(u.km / u.s).value)
    print(f'Barycentric velocity correction: {barycorr_value} km/s')
    print(f'Barycentric correction factor: {barycorr_factor}')
    # Update the wavelength axis in the FITS header
    original_crval3 = hdr['CRVAL3']
    hdr['CRVAL3'] = original_crval3 * barycorr_factor
    hdr['CDELT3'] = hdr['CDELT3'] * barycorr_factor
    hdr.add_history(f'Barycentric correction applied: {barycorr_value} km/s')
    # Save the updated data and header into a new FITS file
    fits.writeto(output_fits_path, data, header=hdr, overwrite=True)
    print(f'Barycentric corrected data saved to {output_fits_path}')
