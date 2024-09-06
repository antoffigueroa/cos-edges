#!/usr/bin/env python

#####################################
# Functions to apply ppxf
#####################################

import glob
from os import path
from time import perf_counter as clock

import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage
import numpy as np

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib

import pickle

import useful

ppxf_dir = path.dirname(path.realpath(util.__file__))
R = 900


def get_goodpixels_and_noise(loglam_gal, var, z, mask_emission,
                             LRIS_blue=False, bad_sky=None):
    """
    Return the goodpixels and a cleaned noise array based on where there
    are infs, nans in the variance array. Replace these with the median
    value and return.

    Parameters
    ----------
    loglam_gal : :obj:'~numpy.ndarray'
        wavelength array of the spectrum. It must be logarithmically sampled.

    var : :obj:'~numpy.ndarray'
        variance array of the spectrum.

    z : float
        redshift of the relevant object.

    mask_emission : boolean
        if True, it will mask the emission lines along with the noisy pixels.

    Returns
    -------
    goodpix : :obj:'~numpy.ndarray'
        array with the indexes of the good pixels.

    ns_clean : :obj:'~numpy.ndarray'
        clean noise array.
    """
    # Use (wrapper for) ppxf_util.determine_goodpixels, to identify and
    # exclude the following emission lines: [NOTE not all lines are within
    # the wavelength range of the spectrum] 2x[OII], Hdelta, Hgamma,
    # Hbeta, 2x[OIII], [OI], 2x[NII], Halpha, 2x [SII]
    # Additionaly, the wrapper also masks the prominent skyline at 5557 A.
    if mask_emission is True:
        goodpix0 = determine_goodpixels(lam_gal=loglam_gal, z=z)
    elif mask_emission is False:
        goodpix0 = np.arange(len(loglam_gal))
    ns = np.sqrt(var)  # The NOISE SPECTRUM from file.
    # Exclude in the condition calculation the pixels with inf noise
    # (where 1/var was set to 0; else the standard deviation screws up)
    # establish the condition that pixels which have an uncertainty higher
    # than 5sigma then it will be cut
    # condition    = (np.nanmedian(ns[~np.isinf(ns)])
    #                + 5 * np.nanstd(ns[~np.isinf(ns)])) , (ns > condition)
    high_err_pix = np.where((ns == 0.) | np.isnan(ns) | np.isinf(ns))[0]
    # Catches the bad pixels in the edge of the detector on LRIS blue side
    if LRIS_blue:
        obs_lam = loglam_gal * (z + 1)
        edges = np.where((obs_lam < 3875) | ((obs_lam > 4925) &
                         (obs_lam < 5610)))[0]
    # Catches infs, nans and where the noise spectrum is zero
    # Remove these high error pixels if they are in goodpixels
        goodpix = np.array([pix for pix in goodpix0
                            if pix not in high_err_pix])
        goodpix = np.array([pix for pix in goodpix
                            if pix not in edges])
    else:
        goodpix = np.array([pix for pix in goodpix0
                            if (pix not in high_err_pix)])
    # Remove the parts where the sky subtraction was not very good
    if bad_sky is not None:
        sky_trim = np.where((loglam_gal > bad_sky[0]) |
                            ((loglam_gal > bad_sky[1]) &
                             (loglam_gal < bad_sky[2])))[0]
        goodpix = np.array([pix for pix in goodpix0
                            if (pix not in sky_trim)])
        if len(bad_sky) > 3:
            for i in range(int((len(bad_sky) - 3) / 2)):
                more_sky_trim = np.where((loglam_gal > bad_sky[2*i + 3]) &
                                         (loglam_gal < bad_sky[2*i + 4]))[0]
                goodpix = np.array([pix for pix in goodpix
                                    if (pix not in more_sky_trim)])
    # Even though these pixels have been exluded from goodpix already,
    # ppxf still messes up if there are inf values in the noise spectrum,
    # so just replace these with the median.
    ns_clean = np.copy(ns)
    ns_clean[high_err_pix] = np.nanmedian(ns[goodpix])
    return(goodpix, ns_clean)


def determine_goodpixels(lam_gal, z):
    """
    Use ppxf_util function determine_goodpixels to mask emission lines,
    plus cover up the skyline in the region around 5557 angstroms and cut
    the crappy end of the spectrum.

    Parameters
    ----------
    lam_gal : :obj:'~numpy.ndarray'
        wavelength of the spectrum.

    z : float
        redshift of the relevant object.
    """
    # The observed (i.e not redshift corrected: must undo correction),
    # np.log_e(wavelength) for wavelength in Angstroms. Have already masked
    # the galaxy spectrum and the templates, so for lamRangeTemp can just put
    # some arbitarily large range so it wont further trim the edges of the
    # spectrum
    obs_lam = lam_gal * (z + 1)
    goodpix = util.determine_goodpixels(ln_lam=np.log(obs_lam),
                                        lam_range_temp=[100, 10000],
                                        redshift=z)

    # Always replace the region around 5557 (nonredshift corrected) due to the
    # sky emission line
    skyline = np.where((obs_lam > 5565) & (obs_lam < 5590))[0]

    # skyline, list(range(50)), and list(range(len(lam_gal) -30, len(lam_gal))
    # contain the indices of "bad pixels" with respect to lam_gal. However, we
    # need the indicies with respect to *GOODPIX*. The method my_list.index(x)
    # returns the index with respect to my_list of the element with value x.
    # Note: this is to remove the first 50 and last 30 pixels from the
    # spectrum. The if part takes avoids a ValueError if s is not in goodpix.
    inds = [list(goodpix).index(s) for s in list(skyline) if s in goodpix]
    goodpix_final = np.delete(goodpix, inds)

    return(goodpix_final)


def ppxf_spec(wav, spec, var, redshift, degree=12, mdegree=-1, fit_all=False,
              reddening=False, ebv=None, log_rebin=False, FWHM_gal=None,
              LRIS_blue=False, bad_sky=None):
    """
    applies ppxf to a spectrum.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength of the spectrum.

    spec : :obj:'~numpy.ndarray'
        flux array of the spectrum.

    var : :obj:'~numpy.ndarray'
        variance of the spectrum.

    redshift : float
        redshift of the relevant object.

    degree : int
        degree of the additive polynomial. Default is 12.

    mdegree : int
        degree of the multiplicative polynomial. Default is -1.

    fit_all : boolean
        if True, it will also fit the emission lines. Default is False.

    reddening : boolean
        if True, it will take into consideration the dust reddening when
        performing the fit.

    ebv : float
        E(B - V) value of the object.

    log_rebin : boolean
        if True, it will assume the data is already logarithmically rebinned.

    Returns
    -------
    pp : ppxf object
        full ppxf object containing the solution.

    linlam_gal : :obj:'~numpy.ndarray'
        wavelength array of the spectrum. It is linearly sampled.

    loglam_gal : :obj:'~numpy.ndarray'
        wavelength array of the spectrum. It is logarithmically sampled.
    """
    linflux_gal = spec
    linvar_gal = var
    linlam_gal = wav
    if LRIS_blue:
        edges = np.where((linlam_gal < 3975) | ((linlam_gal > 4925) &
                         (linlam_gal < 5610)))[0]
        linvar_gal[edges] = 0
    linlam_gal = linlam_gal / (1 + redshift)
    logflux_gal, loglam_gal, velscale = util.log_rebin(lam=[linlam_gal.min(),
                                                            linlam_gal.max()],
                                                       spec=linflux_gal)
    logvar, aa, bb = util.log_rebin(lam=[linlam_gal.min(), linlam_gal.max()],
                                    spec=linvar_gal, velscale=velscale)
    noise = np.sqrt(logvar)
    loglam_gal = np.exp(loglam_gal)
    if FWHM_gal is None:
        FWHM_gal = loglam_gal / R
    goodpix, noise = get_goodpixels_and_noise(loglam_gal, logvar, redshift,
                                              True, LRIS_blue=LRIS_blue,
                                              bad_sky=bad_sky)
    # goodpix = determine_goodpixels(loglam_gal, redshift)
    # Read the list of filenames from the E-Miles Single Stellar Population
    # library by Vazdekis (2016, MNRAS, 463, 3409) http://miles.iac.es/.
    # A subset of the library is included for this example with permission
    #
    vazdekis = glob.glob(ppxf_dir +
                         '/miles_models/EMILES_BASTI_BASE_KB_FITS/*.fits')
    pathname = ppxf_dir + '/miles_models/EMILES_BASTI_BASE_KB_FITS/' + \
                          'EMILES_BASTI_BASE_KB.npz'
    miles = lib.sps_lib(pathname, velscale,
                        {"lam": loglam_gal, "fwhm": FWHM_gal},
                        norm_range=[5070, 5950])
    stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)
    FWHM_tem = 2.51     # Vazdekis+16 spectra have a constant resolution
    #                     FWHM of 2.51A.
    velscale_ratio = 2  # adopts 2x higher spectral sampling for templates
    #                     than for galaxy
    # Extract the wavelength range and logarithmically rebin one spectrum to a
    # velocity scale 2x smaller than the SAURON galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    #
    hdu = fits.open(vazdekis[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    lam2 = h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1'])
    # The E-Miles templates span a large wavelength range. To save some
    # computation time I truncate the spectra to a similar range as the galaxy.
    good_lam = (lam2 > loglam_gal[0]/1.02) & (lam2 < loglam_gal[-1]*1.02)
    ssp, lam2 = ssp[good_lam], lam2[good_lam]
    lamRange2 = [np.min(lam2), np.max(lam2)]
    sspNew, ln_lam2 = util.log_rebin(lamRange2, ssp,
                                     velscale=velscale/velscale_ratio)[:2]
    templates = np.empty((sspNew.size, len(vazdekis)))
    ln_lam2 = np.exp(ln_lam2)
    # Convolve the whole Vazdekis library of spectral templates
    # with the quadratic difference between the SAURON and the
    # Vazdekis instrumental resolution. Logarithmically rebin
    # and store each template as a column in the array TEMPLATES.
    #
    # Quadratic sigma difference in pixels Vazdekis --> SAURON
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    #
    FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
    sigma = FWHM_dif/2.355/h2['CDELT1']  # Sigma difference in pixels
    for j, file in enumerate(vazdekis):
        hdu = fits.open(file)
        ssp = hdu[0].data
        ssp = ndimage.gaussian_filter1d(ssp[good_lam], sigma[-1])
        sspNew = util.log_rebin(lamRange2, ssp,
                                velscale=velscale/velscale_ratio)[0]
        # Normalizes templates
        templates[:, j] = (sspNew / np.median(sspNew[sspNew > 0]))
    goodPixels = util.determine_goodpixels(np.log(loglam_gal),
                                           lamRange2, 0)
    elements_to_remove = np.where(noise <= 0.)
    goodPixels = goodPixels[~np.isin(goodPixels, elements_to_remove)]
    # print(goodPixels.shape)
    # Here the actual fit starts. The best fit is plotted on the screen.
    # Gas emission lines are excluded from the pPXF fit using the GOODPIXELS
    # keyword.
    #
    c = 299792.458
    # vel = c*np.log(1 + 0)   # eq.(8) of Cappellari (2017, MNRAS)
    vel = 0
    start = [vel, 200.]  # (km/s), starting guess for [V, sigma]
    t = clock()
    # Construct a set of Gaussian emission line templates.
    # The `emission_lines` function defines the most common lines, but
    # additional lines can be included by editing the function in the
    # file ppxf_util.py.
    tie_balmer = False
    limit_dbts = False
    gas_tem, gas_names, lw = util.emission_lines(miles.ln_lam_temp,
                                                 [loglam_gal[0],
                                                  loglam_gal[-1]],
                                                 FWHM_gal[0],
                                                 tie_balmer=tie_balmer,
                                                 limit_doublets=limit_dbts)
    templates_combined = np.column_stack([stars_templates, gas_tem])
    n_temps = stars_templates.shape[1]
    # forbidden lines contain "[*]"
    n_forbidden = np.sum(["[" in a for a in gas_names])
    n_balmer = len(gas_names) - n_forbidden
    # Assign component=0 to the stellar templates, component=1 to the Balmer
    # gas emission lines templates and component=2 to the forbidden lines.
    component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden
    # gas_component=True for gas templates
    gas_component = np.array(component) > 0
    # Fit (V, sig, h3, h4) moments=4 for the stars
    # and (V, sig) moments=2 for the two gas kinematic components
    moments = [2, 2, 2]
    # Adopt the same starting value for the stars and the two gas components
    start_components = [start, start, start]
    # If the Balmer lines are tied one should allow for gas reddeining.
    # The gas_reddening can be different from the stellar one, if both are
    # fitted.
    if reddening:
        gas_reddening = ebv
    else:
        gas_reddening = 0 if tie_balmer else None
    # pp = ppxf(templates, logflux_gal, noise, velscale, start,
    #           goodpixels=goodpix, plot=True, moments=2,
    #           lam=loglam_gal, lam_temp=ln_lam2,
    #           degree=degree, velscale_ratio=velscale_ratio)
    if fit_all:
        goodpix, noise = get_goodpixels_and_noise(loglam_gal, logvar, redshift,
                                                  False, LRIS_blue=LRIS_blue,
                                                  bad_sky=bad_sky)
        pp = ppxf(templates_combined, logflux_gal, noise, velscale,
                  start_components, goodpixels=goodpix, moments=moments,
                  degree=degree, mdegree=mdegree, lam=loglam_gal,
                  lam_temp=miles.lam_temp, component=component,
                  gas_component=gas_component, gas_names=gas_names,
                  gas_reddening=gas_reddening, plot=True)
    else:
        pp = ppxf(templates, logflux_gal, noise, velscale, start,
                  goodpixels=goodpix, moments=2, degree=degree,
                  mdegree=mdegree, lam=loglam_gal, lam_temp=ln_lam2,
                  velscale_ratio=velscale_ratio, plot=True)
    return pp, linlam_gal, loglam_gal


def remove_continuum_cube(wav, cube, var_cube, redshift, LRIS_blue=False):
    """
    applies ppxf to a data cube and removes the continuum.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavlength array of the data

    cube : :obj:'~numpy.ndarray'
        data cube with flux values.

    var_cube : :obj:'~numpy.ndarray'
        variance cube.

    redshift : float
        redshift of the relevant object.

    Returns
    -------
    cube_new : :obj:'~numpy.ndarray'
        cube with subtracted continuum

    var_cube : :obj:'~numpy.ndarray'
        variance cube.
    """
    cube_new = cube
    for i in range(cube.shape[1]):
        for j in range(cube.shape[2]):
            spec = cube[:, i, j]
            var = var_cube[:, i, j]
            snr = useful.snr_continuum(wav, spec, var,
                                       [4450*(1+redshift), 4550*(1+redshift)])
            if snr < 5.:
                continue
            pp, linlam_gal, loglam_gal = ppxf_spec(wav, spec, var, redshift,
                                                   LRIS_blue=LRIS_blue)
            new_spec = remove_continuum_spec(spec, pp, linlam_gal, loglam_gal)
            cube_new[:, i, j] = new_spec
    return cube_new, var_cube


def remove_continuum_spec(spec, pp, linlam_gal, loglam_gal, save=True,
                          em_lines=True):
    """
    removes the continuum in a single spectrum.

    Parameters
    ----------
    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    pp : ppxf object
        ppxf object including the continuum solution for this particular
        spectrum.

    linlam_gal : :obj:'~numpy.ndarray'
        wavelength array of the data. It must be linearly sampled.

    loglam_gal : :obj:'~numpy.ndarray'
        wavelength array of the data. It must be logarithmically sampled.

    save : boolean
        if True, it will save the ppxf object with the continuum solution for
        this particular spectrum.

    Returns
    -------
    new_data : :obj:'~numpy.ndarray'
        new spectrum with subtracted continuum.
    """
    if em_lines:
        continuum = pp.bestfit - pp.gas_bestfit
    else:
        continuum = pp.bestfit
    continuum = np.interp(linlam_gal, loglam_gal, continuum)
    new_data = spec - continuum
    if save:
        pickle.dump(pp, open("ppxf_example.p", "wb"))
    return new_data


def get_velocity(wav, spec, var, redshift, LRIS_blue=False):
    """
    gets the stellar velocity calculated by ppxf.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the spectrum.

    spec : :obj:'~numpy.ndarray'
        flux array of the spectrum.

    var : :obj:'~numpy.ndarray'
        variance array of the spectrum.

    redshift : float
        redshift of the relevant object.

    Returns
    -------
    velocity : float
        stellar velocity calculated by ppxf.
    """
    pp, linlam_gal, loglam_gal = ppxf_spec(wav, spec, var, redshift,
                                           LRIS_blue=LRIS_blue)
    velocity = pp.sol[0]
    disp = pp.sol[1]
    return velocity, disp


def get_age_metal(wav, spec, var, redshift, LRIS_blue=False):
    """
    gets stellar age and metallicity calculated by ppxf.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the spectrum.

    spec : :obj:'~numpy.ndarray'
        flux array of the spectrum.

    var : :obj:'~numpy.ndarray'
        variance array of the spectrum

    redshift : float
        redshift of the relevant object.

    Returns
    -------
    age : float
        stellar age calculated by ppxf

    metal : float
        stellar metallicity calculated by ppxf.
    """
    pp, linlam_gal, loglam_gal = ppxf_spec(wav, spec, var, redshift, degree=-1,
                                           mdegree=10, LRIS_blue=LRIS_blue)
    FWHM_gal = loglam_gal / R
    pathname = ppxf_dir + '/miles_models/EMILES_BASTI_BASE_KB_FITS/' + \
                          'EMILES_BASTI_BASE_KB.npz'
    miles = lib.sps_lib(pathname, pp.velscale, FWHM_gal[0],
                        norm_range=[5070, 5950])
    reg_dim = miles.templates.shape[1:]
    light_weights = pp.weights[~pp.gas_component]
    light_weights = light_weights.reshape(reg_dim)
    light_weights /= light_weights.sum()
    age, metal = miles.mean_age_metal(light_weights)
    return age, metal
