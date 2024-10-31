#!/usr/bin/env python

#####################################
# Functions to analise the cubes
#####################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy import stats
from scipy.stats import bootstrap
from scipy.optimize import curve_fit
from astropy import constants as const
from uncertainties import ufloat
from uncertainties import umath
from uncertainties import unumpy as unp

import useful
import display
from constants_wavelengths import *

# Define constants

c = const.c.to('km/s').value

# air

wl_air = np.array([dict_wav_air['OII_l'], dict_wav_air['OII_r'],
                   dict_wav_air['Hgamma'], dict_wav_air['Hbeta'],
                   dict_wav_air['OIII_l'], dict_wav_air['OIII_r']])

wratio2_air = wl_air[0]/wl_air[1]
wratio3_air = wl_air[0]/wl_air[2]
wratio4_air = wl_air[0]/wl_air[3]
wratio5_air = wl_air[0]/wl_air[4]
wratio6_air = wl_air[0]/wl_air[5]

wl_h_air = np.array([dict_wav_air['Hgamma'], dict_wav_air['Hbeta']])

wratioh_air = wl_h_air[0] / wl_h_air[1]

wl_bpt_air = np.array([dict_wav_air['OIII_r'], dict_wav_air['Hbeta'],
                       dict_wav_air['NII'], dict_wav_air['Halpha']])

bptratio2_air = wl_bpt_air[0]/wl_bpt_air[1]
bptratio3_air = wl_bpt_air[0]/wl_bpt_air[2]
bptratio4_air = wl_bpt_air[0]/wl_bpt_air[3]

wl_abs_air = np.array([dict_wav_air['MgII_l'], dict_wav_air['MgII_r'],
                       dict_wav_air['MgI'], dict_wav_air['FeII_l'],
                       dict_wav_air['FeII_r'], dict_wav_air['CaII_H'],
                       dict_wav_air['CaII_K']])

mgii_ratio_air = wl_abs_air[0]/wl_abs_air[1]
feii_ratio_air = wl_abs_air[3]/wl_abs_air[4]
caii_ratio_air = wl_abs_air[5]/wl_abs_air[6]

wratio1_abs_air = wl_abs_air[0]/wl_abs_air[1]
wratio2_abs_air = wl_abs_air[0]/wl_abs_air[2]
wratio3_abs_air = wl_abs_air[0]/wl_abs_air[3]
wratio4_abs_air = wl_abs_air[0]/wl_abs_air[4]
wratio5_abs_air = wl_abs_air[0]/wl_abs_air[5]
wratio6_abs_air = wl_abs_air[0]/wl_abs_air[6]

# vacuumn

wl_vac = np.array([dict_wav_vac['OII_l'], dict_wav_vac['OII_r'],
                   dict_wav_vac['Hgamma'], dict_wav_vac['Hbeta'],
                   dict_wav_vac['OIII_l'], dict_wav_vac['OIII_r']])

wratio2_vac = wl_vac[0]/wl_vac[1]
wratio3_vac = wl_vac[0]/wl_vac[2]
wratio4_vac = wl_vac[0]/wl_vac[3]
wratio5_vac = wl_vac[0]/wl_vac[4]
wratio6_vac = wl_vac[0]/wl_vac[5]

wl_h_vac = np.array([dict_wav_vac['Hgamma'], dict_wav_vac['Hbeta']])

wratioh_vac = wl_h_vac[0] / wl_h_vac[1]

wl_bpt_vac = np.array([dict_wav_vac['OIII_r'], dict_wav_vac['Hbeta'],
                       dict_wav_vac['NII'], dict_wav_vac['Halpha']])

bptratio2_vac = wl_bpt_vac[0]/wl_bpt_vac[1]
bptratio3_vac = wl_bpt_vac[0]/wl_bpt_vac[2]
bptratio4_vac = wl_bpt_vac[0]/wl_bpt_vac[3]

wl_abs_vac = np.array([dict_wav_vac['MgII_l'], dict_wav_vac['MgII_r'],
                       dict_wav_vac['MgI'], dict_wav_vac['FeII_l'],
                       dict_wav_vac['FeII_r'], dict_wav_vac['CaII_H'],
                       dict_wav_vac['CaII_K']])

mgii_ratio_vac = wl_abs_vac[0]/wl_abs_vac[1]
feii_ratio_vac = wl_abs_vac[3]/wl_abs_vac[4]
caii_ratio_vac = wl_abs_vac[5]/wl_abs_vac[6]

wratio1_abs_vac = wl_abs_vac[0]/wl_abs_vac[1]
wratio2_abs_vac = wl_abs_vac[0]/wl_abs_vac[2]
wratio3_abs_vac = wl_abs_vac[0]/wl_abs_vac[3]
wratio4_abs_vac = wl_abs_vac[0]/wl_abs_vac[4]
wratio5_abs_vac = wl_abs_vac[0]/wl_abs_vac[5]
wratio6_abs_vac = wl_abs_vac[0]/wl_abs_vac[6]

gal_names = {'00045': 'Face-on', '00046': 'Edge-on', '00047': 'Merging',
             'G7-J1216mosaic': 'G7-J1216'}
# Create model


def gaussian_model(x, *params):
    """
    Calculates a Gaussian based on the provided x values and params.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A
        (amplitude), mu (centroid) and sigma (standard deviation).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the gaussian.
    """
    A, mu, sigma = params
    y = A * stats.norm.pdf(x, mu, sigma) * np.sqrt(2 * np.pi) * sigma
    return y


def hgamma_hbeta_model_air(x, *params):
    """
    Calculates a model with two gaussians separated by the wavelength ratio of
    Hgamma and Hbeta in air.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A1
        (amplitude of the Hgamma line), mu (centroid of Hgamma), sigma
        (standard deviation) and A2 (amplitude of the Hbeta line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu1, sigma, A2 = params
    gaussian1 = gaussian_model(x, A1, mu1, sigma)
    gaussian2 = gaussian_model(x, A2, mu1/wratioh_air, sigma)
    y = gaussian1 + gaussian2
    return y


def hgamma_hbeta_model_vac(x, *params):
    """
    Calculates a model with two gaussians separated by the wavelength ratio of
    Hgamma and Hbeta in vacuumn.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A1
        (amplitude of the Hgamma line), mu (centroid of Hgamma), sigma
        (standard deviation) and A2 (amplitude of the Hbeta line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu1, sigma, A2 = params
    gaussian1 = gaussian_model(x, A1, mu1, sigma)
    gaussian2 = gaussian_model(x, A2, mu1/wratioh_vac, sigma)
    y = gaussian1 + gaussian2
    return y


def n_hgamma_hbeta_model_vac(x, *params):
    """
    calculates a model with n pairs of Hbeta and Hgamma lines. The lines will
    be separated by their wavelength ratio in the vacuumn.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussians

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A1
        (amplitude of the Hgamma line), mu (centroid of Hgamma), sigma
        (standard deviation) and A2 (amplitude of the Hbeta line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    n = int(len(params) / 4)
    y = 0
    for i in range(n):
        A1 = params[4*i]
        mu = params[4*i + 1]
        sigma = params[4*i + 2]
        A2 = params[4*i + 3]
        comp = [A1, mu, sigma, A2]
        y = y + hgamma_hbeta_model_vac(x, *comp)
    return y


def n_hgamma_hbeta_model_air(x, *params):
    """
    calculates a model with n pairs of Hbeta and Hgamma lines. The lines will
    be separated by their wavelength ratio in the air.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussians

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A1
        (amplitude of the Hgamma line), mu (centroid of Hgamma), sigma
        (standard deviation) and A2 (amplitude of the Hbeta line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    n = int(len(params) / 4)
    y = 0
    for i in range(n):
        A1 = params[4*i]
        mu = params[4*i + 1]
        sigma = params[4*i + 2]
        A2 = params[4*i + 3]
        comp = [A1, mu, sigma, A2]
        y = y + hgamma_hbeta_model_air(x, *comp)
    return y


def four_gaussian_model_air(x, *params):
    """
    Calculates a model with four gaussians: the [O II] doublet, Hbeta and
    [O III]. The centroid positions are fixed at the wavelength ratio of the
    each line with respect to [O II]3727 in air.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A1
        (amplitude of the [O II]3727 line), mu (centroid of the [O II]3727
        line), sigma1 (standard deviation of [O II]), alpha (amplitude ratio of
        [O II]3727 and [O II]3729 lines), A3 (amplitude of the Hbeta line),
        sigma3 (standard deviation of Hbeta), A4 (amplitude of the [O III]
        line) and sigma4 (standard deviation of the [O III] line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu, sigma1, alpha, A3, sigma3, A4, sigma4 = params
    gaussian1 = gaussian_model(x, A1, mu, sigma1)
    gaussian2 = gaussian_model(x, alpha*A1, mu/wratio2_air, sigma1)
    gaussian3 = gaussian_model(x, A3, mu/wratio4_air, sigma3)
    gaussian4 = gaussian_model(x, A4, mu/wratio6_air, sigma4)
    y = gaussian1 + gaussian2 + gaussian3 + gaussian4
    return y


def four_gaussian_model_vac(x, *params):
    """
    Calculates a model with four gaussians: the [O II] doublet, Hbeta and
    [O III]. The centroid positions are fixed at the wavelength ratio of the
    each line with respect to [O II]3727 in vacuumn.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A1
        (amplitude of the [O II]3727 line), mu (centroid of the [O II]3727
        line), sigma1 (standard deviation of [O II]), alpha (amplitude ratio of
        [O II]3727 and [O II]3729 lines), A3 (amplitude of the Hbeta line),
        sigma3 (standard deviation of Hbeta), A4 (amplitude of the [O III]
        line) and sigma4 (standard deviation of the [O III] line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu, sigma1, alpha, A3, sigma3, A4, sigma4 = params
    gaussian1 = gaussian_model(x, A1, mu, sigma1)
    gaussian2 = gaussian_model(x, alpha*A1, mu/wratio2_vac, sigma1)
    gaussian3 = gaussian_model(x, A3, mu/wratio4_vac, sigma3)
    gaussian4 = gaussian_model(x, A4, mu/wratio6_vac, sigma4)
    y = gaussian1 + gaussian2 + gaussian3 + gaussian4
    return y


def bpt_model_air(x, *params):
    """
    Calculates a model with four gaussians: [O III], Hbeta, N II and Halpha.
    The centroid positions are fixed at the wavelength ratio of the each line
    with respect to [O III] in air.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A1
        (amplitude of the [O III] line), mu (centroid of the [O III] line),
        sigma1 (standard deviation of [O II]), A2 (amplitude of the Hbeta
        line), sigma2 (standard deviation of the Hbeta line), A3 (amplitude of
        the N II line), sigma3 (standard deviation of the N II line), A4
        (amplitude of the Halpha line) and sigma4 (standard deviation of the
        Halpha line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu, sigma1, A2, sigma2, A3, sigma3, A4, sigma4 = params
    gaussian1 = gaussian_model(x, A1, mu, sigma1)
    gaussian2 = gaussian_model(x, A2, mu/bptratio2_air, sigma2)
    gaussian3 = gaussian_model(x, A3, mu/bptratio3_air, sigma3)
    gaussian4 = gaussian_model(x, A4, mu/bptratio4_air, sigma4)
    y = gaussian1 + gaussian2 + gaussian3 + gaussian4
    return y


def bpt_model_vac(x, *params):
    """
    Calculates a model with four gaussians: [O III], Hbeta, N II and Halpha.
    The centroid positions are fixed at the wavelength ratio of the each line
    with respect to [O III] in vacuumn.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A1
        (amplitude of the [O III] line), mu (centroid of the [O III] line),
        sigma1 (standard deviation of [O II]), A2 (amplitude of the Hbeta
        line), sigma2 (standard deviation of the Hbeta line), A3 (amplitude of
        the N II line), sigma3 (standard deviation of the N II line), A4
        (amplitude of the Halpha line) and sigma4 (standard deviation of the
        Halpha line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu, sigma1, A2, sigma2, A3, sigma3, A4, sigma4 = params
    gaussian1 = gaussian_model(x, A1, mu, sigma1)
    gaussian2 = gaussian_model(x, A2, mu/bptratio2_vac, sigma2)
    gaussian3 = gaussian_model(x, A3, mu/bptratio3_vac, sigma3)
    gaussian4 = gaussian_model(x, A4, mu/bptratio4_vac, sigma4)
    y = gaussian1 + gaussian2 + gaussian3 + gaussian4
    return y


def two_gaussian_model_air(x, *params):
    """
    Calculates a model with the [O II]3727,3729 doublet. The centroid positions
    are fixed at the wavelength ratio of the each line with respect to
    [O II]3727 in air.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A (amplitude
        of the [O II]3727 line), mu (centroid of the [O II]3727 line), sigma
        (standard deviation of [O II]) and alpha (amplitude ratio of [O II]3727
        and [O II]3729 lines).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A, mu, sigma, alpha = params
    gaussian1 = gaussian_model(x, A, mu, sigma)
    gaussian2 = gaussian_model(x, alpha*A, mu/wratio2_air, sigma)
    y = gaussian1 + gaussian2
    return y


def two_gaussian_model_vac(x, *params):
    """
    Calculates a model with the [O II]3727,3729 doublet. The centroid positions
    are fixed at the wavelength ratio of the each line with respect to
    [O II]3727 in vacuumn.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A (amplitude
        of the [O II]3727 line), mu (centroid of the [O II]3727 line), sigma
        (standard deviation of [O II]) and alpha (amplitude ratio of [O II]3727
        and [O II]3729 lines).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A, mu, sigma, alpha = params
    gaussian1 = gaussian_model(x, A, mu, sigma)
    gaussian2 = gaussian_model(x, alpha*A, mu/wratio2_vac, sigma)
    y = gaussian1 + gaussian2
    return y


def mgii_model_air(x, *params):
    """
    Calculates a model with a single Mg II 2796,2803 doublet. The centroid
    positions are fixed at the wavelength ratio of each line respect Mg II 2796
    in air.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A1 (amplitude
        of the Mg II 2796 line), mu (centroid of the Mg II 2796 line), sigma
        (standard deviation of the Mg II lines), and A2 (amplitude of the Mg II
        2803 line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu, sigma, A2 = params
    gaussian1 = gaussian_model(x, A1, mu, sigma)
    gaussian2 = gaussian_model(x, A2, mu/mgii_ratio_air, sigma)
    y = 1 - gaussian1 - gaussian2
    return y


def mgii_model_vac(x, *params):
    """
    Calculates a model with a single Mg II 2796,2803 doublet. The centroid
    positions are fixed at the wavelength ratio of each line respect Mg II 2796
    in vacuum.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A1 (amplitude
        of the Mg II 2796 line), mu (centroid of the Mg II 2796 line), sigma
        (standard deviation of the Mg II lines), and A2 (amplitude of the Mg II
        2803 line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu, sigma, A2 = params
    gaussian1 = gaussian_model(x, A1, mu, sigma)
    gaussian2 = gaussian_model(x, A2, mu/mgii_ratio_vac, sigma)
    y = 1 - gaussian1 - gaussian2
    return y


def n_mgii_model_air(x, *params):
    """
    Calculates a model with a n Mg II 2796,2803 doublets. The centroid
    positions are fixed at the wavelength ratio of each line respect Mg II 2796
    in air.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        parameters of each gaussian. They must be in the following order: A1
        (amplitude of the Mg II 2796 line), mu (centroid of the Mg II 2796
        line), sigma (standard deviation of the Mg II lines), and A2 (amplitude
        of the Mg II 2803 line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    n = int(len(params) / 4)
    y = 1
    for i in range(n):
        A1 = params[4*i]
        mu = params[4*i + 1]
        sigma = params[4*i + 2]
        A2 = params[4*i + 3]
        gaussian1 = gaussian_model(x, A1, mu, sigma)
        gaussian2 = gaussian_model(x, A2, mu/mgii_ratio_air, sigma)
        y = y - gaussian1 - gaussian2
    return y


def n_mgii_model_vac(x, *params):
    """
    Calculates a model with a n Mg II 2796,2803 doublets. The centroid
    positions are fixed at the wavelength ratio of each line respect Mg II 2796
    in vacuumn.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        parameters of each gaussian. They must be in the following order: A1
        (amplitude of the Mg II 2796 line), mu (centroid of the Mg II 2796
        line), sigma (standard deviation of the Mg II lines), and A2 (amplitude
        of the Mg II 2803 line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    n = int(len(params) / 4)
    y = 1
    for i in range(n):
        A1 = params[4*i]
        mu = params[4*i + 1]
        sigma = params[4*i + 2]
        A2 = params[4*i + 3]
        gaussian1 = gaussian_model(x, A1, mu, sigma)
        gaussian2 = gaussian_model(x, A2, mu/mgii_ratio_vac, sigma)
        y = y - gaussian1 - gaussian2
    return y


def two_emissions_model(x, *params):
    """
    Calculates a model with two emission lines. These lines are not tied in
    velocity nor dispersion.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values of the model.

    *params : :obj:'~numpy.ndarray'
        parameters of each gaussian. They must be in the following order: A1
        (amplitude of the first line), mu1 (centroid of the first line), sigma1
        (standard deviation of the first line), A2 (amplitude of the second
        line), mu2 (centroid of the second line), and sigma2 (standard
        deviation of the second line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu1, sigma1, A2, mu2, sigma2 = params
    gaussian1 = gaussian_model(x, A1, mu1, sigma1)
    gaussian2 = gaussian_model(x, A2, mu2, sigma2)
    y = gaussian1 + gaussian2
    return y


def five_abs_model_vac(x, *params):
    """
    Calculates a model with Mg II 2796,2803, Mg I, and Fe II 2586,2600 lines.
    All the lines have the same velocity and different lines of the same
    transition have the same standard deviation. The line ratios are in
    vacuumn.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values of the model

    *params : :obj:'~numpy.ndarray'
        parameters of the gaussians. They must be in the following order: A1
        (amplitude of the Mg II 2796 line), mu (centroid of the lines), sigma1
        (standard deviation of the Mg II lines), A2 (amplitude of the Mg II
        2803 line), A3 (amplitude of the Mg I line), sigma3 (standard deviation
        of the Mg I line), A4 (amplitude of the Fe II 2586 line), sigma4
        (standard deviation of the Fe II lines), and A5 (amplitude of the Fe II
        2600 line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu, sigma1, A2, A3, sigma3, A4, sigma4, A5 = params
    gaussian1 = gaussian_model(x, A1, mu, sigma1)
    gaussian2 = gaussian_model(x, A2, mu/mgii_ratio_vac, sigma1)
    gaussian3 = gaussian_model(x, A3, mu/wratio2_abs_vac, sigma3)
    gaussian4 = gaussian_model(x, A4, mu/wratio3_abs_vac, sigma4)
    gaussian5 = gaussian_model(x, A5, mu/wratio4_abs_vac, sigma4)
    y = 1 - gaussian1 - gaussian2 - gaussian3 - gaussian4 - gaussian5
    return y


def five_abs_model_air(x, *params):
    """
    Calculates a model with Mg II 2796,2803, Mg I, and Fe II 2586,2600 lines.
    All the lines have the same velocity and different lines of the same
    transition have the same standard deviation. The line ratios are in
    air.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values of the model

    *params : :obj:'~numpy.ndarray'
        parameters of the gaussians. They must be in the following order: A1
        (amplitude of the Mg II 2796 line), mu (centroid of the lines), sigma1
        (standard deviation of the Mg II lines), A2 (amplitude of the Mg II
        2803 line), A3 (amplitude of the Mg I line), sigma3 (standard deviation
        of the Mg I line), A4 (amplitude of the Fe II 2586 line), sigma4
        (standard deviation of the Fe II lines), and A5 (amplitude of the Fe II
        2600 line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu, sigma1, A2, A3, sigma3, A4, sigma4, A5 = params
    gaussian1 = gaussian_model(x, A1, mu, sigma1)
    gaussian2 = gaussian_model(x, A2, mu/mgii_ratio_air, sigma1)
    gaussian3 = gaussian_model(x, A3, mu/wratio2_abs_air, sigma3)
    gaussian4 = gaussian_model(x, A4, mu/wratio3_abs_air, sigma4)
    gaussian5 = gaussian_model(x, A5, mu/wratio4_abs_air, sigma4)
    y = 1 - gaussian1 - gaussian2 - gaussian3 - gaussian4 - gaussian5
    return y


def n_five_abs_model_vac(x, *params):
    """
    Calculates a model with n sets of Mg II 2796,2803, Mg I, and Fe II
    2586,2600 lines. All the lines have the same velocity and different lines
    of the same transition have the same standard deviation. The line ratios
    are in vacuumn.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values of the model

    *params : :obj:'~numpy.ndarray'
        parameters of the gaussians. They must be in the following order: A1
        (amplitude of the Mg II 2796 line), mu (centroid of the lines), sigma1
        (standard deviation of the Mg II lines), A2 (amplitude of the Mg II
        2803 line), A3 (amplitude of the Mg I line), sigma3 (standard deviation
        of the Mg I line), A4 (amplitude of the Fe II 2586 line), sigma4
        (standard deviation of the Fe II lines), and A5 (amplitude of the Fe II
        2600 line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    n = int(len(params) / 9)
    y = 1
    for i in range(n):
        A1 = params[9*i]
        mu = params[9*i + 1]
        sigma1 = params[9*i + 2]
        A2 = params[9*i + 3]
        A3 = params[9*i + 4]
        sigma3 = params[9*i + 5]
        A4 = params[9*i + 6]
        sigma4 = params[9*i + 7]
        A5 = params[9*i + 8]
        comp = [A1, mu, sigma1, A2, A3, sigma3, A4, sigma4, A5]
        y = y + five_abs_model_vac(x, *comp) - 1
    return y


def n_five_abs_model_air(x, *params):
    """
    Calculates a model with n sets of Mg II 2796,2803, Mg I, and Fe II
    2586,2600 lines. All the lines have the same velocity and different lines
    of the same transition have the same standard deviation. The line ratios
    are in air.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values of the model

    *params : :obj:'~numpy.ndarray'
        parameters of the gaussians. They must be in the following order: A1
        (amplitude of the Mg II 2796 line), mu (centroid of the lines), sigma1
        (standard deviation of the Mg II lines), A2 (amplitude of the Mg II
        2803 line), A3 (amplitude of the Mg I line), sigma3 (standard deviation
        of the Mg I line), A4 (amplitude of the Fe II 2586 line), sigma4
        (standard deviation of the Fe II lines), and A5 (amplitude of the Fe II
        2600 line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    n = int(len(params) / 9)
    y = 1
    for i in range(n):
        A1 = params[9*i]
        mu = params[9*i + 1]
        sigma1 = params[9*i + 2]
        A2 = params[9*i + 3]
        A3 = params[9*i + 4]
        sigma3 = params[9*i + 5]
        A4 = params[9*i + 6]
        sigma4 = params[9*i + 7]
        A5 = params[9*i + 8]
        comp = [A1, mu, sigma1, A2, A3, sigma3, A4, sigma4, A5]
        y = y + five_abs_model_air(x, *comp) - 1
    return y


def six_abs_model_vac(x, *params):
    """
    Calculates a model with six absorption lines: Mg II 2796,2803,
    Fe II 2586,2600 Ca II H&K. All the lines have the same velocities.
    Additionally, the Mg II and Fe II have the same velocity dispersions. The
    line ratios are in vacuumn.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values of the model.

    *params : :obj:'~numpy.ndarray'
        parameters of the gaussians. They must be in the following order: A1
        (amplitude of the Mg II 2796 line), mu (centroid of the lines), sigma1
        (standard deviation of the Mg II and Fe II lines), A2 (amplitude of the
        Mg II 2803), A3 (amplitude of the Fe II 2586), A4 (amplitude of the
        Fe II 2600), A5 (amplitude of the Ca II H line), sigma5 (standard
        deviation of the Ca II lines), and A6 (amplitude of the Ca II K line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu, sigma1, A2, A3, A4, A5, sigma5, A6 = params
    gaussian1 = gaussian_model(x, A1, mu, sigma1)
    gaussian2 = gaussian_model(x, A2, mu/mgii_ratio_vac, sigma1)
    gaussian3 = gaussian_model(x, A3, mu/wratio3_abs_vac, sigma1)
    gaussian4 = gaussian_model(x, A4, mu/wratio4_abs_vac, sigma1)
    gaussian5 = gaussian_model(x, A5, mu/wratio5_abs_vac, sigma5)
    gaussian6 = gaussian_model(x, A6, mu/wratio6_abs_vac, sigma5)
    y = 1 - gaussian1 - gaussian2 - gaussian3 - gaussian4 - gaussian5 \
        - gaussian6
    return y


def six_abs_model_air(x, *params):
    """
    Calculates a model with six absorption lines: Mg II 2796,2803,
    Fe II 2586,2600 Ca II H&K. All the lines have the same velocities.
    Additionally, the Mg II and Fe II have the same velocity dispersions. The
    line ratios are in air.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values of the model.

    *params : :obj:'~numpy.ndarray'
        parameters of the gaussians. They must be in the following order: A1
        (amplitude of the Mg II 2796 line), mu (centroid of the lines), sigma1
        (standard deviation of the Mg II and Fe II lines), A2 (amplitude of the
        Mg II 2803), A3 (amplitude of the Fe II 2586), A4 (amplitude of the
        Fe II 2600), A5 (amplitude of the Ca II H line), sigma5 (standard
        deviation of the Ca II lines), and A6 (amplitude of the Ca II K line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu, sigma1, A2, A3, A4, A5, sigma5, A6 = params
    gaussian1 = gaussian_model(x, A1, mu, sigma1)
    gaussian2 = gaussian_model(x, A2, mu/mgii_ratio_air, sigma1)
    gaussian3 = gaussian_model(x, A3, mu/wratio3_abs_air, sigma1)
    gaussian4 = gaussian_model(x, A4, mu/wratio4_abs_air, sigma1)
    gaussian5 = gaussian_model(x, A5, mu/wratio5_abs_air, sigma5)
    gaussian6 = gaussian_model(x, A6, mu/wratio6_abs_air, sigma5)
    y = 1 - gaussian1 - gaussian2 - gaussian3 - gaussian4 - gaussian5 \
        - gaussian6
    return y


def n_six_abs_model_vac(x, *params):
    """
    Calculates a model with n sets of six lines: Mg II 2796,2803,
    Fe II 2586,2600 and Ca II H&K. All the lines on each set have the same
    components. Additionally, the Mg II and Fe II have the same velocity
    dispersions. The line ratios are in vacuumn.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values of the model.

    *params : :obj:'~numpy.ndarray'
        parameters of the gaussians. They must be in the following order: A1
        (amplitude of the Mg II 2796 line), mu (centroid of the lines), sigma1
        (standard deviation of the Mg II and Fe II lines), A2 (amplitude of the
        Mg II 2803), A3 (amplitude of the Fe II 2586), A4 (amplitude of the
        Fe II 2600), A5 (amplitude of the Ca II H line), sigma5 (standard
        deviation of the Ca II lines), and A6 (amplitude of the Ca II K line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    n = int(len(params) / 9)
    y = 1
    for i in range(n):
        A1 = params[9*i]
        mu = params[9*i + 1]
        sigma1 = params[9*i + 2]
        A2 = params[9*i + 3]
        A3 = params[9*i + 4]
        A4 = params[9*i + 5]
        A5 = params[9*i + 6]
        sigma5 = params[9*i + 7]
        A6 = params[9*i + 8]
        comp = [A1, mu, sigma1, A2, A3, A4, A5, sigma5, A6]
        y = y + five_abs_model_vac(x, *comp) - 1
    return y


def n_six_abs_model_air(x, *params):
    """
    Calculates a model with n sets of six lines: Mg II 2796,2803,
    Fe II 2586,2600 and Ca II H&K. All the lines on each set have the same
    components. Additionally, the Mg II and Fe II have the same velocity
    dispersions. The line ratios are in air.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values of the model.

    *params : :obj:'~numpy.ndarray'
        parameters of the gaussians. They must be in the following order: A1
        (amplitude of the Mg II 2796 line), mu (centroid of the lines), sigma1
        (standard deviation of the Mg II and Fe II lines), A2 (amplitude of the
        Mg II 2803), A3 (amplitude of the Fe II 2586), A4 (amplitude of the
        Fe II 2600), A5 (amplitude of the Ca II H line), sigma5 (standard
        deviation of the Ca II lines), and A6 (amplitude of the Ca II K line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    n = int(len(params) / 9)
    y = 1
    for i in range(n):
        A1 = params[9*i]
        mu = params[9*i + 1]
        sigma1 = params[9*i + 2]
        A2 = params[9*i + 3]
        A3 = params[9*i + 4]
        A4 = params[9*i + 5]
        A5 = params[9*i + 6]
        sigma5 = params[9*i + 7]
        A6 = params[9*i + 8]
        comp = [A1, mu, sigma1, A2, A3, A4, A5, sigma5, A6]
        y = y + five_abs_model_air(x, *comp) - 1
    return y


def mgii_feii_model_air(x, *params):
    """
    Calculates a model with four absorption lines: Mg II 2796,2803 and
    Fe II 2586,2600. All the lines have the same velocities and velocity
    dispersions. The line ratios are in air.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values of the model.

    *params : :obj:'~numpy.ndarray'
        parameters of the gaussians. They must be in the following order: A1
        (amplitude of the Mg II 2796 line), mu (centroid of the lines), sigma1
        (standard deviation of the Mg II and Fe II lines), A2 (amplitude of the
        Mg II 2803), A3 (amplitude of the Fe II 2586) and A4 (amplitude of the
        Fe II 2600).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu, sigma, A2, A3, A4 = params
    gaussian1 = gaussian_model(x, A1, mu, sigma)
    gaussian2 = gaussian_model(x, A2, mu/mgii_ratio_air, sigma)
    gaussian3 = gaussian_model(x, A3, mu/wratio3_abs_air, sigma)
    gaussian4 = gaussian_model(x, A4, mu/wratio4_abs_air, sigma)
    y = 1 - gaussian1 - gaussian2 - gaussian3 - gaussian4
    return y


def mgii_feii_model_vac(x, *params):
    """
    Calculates a model with four absorption lines: Mg II 2796,2803 and
    Fe II 2586,2600. All the lines have the same velocities and velocity
    dispersions. The line ratios are in vacuumn.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values of the model.

    *params : :obj:'~numpy.ndarray'
        parameters of the gaussians. They must be in the following order: A1
        (amplitude of the Mg II 2796 line), mu (centroid of the lines), sigma1
        (standard deviation of the Mg II and Fe II lines), A2 (amplitude of the
        Mg II 2803), A3 (amplitude of the Fe II 2586) and A4 (amplitude of the
        Fe II 2600).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu, sigma, A2, A3, A4 = params
    gaussian1 = gaussian_model(x, A1, mu, sigma)
    gaussian2 = gaussian_model(x, A2, mu/mgii_ratio_vac, sigma)
    gaussian3 = gaussian_model(x, A3, mu/wratio3_abs_vac, sigma)
    gaussian4 = gaussian_model(x, A4, mu/wratio4_abs_vac, sigma)
    y = 1 - gaussian1 - gaussian2 - gaussian3 - gaussian4
    return y


def n_mgii_feii_model_air(x, *params):
    """
    Calculates a model with n sets of four absorption lines: Mg II 2796,2803
    and Fe II 2586,2600. All the lines in the set have the same velocities and
    velocity dispersions. The line ratios are in air.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values of the model.

    *params : :obj:'~numpy.ndarray'
        parameters of the gaussians. They must be in the following order: A1
        (amplitude of the Mg II 2796 line), mu (centroid of the lines), sigma1
        (standard deviation of the Mg II and Fe II lines), A2 (amplitude of the
        Mg II 2803), A3 (amplitude of the Fe II 2586) and A4 (amplitude of the
        Fe II 2600).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    n = int(len(params) / 6)
    y = 1
    for i in range(n):
        A1 = params[6*i]
        mu = params[6*i + 1]
        sigma = params[6*i + 2]
        A2 = params[6*i + 3]
        A3 = params[6*i + 4]
        A4 = params[6*i + 5]
        comp = [A1, mu, sigma, A2, A3, A4]
        y = y + mgii_feii_model_air(x, *comp) - 1
    return y


def n_mgii_feii_model_vac(x, *params):
    """
    Calculates a model with n sets of four absorption lines: Mg II 2796,2803
    and Fe II 2586,2600. All the lines in the set have the same velocities and
    velocity dispersions. The line ratios are in vacuumn.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values of the model.

    *params : :obj:'~numpy.ndarray'
        parameters of the gaussians. They must be in the following order: A1
        (amplitude of the Mg II 2796 line), mu (centroid of the lines), sigma1
        (standard deviation of the Mg II and Fe II lines), A2 (amplitude of the
        Mg II 2803), A3 (amplitude of the Fe II 2586) and A4 (amplitude of the
        Fe II 2600).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    n = int(len(params) / 6)
    y = 1
    for i in range(n):
        A1 = params[6*i]
        mu = params[6*i + 1]
        sigma = params[6*i + 2]
        A2 = params[6*i + 3]
        A3 = params[6*i + 4]
        A4 = params[6*i + 5]
        comp = [A1, mu, sigma, A2, A3, A4]
        y = y + mgii_feii_model_vac(x, *comp) - 1
    return y


def feii_air(x, *params):
    """
    Calculates a model with a single Fe II 2586,2600 doublet. The centroid
    positions are fixed at the wavelength ratio of each line respect Fe II 2586
    in air.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A1 (amplitude
        of the Fe II 2586 line), mu (centroid of the Fe II 2600 line), sigma
        (standard deviation of the Fe II lines), and A2 (amplitude of the Fe II
        2600 line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu, sigma, A2 = params
    gaussian1 = gaussian_model(x, A1, mu, sigma)
    gaussian2 = gaussian_model(x, A2, mu/feii_ratio_air, sigma)
    y = 1 - gaussian1 - gaussian2
    return y


def feii_vac(x, *params):
    """
    Calculates a model with a single Fe II 2586,2600 doublet. The centroid
    positions are fixed at the wavelength ratio of each line respect Fe II 2586
    in vacuumn.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A1 (amplitude
        of the Fe II 2586 line), mu (centroid of the Fe II 2600 line), sigma
        (standard deviation of the Fe II lines), and A2 (amplitude of the Fe II
        2600 line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu, sigma, A2 = params
    gaussian1 = gaussian_model(x, A1, mu, sigma)
    gaussian2 = gaussian_model(x, A2, mu/feii_ratio_vac, sigma)
    y = 1 - gaussian1 - gaussian2
    return y


def caii_air(x, *params):
    """
    Calculates a model with a single Ca II H&K doublet. The centroid
    positions are fixed at the wavelength ratio of each line respect Ca II H
    in air.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A1 (amplitude
        of the Ca II H line), mu (centroid of the Ca II H line), sigma
        (standard deviation of the Ca II lines), and A2 (amplitude of the Ca II
        K line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu, sigma, A2 = params
    gaussian1 = gaussian_model(x, A1, mu, sigma)
    gaussian2 = gaussian_model(x, A2, mu/caii_ratio_air, sigma)
    y = 1 - gaussian1 - gaussian2
    return y


def caii_vac(x, *params):
    """
    Calculates a model with a single Ca II H&K doublet. The centroid
    positions are fixed at the wavelength ratio of each line respect Ca II H
    in vacuumn.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values for the gaussian.

    *params : :obj:'~numpy.ndarray'
        gaussian parameters. They must be in the following order: A1 (amplitude
        of the Ca II H line), mu (centroid of the Ca II H line), sigma
        (standard deviation of the Ca II lines), and A2 (amplitude of the Ca II
        K line).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model.
    """
    A1, mu, sigma, A2 = params
    gaussian1 = gaussian_model(x, A1, mu, sigma)
    gaussian2 = gaussian_model(x, A2, mu/caii_ratio_vac, sigma)
    y = 1 - gaussian1 - gaussian2
    return y

# apply curve_fit


def em_fitting(wav, spec, model, p0, bounds, plot=False, var=None, coord=None):
    """
    Applies curve_fit using a given model of emission lines.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    model : function
        model to use in the fitting.

    p0 : :obj:'~numpy.ndarray'
        initial guess for curve_fit.

    bounds : :obj:'~numpy.ndarray'
        lower and upper bounds for the different parameters of the model.

    plot : boolean
        if True, the fit will be plotted. Default is False.

    var : :obj:'~numpy.ndarray'
        variance array of the data. If not None, the variance will be
        considered while doing the fit. Default is None.

    coord : 2-tuple of int
        spaxel coordinates. Only used if plot is True. If not None, the output
        plot will include the spaxel coordinates in the name of the plot.
        Default is None.

    Returns
    -------
    popt : :obj:'~numpy.ndarray'
        optimal parameters found by curve_fit. If the fit is unsuccesful, it
        will return 0.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix calculated by curve_fit. If the fit is unsuccesful,
        it will return 0.
    """
    try:
        if var is None:
            popt, pcov = curve_fit(model, wav, spec, p0=p0, bounds=bounds)
        else:
            popt, pcov = curve_fit(model, wav, spec, p0=p0, bounds=bounds,
                                   sigma=np.sqrt(var))
        return popt, pcov
    except RuntimeError:
        print('Fit failed')
        return 0, 0


def two_gaussian_fitting(wav, spec, redshift, plot=False, var=None,
                         coord=None, wav_type='air'):
    """
    Applies em_fitting to a given spectrum, while using the two-gaussian model.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    redshift : float
        redshift of the lines.

    plot : boolean
        if True, the fit will be plotted. Default is False.

    var : :obj:'~numpy.ndarray'
        variance array of the data. If not None, the variance will be
        considered while doing the fit. Default is None.

    coord : 2-tuple of int
        spaxel coordinates. Only used if plot is True. If not None, the output
        plot will include the spaxel coordinates in the name of the plot.
        Default is None.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    Returns
    -------
    popt : :obj:'~numpy.ndarray'
        optimal parameters found by curve_fit. If the fit is unsuccesful, it
        will return 0.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix calculated by curve_fit. If the fit is unsuccesful,
        it will return 0.
    """
    if wav_type == 'air':
        model = two_gaussian_model_air
        oii_wav = dict_wav_air['OII_l'] * (1+redshift)
    elif wav_type == 'vac':
        model = two_gaussian_model_vac
        oii_wav = dict_wav_vac['OII_l'] * (1+redshift)
    wav_cut, spec_cut, var_cut, p0, bounds = useful.cut_spec(wav, spec, var,
                                                             20, oii_wav,
                                                             cont=True)
    p0 = np.concatenate((p0, np.array([1.0])))
    bounds = np.concatenate((bounds, np.array([[0.5, 2.0]]).T), axis=1)
    popt, pcov = em_fitting(wav_cut, spec_cut, model, p0, bounds, var=var_cut,
                            coord=coord)
    if plot and type(popt) != int:
        display.plot_fit(wav_cut, spec_cut, popt, "oii_doublet", redshift,
                         coord=coord, save=True)
    return popt, pcov


def four_gaussian_fitting(wav, spec, redshift, plot=False, var=None,
                          coord=None, wav_type='air'):
    """
    Applies em_fitting to a given spectrum, while using the four-gaussian
    model.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    redshift : float
        redshift of the lines.

    plot : boolean
        if True, the fit will be plotted. Default is False.

    var : :obj:'~numpy.ndarray'
        variance array of the data. If not None, the variance will be
        considered while doing the fit. Default is None.

    coord : 2-tuple of int
        spaxel coordinates. Only used if plot is True. If not None, the output
        plot will include the spaxel coordinates in the name of the plot.
        Default is None.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    Returns
    -------
    popt : :obj:'~numpy.ndarray'
        optimal parameters found by curve_fit. If the fit is unsuccesful, it
        will return 0.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix calculated by curve_fit. If the fit is unsuccesful,
        it will return 0.
    """
    if wav_type == 'air':
        model = four_gaussian_model_air
        w_list = np.array([dict_wav_air['OII_l'], dict_wav_air['Hbeta'],
                           dict_wav_air['OIII_r']])
    elif wav_type == 'vac':
        model = four_gaussian_model_vac
        w_list = np.array([dict_wav_vac['OII_l'], dict_wav_vac['Hbeta'],
                           dict_wav_vac['OIII_r']])
    wav_cut, spec_cut, var_cut, p0, bounds = useful.cut_spec_n(wav, spec, var,
                                                               20, w_list *
                                                               (1+redshift), 3,
                                                               cont=True,
                                                               four=True)
    popt, pcov = em_fitting(wav_cut, spec_cut, model, p0, bounds, var=var_cut,
                            coord=coord)
    if plot and type(popt) != int:
        display.plot_fit(wav_cut, spec_cut, popt, "all_lines", redshift,
                         coord=coord, save=True)
    return popt, pcov


def hbeta_hgamma_fitting(wav, spec, redshift, plot=False, var=None,
                         coord=None, wav_type='air', n=1):
    """
    Applies em_fitting to a given spectrum, while using the Hbeta-Hgamma model.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    redshift : float
        redshift of the lines.

    plot : boolean
        if True, the fit will be plotted. Default is False.

    var : :obj:'~numpy.ndarray'
        variance array of the data. If not None, the variance will be
        considered while doing the fit. Default is None.

    coord : 2-tuple of int
        spaxel coordinates. Only used if plot is True. If not None, the output
        plot will include the spaxel coordinates in the name of the plot.
        Default is None.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    n : int
        number of Hbeta and Hgamma components to fit. Default is 1.

    Returns
    -------
    popt : :obj:'~numpy.ndarray'
        optimal parameters found by curve_fit. If the fit is unsuccesful, it
        will return 0.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix calculated by curve_fit. If the fit is unsuccesful,
        it will return 0.
    """
    if wav_type == 'air':
        model = n_hgamma_hbeta_model_air
        wl_h = wl_h_air
    elif wav_type == 'vac':
        model = n_hgamma_hbeta_model_vac
        wl_h = wl_h_vac
    wav_cut, spec_cut, var_cut, p0, bounds = useful.cut_spec_n(wav, spec, var,
                                                               15, wl_h *
                                                               (1+redshift), 2,
                                                               cont=True)
    p0 = np.tile(p0, n)
    bounds = np.tile(bounds, n)
    popt, pcov = em_fitting(wav_cut, spec_cut, model, p0, bounds, var=var_cut,
                            coord=coord)
    if plot and type(popt) != int:
        display.plot_fit(wav_cut, spec_cut, popt, "hbeta_hgamma", redshift,
                         coord=coord, save=True, wav_type=wav_type)
    return popt, pcov


def fit_absorption(wav, spec, model, p0, bounds, var=None):
    """
    Fits absorption model to the given data.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    model : function
        model to use on the fitting

    p0 : :obj:'~numpy.ndarray'
        initial guess for curve_fit.

    bounds : :obj:'~numpy.ndarray'
        lower and upper bounds for the different parameters of the model.

    var : :obj:'~numpy.ndarray'
        variance array of the data. If not None, the variance will be
        considered while doing the fit. Default is None.

    Returns
    -------
    popt : :obj:'~numpy.ndarray'
        optimal parameters found by curve_fit. If the fit is unsuccesful, it
        will return 0.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix calculated by curve_fit. If the fit is unsuccesful,
        it will return 0.
    """
    try:
        if var is None:
            popt, pcov = curve_fit(model, wav, spec, p0=p0, bounds=bounds)
        else:
            popt, pcov = curve_fit(model, wav, spec, p0=p0, bounds=bounds,
                                   sigma=np.sqrt(var))
        return popt, pcov
    except RuntimeError:
        print('Fit failed')
        return 0, 0


def fit_mgii(wav, spec, redshift, plot=False, var=None, system=None,
             wav_type='air', n=1):
    """
    Fits n Mg IIs to the given data.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelegnth array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    redshift : float
        redshift of the relevant object.

    plot : boolean
        if True, it will plot the fit. Default is False.

    var : :obj:'~numpy.ndarray'
        variance array of the data. If given, it will take the variance into
        consideration when making the fits. Default is None.

    system : str
        name of the relevant system. If given and plot is True, it will write
        the name of the system in the plot.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    n : int
        number of Mg IIs to fit. Default is 1.

    Returns
    -------
    popt : :obj:'~numpy.ndarray'
        optimal parameters found by curve_fit. If the fit is unsuccesful, it
        will return 0.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix calculated by curve_fit. If the fit is unsuccesful,
        it will return 0.
    """
    if wav_type == 'air':
        if n == 1:
            model = mgii_model_air
        else:
            model = n_mgii_model_air
        wobs = dict_wav_air['MgII_l'] * (1 + redshift)
    elif wav_type == 'vac':
        if n == 1:
            model = mgii_model_vac
        else:
            model = n_mgii_model_vac
        wobs = dict_wav_vac['MgII_l'] * (1 + redshift)
    wav_mgii, spec_mgii, var_mgii = useful.cut_spec(wav, spec, var, 50, wobs,
                                                    full=False, cont=False)
    spec_norm, var_mgii = useful.normalise_spectra(wav_mgii, spec_mgii, var_mgii,
                                         (50, 10, 20, 50), wobs, deg=2)
    low_bounds = np.array([0, wobs - 20, 0, 0])
    up_bounds = np.array([1, wobs + 20, 2, 1])
    p0 = np.array([0.5, wobs, 0.5, 0.5])
    if n > 1:
        for i in range(n - 1):
            low_bounds = np.append(low_bounds, np.array([0, wobs - 20, 0, 0]))
            up_bounds = np.append(up_bounds, np.array([1, wobs + 20, 2, 1]))
            p0 = np.append(p0, np.array([0.5, wobs, 0.5, 0.5]))
        bounds = np.array([low_bounds, up_bounds])
    popt, pcov = fit_absorption(wav_mgii, spec_norm, model, p0, bounds,
                                var=var_mgii)
    if plot:
        display.plot_mgii_absorption(wav_mgii, spec_norm, var_mgii, redshift,
                                     popt, pcov, n, system, wav_type=wav_type)
    return popt, pcov


def fit_outflow(wav, spec, redshift, var=None, wav_type='air', plot=False,
                em_line='Hbeta', n=2, system=None, save=False,
                broad_component=False, flux_factor=10**(-16),
                remove_ism=False):
    """
    Fits an ISM componnet to the specified emission line. It then fits n
    Mg IIs, forcing one to have the same velocity and velocity dispersion as
    the ISM.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    redshift : float
        redshift of the relevant object.

    var : :obj:'~numpy.ndarray'
        variance array of the data. If given, it will take the variance into
        consideration while performing the fits. Default is None.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    plot : boolean
        if True, it wil plot the emission and absorption fits. Default is
        False.

    em_line : str
        name of the emission line used to fit the ISM component. Default is
        "Hbeta".

    n : int
        number of Mg II components to fit. Default is 2.

    system : str
        name of the relevant system. If given and plot is True, it will plot
        the name of the system in the plot. Also, if given and save is True, it
        will include the name of the system in the name of the saved file.
        Default is None.

    save : boolean
        if True and plot is True, it will save the plot into a directory called
        "mgii_outflow/". If system is not None, it will include the name of the
        system into the name of the saved file. Default is False.

    broad_component : boolean
        if True, it will fit two components to the emission lines. Default is
        False.

    Returns
    -------
    popt : :obj:'~numpy.ndarray'
        optimal parameters found by curve_fit. If the fit is unsuccesful, it
        will return 0.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix calculated by curve_fit. If the fit is unsuccesful,
        it will return 0.

    z_em : float
        calculated redshift of the ISM component.

    sigma_em : float
        calculated standard deviation of the ISM component in Angstrom.
    """
    if wav_type == 'air':
        wl_em = dict_wav_air[em_line]
        wobs_em = wl_em * (1 + redshift)
        mgii_ratio = mgii_ratio_air
        wl_mgii = dict_wav_air['MgII_l']
        wobs_abs = wl_mgii * (1 + redshift)
    elif wav_type == 'vac':
        wl_em = dict_wav_vac[em_line]
        wobs_em = wl_em * (1 + redshift)
        mgii_ratio = mgii_ratio_vac
        wl_mgii = dict_wav_vac['MgII_l']
        wobs_abs = wl_mgii * (1 + redshift)
    cut_spec_output = useful.cut_spec(wav, spec, var, 15,
                                      wobs_em, cont=True)
    wav_cut_em = cut_spec_output[0]
    spec_cut_em = cut_spec_output[1]
    var_cut_em = cut_spec_output[2]
    p0_em = cut_spec_output[3]
    bounds_em = cut_spec_output[4]
    if broad_component:
        popt_em, pcov_em = fit_outflow_emission(wav, spec, var, redshift,
                                                wav_type=wav_type,
                                                em_line=em_line)
        if popt_em[0] > popt_em[3]:
            z_em = popt_em[1] / wl_em - 1
            sigma_em = popt_em[2]
        else:
            z_em = popt_em[4] / wl_em - 1
            sigma_em = popt_em[5]
    else:
        popt_em, pcov_em = curve_fit(gaussian_model, wav_cut_em, spec_cut_em,
                                     p0=p0_em, bounds=bounds_em,
                                     sigma=np.sqrt(var_cut_em))
        z_em = popt_em[1] / wl_em - 1
        sigma_em = popt_em[2]

    def mgii_ism(x, *params):
        A1 = params[0]
        alpha1 = params[1]
        y = 1
        gaussian1 = gaussian_model(x, A1, wl_mgii * (1 + z_em), sigma_em)
        gaussian2 = gaussian_model(x, A1 * alpha1,
                                   wl_mgii * (1 + z_em)/mgii_ratio, sigma_em)
        y = y - gaussian1 - gaussian2
        for i in range(n):
            if i == 0:
                pass
            else:
                A2 = params[4 * (i - 1) + 2]
                mu2 = params[4 * (i - 1) + 3]
                sigma2 = params[4 * (i - 1) + 4]
                alpha2 = params[4 * (i - 1) + 5]
                gaussian3 = gaussian_model(x, A2, mu2, sigma2)
                gaussian4 = gaussian_model(x, alpha2 * A2, mu2/mgii_ratio,
                                           sigma2)
                y = y - gaussian3 - gaussian4
        return y
    wav_mgii, spec_mgii, var_mgii = useful.cut_spec(wav, spec, var, 50,
                                                    wobs_abs, full=False,
                                                    cont=False)
    spec_norm, var_mgii = useful.normalise_spectra(wav_mgii, spec_mgii, var_mgii,
                                         (50, 10, 20, 50), wobs_abs, deg=2)
    low_bounds = np.array([0.2, 0.01])
    up_bounds = np.array([1, 1.1])
    p0 = np.array([0.5, 0.5])
    if n > 1:
        for i in range(n - 1):
            low_bounds = np.append(low_bounds,
                                   np.array([0.05, wobs_abs - 20, 0, 0.2]))
            up_bounds = np.append(up_bounds,
                                  np.array([1, wobs_abs + 20, 2.5, 1.1]))
            p0 = np.append(p0, np.array([0.5, wobs_abs, 1., 0.5]))
    bounds = np.array([low_bounds, up_bounds])
    try:
        popt, pcov = fit_absorption(wav_mgii, spec_norm, mgii_ism, p0, bounds,
                                    var=var_mgii)
        plt.figure()
        plt.plot(wav_mgii, spec_norm)
        plt.plot(wav_mgii, mgii_ism(wav_mgii, *popt))
        plt.show()            
    except RuntimeError:
        print('Fit failed')
        return 0, 0
    if plot:
        display.plot_mgii_outflow(wav_cut_em, spec_cut_em, var_cut_em,
                                  wav_mgii, spec_norm, var_mgii, popt_em, popt,
                                  pcov, n, system, wav_type=wav_type,
                                  save=save, em_line=em_line,
                                  broad_component=broad_component,
                                  flux_factor=flux_factor)
    if remove_ism:
        plt.figure()
        plt.plot(wav_mgii, spec_norm)
        mgii_2796_ism = gaussian_model(wav_mgii, popt[0], wl_mgii * (1 + z_em),
                                       sigma_em)
        mgii_2803_ism = gaussian_model(wav_mgii, popt[0] * popt[1],
                                       wl_mgii * (1 + z_em) / mgii_ratio,
                                       sigma_em)
        spec_no_ism = spec_norm + mgii_2796_ism + mgii_2803_ism
        plt.plot(wav_mgii, spec_no_ism)
        plt.show()
        return popt, pcov, z_em, sigma_em, wav_mgii, spec_no_ism, var_mgii
    return popt, pcov, z_em, sigma_em


def fit_outflow_emission(wav, spec, var, redshift, wav_type='air',
                         em_line='Hbeta', plot=False, system=None, save=False):
    """
    Fits two components to a given emission line: an ISM and an outflow
    component.


    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    var : :obj:'~numpy.ndarray'
        variance array of the data.

    redshift : float
        redshift of the relevant object.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    em_line : str
        name of the emission line used to fit the ISM component. Default is
        "Hbeta".

    plot : boolean
        if True, it will plot the fit. Default is False.

    system : str
        if given and plot is True, it will include the name of the system in
        the plot. Additionally, if given and save is True, it will include the
        name of the system in the saved plot. Default is None.

    save : boolean
        if True and plot is True, it will save the generated plot into a
        directory called "emission_outflow/". If system is not None, it will
        also include the name of the system in the name of the saved file.

    Returns
    -------
    popt : :obj:'~numpy.ndarray'
        optimal parameters found by curve_fit. If the fit is unsuccesful, it
        will return 0.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix calculated by curve_fit. If the fit is unsuccesful,
        it will return 0.
    """
    if wav_type == 'air':
        wobs = dict_wav_air[em_line] * (1 + redshift)
    elif wav_type == 'vac':
        wobs = dict_wav_vac[em_line] * (1 + redshift)
    wav_cut, spec_cut, var_cut = useful.cut_spec(wav, spec, var, 15,
                                                 wobs, cont=True, full=False)
    p0 = [max(spec_cut), wobs - 1, np.std(wav_cut),
          max(spec_cut), wobs + 1, np.std(wav_cut)]
    bounds = [[0, wobs - 7, 1.0, 0, wobs - 7, 1.0],
              [np.inf, wobs + 7, 15, np.inf, wobs + 7, 15]]
    try:
        popt, pcov = curve_fit(two_emissions_model, wav_cut, spec_cut,
                               p0=p0, bounds=bounds,
                               sigma=np.sqrt(var_cut))
    except RuntimeError:
        print('Fit failed')
        return 0, 0
    if plot:
        display.plot_emission_outflow(wav_cut, spec_cut, var_cut, redshift,
                                      popt, pcov, wav_type=wav_type,
                                      em_line=em_line, system=system,
                                      save=save)
    return popt, pcov


def fit_five_abs(wav, spec, var, redshift, wav_type='air', plot=False,
                 system=None, n=1, save=False):
    """
    Fits the Mg II 2796,2803, Mg II and Fe II 2586,2600 lines at the same
    velocity. If n is higher than 1, it will fit n components to the lines.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavlength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    var : :obj:'~numpy.ndarray'
        variance array of the data.

    redshift : float
        redshift of the relevant object.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    plot : boolean
        if True, it will plot the calculated fit. Default is False.

    system : str
        if given and plot is True, it will include the name of the system in
        the plot. Additionally, if given and save it True, it will include the
        name of the system in the saved file with the plot.

    n : int
        number of velocity components to fit. Default is 1.

    save : boolean
        if True and plot is True, it will save the generated plot into a
        directory named "abs_qso". Default is False.

    Returns
    -------
    popt : :obj:'~numpy.ndarray'
        optimal parameters found by curve_fit. If the fit is unsuccesful, it
        will return 0.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix calculated by curve_fit. If the fit is unsuccesful,
        it will return 0.
    """
    if wav_type == 'air':
        model = n_five_abs_model_air
        wobs_mgii = dict_wav_air['MgII_l'] * (1 + redshift)
        wobs_mgi = dict_wav_air['MgI'] * (1 + redshift)
        wobs_feii = dict_wav_air['FeII_l'] * (1 + redshift)
    elif wav_type == 'vac':
        model = n_five_abs_model_vac
        wobs_mgii = dict_wav_vac['MgII_l'] * (1 + redshift)
        wobs_mgi = dict_wav_vac['MgI'] * (1 + redshift)
        wobs_feii = dict_wav_vac['FeII_l'] * (1 + redshift)
    # cut Mg II
    wav_mgii, spec_mgii, var_mgii = useful.cut_spec(wav, spec, var, 50,
                                                    wobs_mgii, full=False,
                                                    cont=False)
    spec_norm_mgii, var_mgii = useful.normalise_spectra(wav_mgii, spec_mgii, var_mgii,
                                              (50, 10, 20, 50), wobs_mgii,
                                              deg=2)
    # cut Mg I
    wav_mgi, spec_mgi, var_mgi = useful.cut_spec(wav, spec, var, 50, wobs_mgi,
                                                 full=False, cont=False)
    spec_norm_mgi, var_mgi = useful.normalise_spectra(wav_mgi, spec_mgi, var_mgi,
                                             (11.75, 5, 5, 11.75), wobs_mgi,
                                             deg=2)
    # cut Fe II
    wav_feii, spec_feii, var_feii = useful.cut_spec(wav, spec, var, 50,
                                                    wobs_feii, full=False,
                                                    cont=False)
    spec_norm_feii, var_feii = useful.normalise_spectra(wav_feii, spec_feii, var_feii,
                                              (12, 5, 27, 50), wobs_feii,
                                              deg=3)
    # put them all together
    wav_all = np.append(wav_mgii, wav_mgi)
    wav_all = np.append(wav_feii, wav_all)
    spec_norm_all = np.append(spec_norm_mgii, spec_norm_mgi)
    spec_norm_all = np.append(spec_norm_feii, spec_norm_all)
    var_all = np.append(var_mgii, var_mgi)
    var_all = np.append(var_feii, var_all)
    # bounds and p0
    p0 = np.array([0.5, wobs_mgii, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    low_bounds = np.array([0, wobs_mgii - 20, 0, 0, 0, 0, 0, 0, 0])
    up_bounds = np.array([1, wobs_mgii + 20, 7, 1, 1, 7, 1, 7, 1])
    if n > 1:
        for i in range(n - 1):
            low_bounds = np.append(low_bounds,
                                   np.array([0, wobs_mgii - 20, 0, 0, 0, 0,
                                             0, 0, 0]))
            up_bounds = np.append(up_bounds,
                                  np.array([1, wobs_mgii + 20, 7, 1, 1, 7, 1,
                                            7, 1]))
            p0 = np.append(p0, np.array([0.5, wobs_mgii, 0.5, 0.5, 0.5, 0.5,
                                         0.5, 0.5, 0.5]))
    bounds = np.array([low_bounds, up_bounds])
    try:
        popt, pcov = fit_absorption(wav_all, spec_norm_all, model, p0, bounds,
                                    var=var_all)
    except RuntimeError:
        print('Fit failed')
        return 0, 0
    if plot:
        display.plot_five_abs(wav_all, spec_norm_all, var_all, redshift, popt,
                              pcov, wav_type=wav_type, system=system,
                              save=save)
    return popt, pcov


def fit_feii(wav, spec, var, redshift, wav_type='air', plot=False, system=None,
             n=1, save=False, ism_comp=False, em_line='Hbeta',
             broad_component=False):
    """
    Fits n sets of the Fe II 2586,2600 doublet.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    var : :obj:'~numpy.ndarray'
        variance array of the data.

    redshift : float
        redshift of the relevant object

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    plot : boolean
        if True, it will plot the calculated fit. Default is False.

    system : str
        if given and plot is True, it will include the name of the system in
        the generated plot. Additionally, if save is True, it will include the
        name of the system in the name of the saved file.

    n : int
        number of velocity components to fit. Default is 1.

    save : boolean
        if True and plot is True, it will plot the calculated fit and save it
        into a directory called "feii_gal/". Default is False.

    ism_comp : boolean
        if True, it will use the emission lines to constrain one of the
        components as an ISM component. Default is False.

    em_line : str
        name of the emission line used to constrain the ISM component. Default
        is "Hbeta".

    broad_component : boolean
        if True, it will assume the emission has two velocity components.
        Default is False.

    Returns
    -------
    popt_feii : :obj:'~numpy.ndarray'
        optimal parameters found by curve_fit. If the fit is unsuccesful, it
        will return 0.

    pcov_feii : :obj:'~numpy.ndarray'
        covariance matrix calculated by curve_fit. If the fit is unsuccesful,
        it will return 0.
    """
    if wav_type == 'air':
        model = feii_air
        wl_mgii = dict_wav_air['MgII_l']
        wl_feii = dict_wav_air['FeII_l']
        feii_ratio = feii_ratio_air
        wobs_feii = dict_wav_air['FeII_l'] * (1 + redshift)
        wobs_mgii = dict_wav_air['MgII_l'] * (1 + redshift)
    elif wav_type == 'vac':
        model = feii_vac
        wl_mgii = dict_wav_vac['MgII_l']
        wl_feii = dict_wav_vac['FeII_l']
        feii_ratio = feii_ratio_vac
        wobs_feii = dict_wav_vac['FeII_l'] * (1 + redshift)
        wobs_mgii = dict_wav_vac['MgII_l'] * (1 + redshift)
    # fit the ISM component if necessary
    if ism_comp:
        fit_outflow_output = fit_outflow(wav, spec, redshift, var=var,
                                         wav_type=wav_type, em_line=em_line,
                                         n=n, system=system,
                                         broad_component=broad_component)
        popt = fit_outflow_output[0]
        pcov = fit_outflow_output[1]
        z_em = fit_outflow_output[2]
        sigma_em = fit_outflow_output[3]

        def feii_ism(x, *params):
            y = 1
            A1 = params[0]
            A2 = params[1]
            gaussian1 = gaussian_model(x, A1, wl_feii * (1 + z_em), sigma_em)
            gaussian2 = gaussian_model(x, A2,
                                       wl_feii * (1 + z_em) / feii_ratio,
                                       sigma_em)
            y = y - gaussian1 - gaussian2
            for i in range(n):
                if i == 0:
                    pass
                else:
                    A3 = params[4 * (i - 1) + 2]
                    mu3 = params[4 * (i - 1) + 3]
                    z_3 = mu3/wl_mgii - 1
                    sigma3 = params[4 * (i - 1) + 4]
                    A4 = params[4 * (i - 1) + 5]
                    gaussian3 = gaussian_model(x, A3, wl_feii * (1 + z_3),
                                               sigma3)
                    gaussian4 = gaussian_model(x, A4,
                                               wl_feii * (1 + z_3)
                                               / feii_ratio, sigma3)
                    y = y - gaussian3 - gaussian4
            return y
    # cut Fe II
    wav_feii, spec_feii, var_feii = useful.cut_spec(wav, spec, var, 50,
                                                    wobs_feii, full=False,
                                                    cont=False)
    spec_norm_feii, var_feii = useful.normalise_spectra(wav_feii, spec_feii, var_feii,
                                              (40, 5, 30, 50), wobs_feii,
                                              deg=3)
    p0_feii = np.array([0.5, 0.5])
    low_bounds_feii = np.array([0, 0])
    up_bounds_feii = np.array([1, 1])
    if n > 1:
        for i in range(n - 1):
            low_bounds_feii = np.append(low_bounds_feii,
                                        np.array([0, wobs_mgii - 20, 0.25, 0]))
            up_bounds_feii = np.append(up_bounds_feii,
                                       np.array([1, wobs_mgii + 20, 7, 1]))
            p0_feii = np.append(p0_feii, np.array([0.5, wobs_mgii, 0.5, 0.5]))
    bounds_feii = np.array([low_bounds_feii, up_bounds_feii])
    try:
        popt_feii, pcov_feii = fit_absorption(wav_feii, spec_norm_feii,
                                              feii_ism, p0_feii, bounds_feii,
                                              var=var_feii)
    except RuntimeError:
        print('Fit failed')
        return 0, 0
    if plot:
        display.plot_feii(wav_feii, spec_norm_feii, var_feii, popt_feii,
                          pcov_feii, z_em, sigma_em, n=n, wav_type=wav_type,
                          system=system, save=save)
    return popt_feii, pcov_feii


def fit_caii(wav, spec, var, redshift, wav_type='air', plot=False, system=None,
             n=1, save=False, ism_comp=False, em_line='Hbeta',
             broad_component=False):
    """
    Fits n sets of the Ca II H&K doublet.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    var : :obj:'~numpy.ndarray'
        variance array of the data.

    redshift : float
        redshift of the relevant object

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    plot : boolean
        if True, it will plot the calculated fit. Default is False.

    system : str
        if given and plot is True, it will include the name of the system in
        the generated plot. Additionally, if save is True, it will include the
        name of the system in the name of the saved file.

    n : int
        number of velocity components to fit. Default is 1.

    save : boolean
        if True and plot is True, it will plot the calculated fit and save it
        into a directory called "caii_gal/". Default is False.

    ism_comp : boolean
        if True, it will use the emission lines to constrain one of the
        components as an ISM component. Default is False.

    em_line : str
        name of the emission line used to constrain the ISM component. Default
        is "Hbeta".

    broad_component : boolean
        if True, it will assume the emission has two velocity components.
        Default is False.

    Returns
    -------
    popt_caii_h_k : :obj:'~numpy.ndarray'
        optimal parameters found by curve_fit. If the fit is unsuccesful, it
        will return 0.

    pcov_caii_h_k : :obj:'~numpy.ndarray'
        covariance matrix calculated by curve_fit. If the fit is unsuccesful,
        it will return 0.
    """
    if wav_type == 'air':
        model = caii_air
        wl_mgii = dict_wav_air['MgII_l']
        wl_caii_h = dict_wav_air['CaII_H']
        wl_caii_k = dict_wav_air['CaII_K']
        caii_ratio = caii_ratio_air
        wobs_caii_h = dict_wav_air['CaII_H'] * (1 + redshift)
        wobs_caii_k = dict_wav_air['CaII_K'] * (1 + redshift)
        wobs_mgii = dict_wav_air['MgII_l'] * (1 + redshift)
    elif wav_type == 'vac':
        model = caii_vac
        wl_mgii = dict_wav_vac['MgII_l']
        wl_caii_h = dict_wav_vac['CaII_H']
        wl_caii_k = dict_wav_vac['CaII_K']
        caii_ratio = caii_ratio_vac
        wobs_caii_h = dict_wav_vac['CaII_H'] * (1 + redshift)
        wobs_caii_k = dict_wav_vac['CaII_K'] * (1 + redshift)
        wobs_mgii = dict_wav_vac['MgII_l'] * (1 + redshift)
    # fit the ISM component if necessary
    if ism_comp:
        fit_outflow_output = fit_outflow(wav, spec, redshift, var=var,
                                         wav_type=wav_type, em_line=em_line,
                                         n=n, system=system,
                                         broad_component=broad_component)
        popt = fit_outflow_output[0]
        pcov = fit_outflow_output[1]
        z_em = fit_outflow_output[2]
        sigma_em = fit_outflow_output[3]

        def caii_ism(x, *params):
            y = 1
            A1 = params[0]
            A2 = params[1]
            gaussian1 = gaussian_model(x, A1, wl_caii_h * (1 + z_em), sigma_em)
            gaussian2 = gaussian_model(x, A2,
                                       wl_caii_h * (1 + z_em) / caii_ratio,
                                       sigma_em)
            y = y - gaussian1 - gaussian2
            for i in range(n):
                if i == 0:
                    pass
                else:
                    A3 = params[4 * (i - 1) + 2]
                    mu3 = params[4 * (i - 1) + 3]
                    z_3 = mu3/wl_mgii - 1
                    sigma3 = params[4 * (i - 1) + 4]
                    A4 = params[4 * (i - 1) + 5]
                    gaussian3 = gaussian_model(x, A3, wl_caii_h * (1 + z_3),
                                               sigma3)
                    gaussian4 = gaussian_model(x, A4, wl_caii_h * (1 + z_3) /
                                               caii_ratio, sigma3)
                    y = y - gaussian3 - gaussian4
            return y
    # cut Ca II H & K
    w_central = (wobs_caii_h + wobs_caii_k) / 2
    wav_caii_h_k, spec_caii_h_k, var_caii_h_k = useful.cut_spec(wav, spec, var,
                                                                100, w_central,
                                                                full=False,
                                                                cont=False)
    spec_norm_caii_h_k, var_caii_h_k = useful.normalise_spectra(wav_caii_h_k, spec_caii_h_k,
                                                  var_caii_h_k,
                                                  (90, 50, 60, 100), w_central,
                                                  deg=2)
    p0_caii_h_k = np.array([0.5, 0.5])
    low_bounds_caii_h_k = np.array([0, 0])
    up_bounds_caii_h_k = np.array([1, 1])
    if n > 1:
        for i in range(n - 1):
            low_bounds_caii_h_k = np.append(low_bounds_caii_h_k,
                                            np.array([0, wobs_mgii - 5, 2, 0]))
            up_bounds_caii_h_k = np.append(up_bounds_caii_h_k,
                                           np.array([1, wobs_mgii + 5, 7, 1]))
            p0_caii_h_k = np.append(p0_caii_h_k,
                                    np.array([0.5, wobs_mgii, 5, 0.5]))
    bounds_caii_h_k = np.array([low_bounds_caii_h_k, up_bounds_caii_h_k])
    try:
        popt_caii_h_k, pcov_caii_h_k = fit_absorption(wav_caii_h_k,
                                                      spec_norm_caii_h_k,
                                                      caii_ism, p0_caii_h_k,
                                                      bounds_caii_h_k,
                                                      var=var_caii_h_k)
    except RuntimeError:
        print('Fit failed')
        return 0, 0
    if plot:
        display.plot_caii(wav_caii_h_k, spec_norm_caii_h_k, var_caii_h_k,
                          popt_caii_h_k, pcov_caii_h_k, z_em, sigma_em, n=n,
                          wav_type=wav_type, system=system, save=save)
    return popt_caii_h_k, pcov_caii_h_k


def hbeta_hgamma_fluxes(popt, n=1):
    """
    Calculates the Hbeta and Hgamma flux of a given spectrum.

    Parameters
    ----------
    popt : :obj:'~numpy.ndarray'
        optimal parameters of the Hbeta-Hgamma fit calculated by curve_fit.

    n : int
        number of Hbeta and Hgamma pairs to fit.

    Returns
    -------
    hbeta_flux : float
        calculated Hbeta flux. If the curve_fit fit is unsuccessful, it will
        return 1.

    hgamma_flux : float
        calculated Hgamma flux. If the curve_fit fit is unsuccessful, it will
        return 1.
    """
    if type(popt) == int:
        hbeta_flux = 1
        hgamma_flux = 1
    else:
        hgamma_flux = 0
        hbeta_flux = 0
        for i in range(n):
            hgamma_flux += popt[4*i]*popt[4*i + 2]*np.sqrt(2*np.pi)
            hbeta_flux += popt[4*i + 3]*popt[4*i + 2]*np.sqrt(2*np.pi)
    return hbeta_flux, hgamma_flux


def calculate_velocity(wav, spec, redshift, var=None,
                       model="four_gaussian", wav_type='air', rest_wav=0):
    """
    Calculates velocity of a spectrum given a redshift.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    redshift : float
        redshift of the lines.

    var : :obj:'~numpy.ndarray'
        variance array of the data. If not None, the variance will be
        considered while doing the fit. Default is None.

    model : str
        model to use in the fitting. Supported models are "four_gaussian" and
        "two_gaussian".

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    rest_wav : float
        rest-frame wavelength of the line used to calculate velocity. Default
        is 0.

    Returns
    -------
    velocity : float
        calculated velocity of the spectrum. If the fit is unsuccessful, it
        will return nan.
    """
    if model == "four_gaussian":
        popt, pcov = four_gaussian_fitting(wav, spec, redshift, var=var,
                                           wav_type=wav_type)
    elif model == "two_gaussian":
        popt, pcov = two_gaussian_fitting(wav, spec, redshift, var=var,
                                          wav_type=wav_type)
    elif model == "one_gaussian":
        wav_obs = rest_wav * (1 + redshift)
        wav_cut, spec_cut, var_cut, p0, bounds = useful.cut_spec(wav, spec,
                                                                 var, 20,
                                                                 wav_obs)
        popt, pcov = em_fitting(wav_cut, spec_cut, gaussian_model, p0, bounds)
    if type(popt) != int:
        if rest_wav == 0:
            if wav_type == 'air':
                oii_wav = dict_wav_air['OII_l']
            elif wav_type == 'vac':
                oii_wav = dict_wav_vac['OII_l']
            velocity = useful.vel(oii_wav * (1 + redshift), popt[1])
        else:
            velocity = useful.vel(rest_wav * (1 + redshift), popt[1])
    else:
        velocity = np.nan
    return velocity


def velocity_map(wav, data, var, redshift, mask_oii=None, mask_hbeta=None,
                 mask_other_line=None, em_line='Halpha', plot=False,
                 cubeid=None, wav_type='air'):
    """
    Applies calculate_velocity to a data cube. Returns a velocity map.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    data : :obj:'~numpy.ndarray'
        flux data cube.

    var : :obj:'~numpy.ndarray'
        variance cube of the data.

    redshift : float
        redshift of the relevant object.

    mask_oii : :obj:'~numpy.ndarray'
        if not None, it will only perform the velocity calculation in the
        spaxels marked as True. If the corresponding value of mask_hbeta is
        also True, it will use the four_gaussian_model as the model fit. In
        case the corresponding value of mask_habeta is False, it will use the
        two_gaussian_model as the model fit. Default is None.

    mask_hbeta : :obj:'~numpy.ndarray'
        if not None, it will use the four_gaussian_model in the spaxels marked
        as True and the two_gaussian_model in the spaxels marked as False.
        Default is None.

    mask_other_line : :obj:'~numpy.ndarray'
        if both the other masks are None and this mask is given, it will use a
        different emission line to calculate the velocity. Default is None.

    em_line : str
        name of the extra emission line used to calculate the velocity. Only
        used if mask_other_line is not None. Default is "Halpha".

    plot : boolean
        if True, it will plot the velocity map and save it to the working
        directory. Default is False.

    cubeid : str
        if given, it will include the name of the cube in the name of the saved
        plot. Default is None.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    Returns
    -------
    velocity_map : :obj:'~numpy.ndarray'
        calculated velocity map of the data cube.
    """
    image = data[0, :, :]
    y_total, x_total = image.shape
    velocity_map = np.full_like(image, np.nan, dtype=np.double)
    for y in range(y_total):
        for x in range(x_total):
            spectrum = data[:, y, x]
            variance = var[:, y, x]
            if mask_oii is not None:
                if ~ mask_oii[y, x]:
                    continue
                if mask_hbeta[y, x]:
                    model = "four_gaussian"
                else:
                    model = "two_gaussian"
                velocity_map[y, x] = calculate_velocity(wav, spectrum,
                                                        redshift, var=variance,
                                                        model=model,
                                                        wav_type=wav_type)
            elif mask_other_line is not None:
                if ~ mask_other_line[y, x]:
                    continue
                if wav_type == 'air':
                    rest_wav = dict_wav_air[em_line]
                elif wav_type == 'vac':
                    rest_wav = dict_wav_vac[em_line]
                model = "one_gaussian"
                velocity_map[y, x] = calculate_velocity(wav, spectrum,
                                                        redshift, var=variance,
                                                        model=model,
                                                        wav_type=wav_type,
                                                        rest_wav=rest_wav)
    if plot:
        display.plot_velocity_map(velocity_map, save=True,
                                  system_name=gal_names[cubeid])
    return velocity_map


def mu_map(wav, data, var, redshift, mask_oii=None, mask_hbeta=None,
           plot=False, cubeid=None, wav_type='air'):
    """
    Applies calculate_velocity to a data cube. Returns a central wavelength
    map.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    data : :obj:'~numpy.ndarray'
        flux data cube.

    var : :obj:'~numpy.ndarray'
        variance cube of the data.

    redshift : float
        redshift of the relevant object.

    mask_oii : :obj:'~numpy.ndarray'
        if not None, it will only perform the velocity calculation in the
        spaxels marked as True. If the corresponding value of mask_hbeta is
        also True, it will use the four_gaussian_model as the model fit. In
        case the corresponding value of mask_habeta is False, it will use the
        two_gaussian_model as the model fit. Default is None.

    mask_hbeta : :obj:'~numpy.ndarray'
        if not None, it will use the four_gaussian_model in the spaxels marked
        as True and the two_gaussian_model in the spaxels marked as False.
        Default is None.

    plot : boolean
        if True, it will plot the central wavelength map and save it to the
        working directory. Default is False.

    cubeid : str
        if given, it will include the name of the cube in the name of the saved
        plot. Default is None.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    Returns
    -------
    mu_map_array : :obj:'~numpy.ndarray'
        calculated central wavelength map of the data cube.
    """
    image = data[0, :, :]
    y_total, x_total = image.shape
    mu_map_array = np.full_like(image, np.nan, dtype=np.double)
    for y in range(y_total):
        for x in range(x_total):
            if mask_oii is not None:
                if ~ mask_oii[y, x]:
                    continue
                spectrum = data[:, y, x]
                variance = var[:, y, x]
                if mask_hbeta[y, x]:
                    model = four_gaussian_model
                else:
                    model = two_gaussian_model
                if model == four_gaussian_model:
                    popt, pcov = four_gaussian_fitting(wav, spectrum,
                                                       redshift, var=variance,
                                                       wav_type=wav_type)
                elif model == two_gaussian_model:
                    popt, pcov = two_gaussian_fitting(wav, spectrum,
                                                      redshift, var=variance,
                                                      wav_type=wav_type)
                if type(popt) != int:
                    mu_map_array[y, x] = popt[1]
                else:
                    mu_map_array[y, x] = np.nan
    if plot:
        display.plot_velocity_map(velocity_map, save=True,
                                  system_name=gal_names[cubeid])
    return mu_map_array


def flux_maps(wav, data, var, redshift, mask_oii, mask_hbeta, separate=False,
              error=False, plot=False, wav_type='air'):
    """
    Calculates [O II], Hbeta and [O III] fluxes.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    data : :obj:'~numpy.ndarray'
        flux data cube.

    var : :obj:'~numpy.ndarray'
        variance cube of the data.

    redshift : float
        redshift of the relevant object.

    mask_oii : :obj:'~numpy.ndarray'
        it will only perform the velocity calculation in the spaxels marked as
        True. If the corresponding value of mask_hbeta is also True, it will
        use the four_gaussian_model as the model fit. In case the corresponding
        value of mask_habeta is False, it will use the two_gaussian_model as
        the model fit.

    mask_hbeta : :obj:'~numpy.ndarray'
        it will use the four_gaussian_model in the spaxels marked as True and
        the two_gaussian_model in the spaxels marked as False.

    separate : boolean
        if True, it will calculate the fluxes of [O II]3727 and [O II]3729
        separately. Default is False.

    error : boolean
        if True, it will also calculate the error in the fluxes. Default is
        False.

    plot : boolean
        if True, it will plot the fits to the emission lines. Default is False.

    Returns
    -------
    oii_flux_map : :obj:'~numpy.ndarray'
        calculated [O II] flux map. If separate is True, it will only include
        the flux from [O II]3727, if separate is False, its value will be
        [O II]3727 + [O II]3729.

    oii_error_map : :obj:'~numpy.ndarray'
        only if error is True. Calculated [O II] flux error map. If separate is
        True, it will only include the error from [O II]3727, if separate is
        False, its value will be [O II]3727 + [O II]3729.

    oii_2_flux_map : :obj:'~numpy.ndarray'
        only if separate is True. Calculated [O II]3729 flux.

    oii_2_error_map : :obj:'~numpy.ndarray'
        only if separate and error are both True. Calculated [O II]3727 flux
        error map.

    hbeta_flux_map : :obj:'~numpy.ndarray'
        calculated Hbeta flux map.

    hbeta_error_map : :obj:'~numpy.ndarray'
        only if error is True. Calculated Hbeta error flux.

    oiii_flux_map : :obj:'~numpy.ndarray'
        calculated [O III] flux map.

    oiii_error_map : :obj:'~numpy.ndarray'
        only if error is True. Calculated [O III] error flux.
    """
    image = data[0, :, :]
    y_total, x_total = image.shape
    oii_flux_map = np.full_like(image, np.nan, dtype=np.double)
    oii_error_map = np.full_like(image, np.nan, dtype=np.double)
    if separate:
        oii_2_flux_map = np.full_like(image, np.nan, dtype=np.double)
        oii_2_error_map = np.full_like(image, np.nan, dtype=np.double)
    oiii_flux_map = np.full_like(image, np.nan, dtype=np.double)
    oiii_error_map = np.full_like(image, np.nan, dtype=np.double)
    hbeta_flux_map = np.full_like(image, np.nan, dtype=np.double)
    hbeta_error_map = np.full_like(image, np.nan, dtype=np.double)
    for y in range(y_total):
        for x in range(x_total):
            if mask_oii is not None:
                if ~ mask_oii[y, x]:
                    continue
                spectrum = data[:, y, x]
                variance = var[:, y, x]
                if mask_hbeta[y, x]:
                    popt, pcov = four_gaussian_fitting(wav, spectrum, redshift,
                                                       var=variance, plot=plot,
                                                       coord=(x, y),
                                                       wav_type=wav_type)
                    if type(popt) != int:
                        perr = np.sqrt(np.diag(pcov))
                        hbeta_flux_map[y, x] = (popt[4] * popt[5] *
                                                np.sqrt(2*np.pi))
                        hbeta_error_map[y, x] = (ufloat(popt[4], perr[4]) *
                                                 ufloat(popt[5], perr[5]) *
                                                 np.sqrt(2*np.pi)).std_dev
                        oiii_flux_map[y, x] = (popt[6] * popt[7] *
                                               np.sqrt(2*np.pi) * 4./3.)
                        oiii_error_map[y, x] = (ufloat(popt[6], perr[6]) *
                                                ufloat(popt[7], perr[7]) *
                                                np.sqrt(2*np.pi) *
                                                4./3.).std_dev
                else:
                    popt, pcov = two_gaussian_fitting(wav, spectrum, redshift,
                                                      var=variance, plot=plot,
                                                      coord=(x, y),
                                                      wav_type=wav_type)
                if type(popt) != int:
                    perr = np.sqrt(np.diag(pcov))
                    if separate:
                        oii_flux_map[y, x] = (popt[0] * popt[2] *
                                              np.sqrt(2*np.pi))
                        oii_error_map[y, x] = (ufloat(popt[0], perr[0]) *
                                               ufloat(popt[2], perr[2]) *
                                               np.sqrt(2*np.pi)).std_dev
                        oii_2_flux_map[y, x] = (popt[0] * popt[3] * popt[2] *
                                                np.sqrt(2*np.pi))
                        oii_2_error_map[y, x] = (ufloat(popt[0], perr[0]) *
                                                 ufloat(popt[2], perr[2]) *
                                                 ufloat(popt[3], perr[3]) *
                                                 np.sqrt(2*np.pi)).std_dev
                    else:
                        oii_flux_map[y, x] = (popt[0] * popt[2] *
                                              np.sqrt(2*np.pi) + popt[0] *
                                              popt[3] * popt[2] *
                                              np.sqrt(2*np.pi))
                        oii_error_map[y, x] = (ufloat(popt[0], perr[0]) *
                                               ufloat(popt[2], perr[2]) *
                                               np.sqrt(2*np.pi) +
                                               ufloat(popt[0], perr[0]) *
                                               ufloat(popt[2], perr[2]) *
                                               ufloat(popt[3], perr[3]) *
                                               np.sqrt(2*np.pi)).std_dev
    if separate:
        if error:
            return oii_flux_map, oii_error_map, oii_2_flux_map, \
                   oii_2_error_map, hbeta_flux_map, hbeta_error_map, \
                   oiii_flux_map, oiii_error_map
        else:
            return oii_flux_map, oii_2_flux_map, hbeta_flux_map, oiii_flux_map
    else:
        if error:
            return oii_flux_map, oii_error_map, hbeta_flux_map, \
                   hbeta_error_map, oiii_flux_map, oiii_error_map
        else:
            return oii_flux_map, hbeta_flux_map, oiii_flux_map


def sfr_map(hbeta_flux_map, redshift, plot=False, cubeid=None, error=False,
            hbeta_error=None, flux_factor=10**(-16)):
    """
    Calculates SFR map of a data cube. Assumes kroupa IMF.

    Parameters
    ----------
    hbeta_flux_map : :obj:'~numpy.ndarray'
        Hbeta flux map.

    redshift : float
        redshift of the relevant object.

    plot : boolean
        if True, it will plot the calculated SFR map and save it to the working
        directory. Default is False.

    cubeid : str
        if given, it will include the cubeid in the name of the saved plot.
        Default is None.

    error: boolean
        if True, it will calculate the error in the SFR and return a SFR error
        map. If True, it is necessary to include an Hbeta flux error map.
        Default is False.

    hbeta_error : :obj:'~numpy.ndarray'
        will only be used if error is True. Hbeta flux error map. Default is
        None.

    flux_factor : float
        factor to multiply your flux so that it is in units of erg / s / cm^2.
        Default is 10^-16.

    Returns
    -------
    SFR_map : :obj:'~numpy.ndarray'
        calculated SFR map.

    SFR_map_error : :obj:'~numpy.ndarray'
        calculated SFR error map. Only if error is True.
    """
    Dlumincm = useful.d_lumin_cm(redshift)
    hbeta_lumin_map = (hbeta_flux_map * flux_factor *
                       (4.0 * np.pi * Dlumincm**2.0))
    A_hb = 0.0
    C_ha = 10.0**(-41.257)
    lumin_ratio = 2.87
    SFR_map = hbeta_lumin_map * lumin_ratio * C_ha * 10.0**(-0.4 * A_hb)
    if error:
        hbeta = unp.uarray(hbeta_flux_map, hbeta_error)
        hbeta_lumin_map_error = (hbeta * flux_factor *
                                 (4.0 * np.pi * Dlumincm**2.0))
        SFR_map_error = unp.std_devs(hbeta_lumin_map_error * lumin_ratio *
                                     C_ha * 10.0**(-0.4 * A_hb))
    if plot:
        display.plot_sfr_map(SFR_map, save=True, system_name=gal_names[cubeid])
    if error:
        return SFR_map, SFR_map_error
    return SFR_map


def sfr_oii_map(oii_flux_map, redshift, plot=False, cubeid=None, error=False,
                oii_error=None, flux_factor=10**(-16)):
    """
    Calculates [O II] SFR map of a data cube. Assumes kroupa IMF.

    Parameters
    ----------
    oii_flux_map : :obj:'~numpy.ndarray'
        [O II] flux map.

    redshift : float
        redshift of the relevant object.

    plot : boolean
        if True, it will plot the calculated [O II] SFR map and save it to the
        working directory. Default is False.

    cubeid : str
        if given, it will include the cubeid in the name of the saved plot.
        Default is None.

    error: boolean
        if True, it will calculate the error in the [O II] SFR and return a
        [O II] SFR error map. If True, it is necessary to include an [O II]
        flux error map. Default is False.

    oii_error : :obj:'~numpy.ndarray'
        [O II] flux error map. Will only be used if error is True. Default is
        None.

    flux_factor : float
        factor to multiply your flux so that it is in units of erg / s / cm^2.
        Default is 10^-16.

    Returns
    -------
    sfr_oii : :obj:'~numpy.ndarray'
        calculated [O II] SFR map.

    sfr_oii_error : :obj:'~numpy.ndarray'
        calculated [O II] SFR error map. Only if error is True.
    """
    Dlumincm = useful.d_lumin_cm(redshift)
    oii_lumin_map = oii_flux_map * flux_factor * (4.0 * np.pi * Dlumincm**2.0)
    mk_ms = 0.62
    sfr_oii = 6.58e-42 * oii_lumin_map * mk_ms
    if error:
        oii = unp.uarray(oii_flux_map, oii_error)
        oii_lumin_map_error = oii * 10**(-16) * (4.0 * np.pi * Dlumincm**2.0)
        sfr_oii_error = unp.std_devs(6.58e-42 * oii_lumin_map_error * mk_ms)
    if plot:
        display.plot_sfr_map(sfr_oii, save=True, system_name=gal_names[cubeid],
                             oii=True)
    if error:
        return sfr_oii, sfr_oii_error
    return sfr_oii


def R23(oii_flux_map, oiii_flux_map, hbeta_flux_map, error=False,
        oii_flux_error=None, oiii_flux_error=None, hbeta_flux_error=None):
    """
    Calculates the R23 ratio given [O II], Hbeta and [O III] flux maps.

    Parameters
    ----------
    oii_flux_map : :obj:'~numpy.ndarray'
        [O II] flux map.

    oiii_flux_map : :obj:'~numpy.ndarray'
        [O III] flux map.

    hbeta_flux_map : :obj:'~numpy.ndarray'
        Hbeta flux map.

    error : boolean
        if True, it will calculate the error in the R23 ratio and return a R23
        ratio error map. If True, it is necessary to include [O II], Hbeta and
        [O III] flux error maps. Default is False.

    oii_flux_error : :obj:'~numpy.ndarray'
        [O II] flux error map. Will only be used if error is True. Default is
        None.

    oiii_flux_error : :obj:'~numpy.ndarray'
        [O III] flux error map. Will only be used if error is True. Default is
        None.

    hbeta_flux_error : :obj:'~numpy.ndarray'
        Hbeta flux error map. Will only be used if error is True. Default is
        None.

    Returns
    -------
    R23 : :obj:'~numpy.ndarray'
        calculated R23 map.

    R23_error : :obj:'~numpy.ndarray'
        calculated R23 error map. Only if error is True.
    """
    ratio = (oii_flux_map + oiii_flux_map) / hbeta_flux_map
    R23 = np.log10(ratio)
    if error:
        oii_flux = unp.uarray(oii_flux_map, oii_flux_error)
        oiii_flux = unp.uarray(oiii_flux_map, oiii_flux_error)
        hbeta_flux = unp.uarray(hbeta_flux_map, hbeta_flux_error)
        ratio_error = (oii_flux + oiii_flux) / hbeta_flux
        R23_error = unp.std_devs(unp.log10(ratio_error))
        return R23, R23_error
    else:
        return R23


def O23(oii_flux_map, oiii_flux_map, error=False, oii_flux_error=None,
        oiii_flux_error=None):
    """
    Calculates the O23 ratio given [O II] and [O III] maps.

        Parameters
    ----------
    oii_flux_map : :obj:'~numpy.ndarray'
        [O II] flux map.

    oiii_flux_map : :obj:'~numpy.ndarray'
        [O III] flux map.

    error : boolean
        if True, it will calculate the error in the O23 ratio and return a O23
        ratio error map. If True, it is necessary to include [O II] and [O III]
        flux error maps. Default is False.

    oii_flux_error : :obj:'~numpy.ndarray'
        [O II] flux error map. Will only be used if error is True. Default is
        None.

    oiii_flux_error : :obj:'~numpy.ndarray'
        [O III] flux error map. Will only be used if error is True. Default is
        None.

    Returns
    -------
    O23 : :obj:'~numpy.ndarray'
        calculated O23 map.

    O23_error : :obj:'~numpy.ndarray'
        calculated O23 error map. Only if error is True.
    """
    ratio = oiii_flux_map / oii_flux_map
    O23 = np.log10(ratio)
    if error:
        oii_flux = unp.uarray(oii_flux_map, oii_flux_error)
        oiii_flux = unp.uarray(oiii_flux_map, oiii_flux_error)
        ratio_error = oii_flux / oiii_flux
        O23_error = unp.std_devs(unp.log10(ratio_error))
        return O23, O23_error
    else:
        return O23


def z94_method(R23, error=False, R23_error=None):
    """
    Implementation of the Zaritsky et al. 1994 method to calculate an initial
    guess of the oxygen abundance.

    Parameters
    ----------
    R23 : :obj:'~numpy.ndarray'
        R23 map.

    error : boolean
        if True, it will calculate the error in the initial oxygen abundance
        and return it. If True, it is necessary to include R23 error map.
        Default is False.

    R23_error : :obj:'~numpy.ndarray'
        R23 error map. Will only be used if error is True. Default is None.

    Returns
    -------
    abundance : :obj:'~numpy.ndarray'
        initial abundance map.

    abundance_error : :obj:'~numpy.ndarray'
        initial abundance error map.
    """
    abundance = (9.265 - 0.33 * R23 - 0.202 * R23**2 - 0.207 * R23**3 -
                 0.333 * R23**4)
    if error:
        R23_error = unp.uarray(R23, R23_error)
        abundance_error = unp.std_devs(9.265 - 0.33 * R23_error -
                                       0.202 * R23_error**2 -
                                       0.207 * R23_error**3 -
                                       0.333 * R23_error**4)
        return abundance, abundance_error
    else:
        return abundance


def m91_method(R23, O23, error=False, R23_error=None, O23_error=None):
    """
    Implementation of the McGaugh 1991 method to calculate an initial guess of
    the oxygen abundance.

    Parameters
    ----------
    R23 : :obj:'~numpy.ndarray'
        R23 map.

    O23 : :obj:'~numpy.ndarray'
        O23 map.

    error : boolean
        if True, it will calculate the error in the initial oxygen abundance
        and return it. If True, it is necessary to include R23 and O23 error
        maps. Default is False.

    R23_error : :obj:'~numpy.ndarray'
        R23 error map. Will only be used if error is True. Default is None.

    O23_error : :obj:'~numpy.ndarray'
        O23 error map. Will only be used if error is True. Default is None.

    Returns
    -------
    abundance : :obj:'~numpy.ndarray'
        initial abundance map.

    abundance_error : :obj:'~numpy.ndarray'
        initial abundance error map.
    """
    y = O23
    abundance = (12.0 - 4.944 + 0.767 * R23 + 0.602 * R23**2 -
                 y*(0.29 + 0.332 * R23 - 0.331 * R23**2))
    if error:
        y_error = unp.uarray(O23, O23_error)
        R23_error = unp.uarray(R23, R23_error)
        abundance_error = unp.std_devs(12.0 - 4.944 + 0.767 * R23_error +
                                       0.602 * R23_error**2 -
                                       y_error*(0.29 + 0.332 * R23_error -
                                                0.331 * R23_error**2))
        return abundance, abundance_error
    else:
        return abundance


def m91_method_upper(R23, O23, error=False, R23_error=None, O23_error=None):
    """
    Implementation of the McGaugh 1991 method to calculate an initial guess of
    the upper branch of oxygen abundance.

    Parameters
    ----------
    R23 : :obj:'~numpy.ndarray'
        R23 map.

    O23 : :obj:'~numpy.ndarray'
        O23 map.

    error : boolean
        if True, it will calculate the error in the initial oxygen abundance
        and return it. If True, it is necessary to include R23 and O23 error
        maps. Default is False.

    R23_error : :obj:'~numpy.ndarray'
        R23 error map. Will only be used if error is True. Default is None.

    O23_error : :obj:'~numpy.ndarray'
        O23 error map. Will only be used if error is True. Default is None.

    Returns
    -------
    abundance : :obj:'~numpy.ndarray'
        initial abundance map.

    abundance_error : :obj:'~numpy.ndarray'
        initial abundance error map.
    """
    y = O23
    x = R23
    abundance = (12.0 - 2.939 - 0.2 * x - 0.237 * x**2 - 0.305 * x**3 -
                 0.0283 * x**4 - y*(0.0047 - 0.0221 * x - 0.102 * x**2 -
                                    0.0817 * x**3 - 0.00717 * x**4))
    if error:
        y_error = unp.uarray(O23, O23_error)
        x_error = unp.uarray(R23, R23_error)
        abundance_error = unp.std_devs(12.0 - 2.939 - 0.2 * x_error -
                                       0.237 * x_error**2 -
                                       0.305 * x_error**3 -
                                       0.0283 * x_error**4 -
                                       y_error*(0.0047 - 0.0221 * x_error -
                                                0.102 * x_error**2 -
                                                0.0817 * x_error**3 -
                                                0.00717 * x_error**4))
        return abundance, abundance_error
    else:
        return abundance


def abundance_ionization_map_KD(R23, O23, plot=False, cubeid=None):
    """
    Implements the Kewley and Dopita 2002 method to calculate oxygen abundance
    and ionisation parameter.

    Parameters
    ----------
    R23 : :obj:'~numpy.ndarray'
        R23 map.

    O23 : :obj:'~numpy.ndarray'
        O23 map.

    plot : boolean
        if True, will plot the calculated oxygen abundance and ionisation
        parameter maps. It will save the plot to the working directory. Default
        is False.

    cubeid : str
        if given, it will include the cubeid in the name of the saved plots.
        Default is None.

    Returns
    -------
    initial_abundance : :obj:'~numpy.ndarray'
        calculated oxygen abundance map.

    ionization : :obj:'~numpy.ndarray'
        calculated ionisation parameter map.
    """
    initial_abundance = (m91_method(R23, O23) + z94_method(R23)) / 2.
    y_total, x_total = R23.shape
    ionization = np.full_like(R23, np.nan, dtype=np.double)
    for y in range(y_total):
        for x in range(x_total):
            if (initial_abundance[y, x] < 0 or
                    np.isnan(initial_abundance[y, x])):
                continue
            else:
                ionization_dict = {'0.05': np.array([0.0328614, -0.957421,
                                                     10.2838,
                                                     -36.9772-O23[y, x]]),
                                   '0.1': np.array([0.110773, -2.79194,
                                                    24.6206,
                                                    -74.2814-O23[y, x]]),
                                   '0.2': np.array([0.0300472, -0.914212,
                                                    10.0581,
                                                    -36.7948-O23[y, x]]),
                                   '0.5': np.array([0.128252, -3.19126,
                                                    27.5082,
                                                    -81.1880-O23[y, x]]),
                                   '1.0': np.array([0.0609004, -1.67443,
                                                    16.0880,
                                                    -52.6367-O23[y, x]]),
                                   '1.5': np.array([0.108311, -3.01747,
                                                    28.0455,
                                                    -86.8674-O23[y, x]]),
                                   '2.0': np.array([-0.0491711, 0.452486,
                                                    2.51913,
                                                    -24.4044-O23[y, x]]),
                                   '3.0': np.array([-0.232228, 4.50304,
                                                    -27.4711,
                                                    -49.4728-O23[y, x]])}
                init_z = useful.abundance2metallicity(initial_abundance[y, x])
                metallicity = (useful.closest_key(ionization_dict, init_z))
                coefficients = ionization_dict[str(metallicity)]
                solutions = np.roots(coefficients)
                real_sol = solutions[np.where(solutions.imag == 0.)[0][0]].real
                ionization[y, x] = real_sol
    if plot:
        display.plot_abundance_map(initial_abundance, save=True,
                                   system_name=gal_names[cubeid])
        display.plot_ionization_map(ionization, save=True,
                                    system_name=gal_names[cubeid])
    return initial_abundance, ionization


def initial_q(O32, initial_abundance, error=False, O32_error=None,
              initial_abundance_error=None):
    """
    Initial ionisation parameter value to be used in the implementation of the
    Kobulnicky and Kewley 2004 method to calculate oxygen abundance.

    Parameters
    ----------
    O32 : :obj:'~numpy.ndarray'
        O32 map.

    initial_abundance : :obj:'~numpy.ndarray'
        initial oxygen abundance maps.

    error : boolean
        if True, it will also calculate the error in the initial ionisation
        parameters. If True, it is necessary to provide O32 error and initial
        oxygen abundance error maps. Default is False.

    O32_error : :obj:'~numpy.ndarray'
        O32 error map. Will only be used if error is True. Default is None.

    initial_abundance_error : :obj:'~numpy.ndarray'
        Initial oxygen abundance error map. Will only be used if error is True.
        Default is None.

    Returns
    -------
    log_q : :obj:'~numpy.ndarray'
        calculated ionisation parameter map.

    log_q_error : :obj:'~numpy.ndarray'
        calculated ionisation parameter error map.
    """
    y = O32
    log_q = ((32.81 - 1.153 * y**2 +
              initial_abundance*(-3.396 - 0.025 * y + 0.1444 * y**2)) /
             (4.603 - 0.3119 * y - 0.163 * y**2 +
              initial_abundance * (-0.48 + 0.0271 * y + 0.02037 * y**2)))
    if error:
        y_error = unp.uarray(O32, O32_error)
        initial_abundance_error = unp.uarray(initial_abundance,
                                             initial_abundance_error)
        log_q_error = unp.std_devs((32.81 - 1.153 * y_error**2 +
                                    initial_abundance_error *
                                    (-3.396 - 0.025 * y_error +
                                     0.1444 * y_error**2)) /
                                   (4.603 - 0.3119 * y_error -
                                    0.163 * y_error**2 +
                                    initial_abundance_error *
                                    (-0.48 + 0.0271 * y_error +
                                     0.02037 * y_error**2)))
        return log_q, log_q_error
    else:
        return log_q


def abundance_lower(R23, log_q, error=False, R23_error=None, log_q_error=None):
    """
    Implementation of the lower branch of oxygen abundance in the Kobulnicky
    and Kewley 2004 method.

    Parameters
    ----------
    R23 : :obj:'~numpy.ndarray'
        R23 map.

    log_q : :obj:'~numpy.ndarray'
        ionisation parameter map. It must be in log scale.

    error : boolean
        if True, it will also calculate the error in the oxygen abundance. If
        True, it is necessary to provide R23 error and ionisation parameter
        error maps. Default is False.

    R23_error : :obj:'~numpy.ndarray'
        R23 error map. Will only be used if error is True. Default is None.

    log_q_error : :obj:'~numpy.ndarray'
        ionisation paramter error map. It must be in log scale. Will only be
        used if error is True. Default is None.

    Returns
    -------
    abundance : :obj:'~numpy.ndarray'
        lower abundance map.

    abundance_error : :obj:'~numpy.ndarray'
        lower abundance error map.
    """
    x = R23
    abundance = (9.40 - 4.65 * x - 3.17 * x**2 -
                 log_q*(0.272 + 0.547 * x - 0.513 * x**2))
    if error:
        x_error = unp.uarray(R23, R23_error)
        log_q_error = unp.uarray(log_q, log_q_error)
        abundance_error = unp.std_devs(9.40 - 4.65 * x_error -
                                       3.17 * x_error**2 -
                                       log_q_error*(0.272 + 0.547 * x_error -
                                                    0.513 * x_error**2))
        return abundance, abundance_error
    else:
        return abundance


def abundance_upper(R23, log_q, error=False, R23_error=None, log_q_error=None):
    """
    Implementation of the upper branch of oxygen abundance in the Kobulnicky
    and Kewley 2004 method.

    Parameters
    ----------
    R23 : :obj:'~numpy.ndarray'
        R23 map.

    log_q : :obj:'~numpy.ndarray'
        ionisation parameter map. It must be in log scale.

    error : boolean
        if True, it will also calculate the error in the oxygen abundance. If
        True, it is necessary to provide R23 error and ionisation parameter
        error maps. Default is False.

    R23_error : :obj:'~numpy.ndarray'
        R23 error map. Will only be used if error is True. Default is None.

    log_q_error : :obj:'~numpy.ndarray'
        ionisation paramter error map. It must be in log scale. Will only be
        used if error is True. Default is None.

    Returns
    -------
    abundance : :obj:'~numpy.ndarray'
        upper abundance map.

    abundance_error : :obj:'~numpy.ndarray'
        upper abundance error map.
    """
    x = R23
    abundance = (9.72 - 0.777 * x - 0.951 * x**2 - 0.072 * x**3 -
                 0.811 * x**4 - log_q*(0.0737 - 0.0713 * x - 0.141 * x**2 +
                                       0.0373 * x**3 - 0.058 * x**4))
    if error:
        x_error = unp.uarray(R23, R23_error)
        log_q_error = unp.uarray(log_q, log_q_error)
        abundance_error = unp.std_devs(9.72 - 0.777 * x_error -
                                       0.951 * x_error**2 -
                                       0.072 * x_error**3 -
                                       0.811 * x_error**4 -
                                       log_q_error*(0.0737 - 0.0713 * x_error -
                                                    0.141 * x_error**2 +
                                                    0.0373 * x_error**3 -
                                                    0.058 * x_error**4))
        return abundance, abundance_error
    else:
        return abundance


def sigma_sfr_map(hbeta_flux_map, redshift, spaxel_size, plot=False,
                  cubeid=None, error=False, hbeta_error=None,
                  redshift_error=None):
    """
    Calculates Sigma SFR given an Hbeta map, a redshift and a spaxel size.

    Parameters
    ----------
    hbeta_flux_map : :obj:'~numpy.ndarray'
        Hbeta flux map.

    redshift : float
        redshift of the relevant object.

    spaxel_size : float
        spaxel size in arcsec. It assumes that the spaxels are squares.

    plot : boolean
        if True, it will plot the Sigma SFR map and save it to the working
        directory. Default is False.

    cubeid : str
        id of the working data cube. If plot is True, it will include the cube
        id in the name of the saved plot. Default is None.

    error : boolean
        if True, it will calculate the error in the Sigma SFR. If True, it is
        necessary to include hbeta_error and redshift_error. Default is False.

    hbeta_error : :obj:'~numpy.ndarray'
        Hbeta flux error map. Only used if error is True.

    redshift_error : 2-tuple of float
        lower and upper errors in the redshift. Only used if error is True.

    Returns
    -------
    sigma_sfr_map : :obj:'~numpy.ndarray'
        Sigma SFR map.

    sigma_sfr_error_low_map : :obj:'~numpy.ndarray'
        lower Sigma SFR error map.

    sigma_sfr_error_high_map : :obj:'~numpy.ndarray'
        upper Sigma SFR error map.
    """
    scale = useful.scale_kpc_arcsec(redshift)
    sq_spaxel = spaxel_size**2 * scale**2
    SFR_map = sfr_map(hbeta_flux_map, redshift)
    sigma_sfr_map = SFR_map / sq_spaxel
    if error:
        SFR_map, SFR_map_error = sfr_map(hbeta_flux_map, redshift, error=True,
                                         hbeta_error=hbeta_error)
        SFR_error = unp.uarray(SFR_map, SFR_map_error)
        scale_low = useful.scale_kpc_arcsec(redshift - redshift_error[0])
        scale_high = useful.scale_kpc_arcsec(redshift + redshift_error[1])
        scale_error_low = ufloat(scale, scale_low)
        scale_error_high = ufloat(scale, scale_high)
        sq_spaxel_error_low = spaxel_size**2 * scale_error_low**2
        sq_spaxel_error_high = spaxel_size**2 * scale_error_high**2
        sigma_sfr_map_error_low = SFR_error / sq_spaxel_error_low
        sigma_sfr_map_error_high = SFR_error / sq_spaxel_error_high
    if plot:
        display.plot_sigma_sfr_map(sigma_sfr_map, save=True,
                                   system_name=gal_names[cubeid])
    if error:
        sigma_sfr_map = unp.nominal_values(sigma_sfr_map_error_low)
        sigma_sfr_error_low_map = unp.std_devs(sigma_sfr_map_error_low)
        sigma_sfr_error_high_map = unp.std_devs(sigma_sfr_map_error_high)
        return sigma_sfr_map, sigma_sfr_error_low_map, sigma_sfr_error_high_map
    return sigma_sfr_map


def sigma_sfr_map_oii(sfr_map, redshift, spaxel_size, plot=False, cubeid=None,
                      error=False, sfr_map_error=None, redshift_error=None):
    """
    Calculates Sigma [O II] SFR given an Hbeta map, a redshift and a spaxel
    size.

    Parameters
    ----------
    sfr_map : :obj:'~numpy.ndarray'
        [O II] SFR map.

    redshift : float
        redshift of the relevant object.

    spaxel_size : float
        spaxel size in arcsec. It assumes that the spaxels are squares.

    plot : boolean
        if True, it will plot the Sigma SFR map and save it to the working
        directory. Default is False.

    cubeid : str
        id of the working data cube. If plot is True, it will include the cube
        id in the name of the saved plot. Default is None.

    error : boolean
        if True, it will calculate the error in the Sigma SFR. If True, it is
        necessary to include hbeta_error and redshift_error. Default is False.

    sfr_map_error : :obj:'~numpy.ndarray'
        [O II] SFR error map. Only used if error is True.

    redshift_error : 2-tuple of float
        lower and upper errors in the redshift. Only used if error is True.

    Returns
    -------
    sigma_sfr_map : :obj:'~numpy.ndarray'
        Sigma SFR map.

    sigma_sfr_error_low_map : :obj:'~numpy.ndarray'
        lower Sigma SFR error map.

    sigma_sfr_error_high_map : :obj:'~numpy.ndarray'
        upper Sigma SFR error map.
    """
    scale = useful.scale_kpc_arcsec(redshift)
    sq_spaxel = spaxel_size**2 * scale**2
    sigma_sfr_map = sfr_map / sq_spaxel
    if error:
        SFR_error = unp.uarray(sfr_map, sfr_map_error)
        scale_low = useful.scale_kpc_arcsec(redshift - redshift_error[0])
        scale_high = useful.scale_kpc_arcsec(redshift + redshift_error[1])
        scale_error_low = ufloat(scale, scale_low)
        scale_error_high = ufloat(scale, scale_high)
        sq_spaxel_error_low = spaxel_size**2 * scale_error_low**2
        sq_spaxel_error_high = spaxel_size**2 * scale_error_high**2
        sigma_sfr_map_error_low = SFR_error / sq_spaxel_error_low
        sigma_sfr_map_error_high = SFR_error / sq_spaxel_error_high
    if plot:
        display.plot_sigma_sfr_map(sigma_sfr_map, save=True,
                                   system_name=gal_names[cubeid], oii=True)
    if error:
        sigma_sfr_map = unp.nominal_values(sigma_sfr_map_error_low)
        sigma_sfr_error_low_map = unp.std_devs(sigma_sfr_map_error_low)
        sigma_sfr_error_high_map = unp.std_devs(sigma_sfr_map_error_high)
        return sigma_sfr_map, sigma_sfr_error_low_map, sigma_sfr_error_high_map
    return sigma_sfr_map


def abundance_ionization_map_KK(R23, O32, plot=False, cubeid=None, error=False,
                                R23_error=None, O32_error=None):
    """
    Implementation of the Kobulnicky and Kewley 2004 method to calculate oxygen
    abundance and ionisation parameter.

    Parameters
    ----------
    R23 : :obj:'~numpy.ndarray'
        R23 map.

    O32 : :obj:'~numpy.ndarray'
        O32 map.

    plot : boolean
        if True, it will plot the calculated maps and save them to the working
        directory. Default is False.

    cubeid : str
        id of the working data cube. If plot is True, it will include the cube
        id in the name of the saved plot. Default is None.

    error : boolean
        if True, it will calculate the error of the oxugen abundance and the
        ionisation parameter. If True, it is necessary to provide R23 and 032
        error maps. Default is False.

    R23_error : :obj:'~numpy.ndarray'
        R23 error map.

    O32_error : :obj:'~numpy.ndarray'
        O32 error map.

    Returns
    -------
    abundance : :obj:'~numpy.ndarray'
        oxygen abundance map.

    abundance_error : :obj:'~numpy.ndarray'
        oxygen abundance error map. Only if error is True.

    ionization : :obj:'~numpy.ndarray'
        ionisation parameter map.

    ion_error : :obj:'~numpy.ndarray'
        ionisation parameter error map. Only if error is True.
    """
    abundance = z94_method(R23)
    ionization = initial_q(O32, abundance)
    if error:
        R23_error = unp.uarray(R23, R23_error)
        O32_error = unp.uarray(O32, O32_error)
        aa, abundance_error = z94_method(R23, error=True,
                                         R23_error=unp.std_devs(R23_error))
        bb, ion_error = initial_q(O32, abundance, error=True,
                                  O32_error=unp.std_devs(O32_error),
                                  initial_abundance_error=abundance_error)
    y_total, x_total = R23.shape
    for y in range(y_total):
        for x in range(x_total):
            if abundance[y, x] < 0 or np.isnan(abundance[y, x]):
                continue
            else:
                for i in range(3):
                    if abundance[y, x] >= 8.4:
                        abundance[y, x] = ((abundance_upper(R23[y, x],
                                                            ionization[y, x])
                                           + m91_method_upper(R23, O32)[y, x])
                                           / 2.)
                        if error:
                            au, au_error = abundance_upper(
                                R23[y, x], ionization[y, x], error=True,
                                R23_error=unp.std_devs(R23_error)[y, x],
                                log_q_error=ion_error[y, x])
                            m91_upper, m91_upper_error = m91_method_upper(
                                R23, O32, error=True,
                                R23_error=unp.std_devs(R23_error),
                                O23_error=unp.std_devs(O32_error))
                            abun_upper = ufloat(au, au_error)
                            m91_upper = ufloat(m91_upper[y, x],
                                               m91_upper_error[y, x])
                            abundance_error[y, x] = ((abun_upper + m91_upper) /
                                                     2).std_dev
                    else:
                        abundance[y, x] = abundance_lower(R23[y, x],
                                                          ionization[y, x])
                        if error:
                            abun_lower, abun_lower_error = abundance_lower(
                                R23[y, x], ionization[y, x], error=True,
                                R23_error=unp.std_devs(R23_error)[y, x],
                                log_q_error=ion_error[y, x])
                            abundance_error[y, x] = abun_lower_error
                    ionization[y, x] = initial_q(O32[y, x], abundance[y, x])
                    if error:
                        miau, ion_error[y, x] = initial_q(
                            O32[y, x], abundance[y, x], error=True,
                            O32_error=unp.std_devs(O32_error)[y, x],
                            initial_abundance_error=abundance_error[y, x])
    if plot:
        display.plot_abundance_map(abundance, save=True,
                                   system_name=gal_names[cubeid])
        display.plot_ionization_map(ionization, save=True,
                                    system_name=gal_names[cubeid])
    if error:
        return abundance, abundance_error, ionization, ion_error
    else:
        return abundance, ionization


def fit_bpt(wav, spec, redshift, plot=False, var=None, coord=None,
            wav_type='air'):
    """
    Fits the lines of the BPT diagram ([O III], Hbeta, N II and Halpha).

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    redshift : float
        redshift of the object

    plot : boolean
        if True, it will plot the fit and save it to the working directory.
        Default is False.

    var : :obj:'~numpy.ndarray'
        if provided, it will take the error into consideration when calculating
        the fits. Default is None.

    coord : 2-tuple of int
        coordinates of the spaxel the spectrum belongs to. If plot is True,
        these coordinates will be included in the name of the saved plot.
        Default is None.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    Returns
    -------
    popt : :obj:'~numpy.ndarray'
        optimal parameters found by curve_fit. If the fit is unsuccesful, it
        will return 0.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix calculated by curve_fit. If the fit is unsuccesful,
        it will return 0.
    """
    if wav_type == 'air':
        model = bpt_model_air
        w_list = wl_bpt_air
    elif wav_type == 'vac':
        model = bpt_model_vac
        w_list = wl_bpt_vac
    wav_cut, spec_cut, var_cut, p0, bounds = useful.cut_spec_n(wav, spec, var,
                                                               20, w_list *
                                                               (1+redshift), 4,
                                                               cont=False,
                                                               bpt=True)
    popt, pcov = em_fitting(wav_cut, spec_cut, model, p0, bounds, var=var_cut,
                            coord=coord)
    if plot and type(popt) != int:
        display.plot_fit(wav_cut, spec_cut, popt, "all_lines", redshift,
                         coord=coord, save=True)
    return popt, pcov


def get_flux_gaussian(A, sigma, error=None):
    """
    Calculates the area under the curve of a gaussian.

    Parameters
    ----------
    A : float
        amplitude of the gaussian.

    sigma : float
        standard deviation of the gaussian

    error : 2-tuple of float
        error of the amplitude and the standard deviation. If given, it will
        calculate the error in the area under the curve. Default is None.

    Returns
    -------
    flux : float or ufloat
        calculated area under the curve of the gaussian. If error is given,
        this variable will be a ufloat.
    """
    if error is not None:
        A_error = error[0]
        sigma_error = error[1]
        A_total = ufloat(A, A_error)
        sigma_total = ufloat(sigma, sigma_error)
        flux = A_total*sigma_total*np.sqrt(2*np.pi)
    else:
        flux = A * sigma * np.sqrt(2*np.pi)
    return flux


def fluxes_bpt(popt, pcov):
    """
    Calculates the area under the curve of each of the lines necessary to make
    a BPT plot.

    Parameters
    ----------
    popt : :obj:'~numpy.ndarray'
        optimal parameters of the gaussian fit calculated by curve_fit.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix of the gaussian fit calculated by curve_fit.

    Returns
    -------
    flux_oiii : ufloat
        flux of the [O III] line.

    flux_hbeta : ufloat
        flux of the Hbeta line.

    flux_nii : ufloat
        flux of the N II line.

    flux_halpha : ufloat
        flux of the Halpha line.
    """
    flux_oiii = get_flux_gaussian(popt[0], popt[2],
                                  error=(pcov[0, 0], pcov[2, 2]))
    flux_hbeta = get_flux_gaussian(popt[3], popt[4],
                                   error=(pcov[3, 3], pcov[4, 4]))
    flux_nii = get_flux_gaussian(popt[5], popt[6],
                                 error=(pcov[5, 5], pcov[6, 6]))
    flux_halpha = get_flux_gaussian(popt[7], popt[8],
                                    error=(pcov[7, 7], pcov[8, 8]))
    return flux_oiii, flux_hbeta, flux_nii, flux_halpha


def bpt_diagram(popt, pcov, plot=False, wav_type='air'):
    """
    Calculate the flux ratios of the BPT diagram.

    Parameters
    ----------
    popt : :obj:'~numpy.ndarray'
        optimal parameters of the gaussian fit calculated by curve_fit.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix of the gaussian fit calculated by curve_fit.

    plot : boolean
        if True, it will plot a BPT diagram with the corresponding point and
        save it to the working directory. Default is False.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    Returns
    -------
    oiii_hbeta : ufloat
        [O III]/Hbeta flux ratio. It is in log scale.

    nii_halpha : ufloat
        N II/Halpha flux ratio. It is in log scale.
    """
    flux_oiii, flux_hbeta, flux_nii, flux_halpha = fluxes_bpt(popt, pcov)
    oiii_hbeta = unp.log10(flux_oiii/flux_hbeta)
    nii_halpha = unp.log10(flux_nii/flux_halpha)
    if wav_type == 'air':
        oiii_wav = dict_wav_air['OIII_r']
    elif wav_type == 'vac':
        oiii_wav = dict_wav_vac['OIII_r']
    redshift = popt[1] / oiii_wav - 1
    if plot:
        display.bpt_diagram(oiii_hbeta, nii_halpha, redshift, save=False)
    return oiii_hbeta, nii_halpha


def tanh_model(x, *params):
    """
    Calculates a tanh model given a set of parameters.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        radius values of the model.

    *params : :obj:'~numpy.ndarray'
        model parameters. The must be in the following order: vmax (maximum
        velocity of the rotation curve) and r_v (turnover radius).

    Returns
    -------
    y : :obj:'~numpy.ndarray'
        y-values of the model
    """
    vmax, r_v = params
    y = vmax * np.tanh(x / r_v)
    return y


def fit_velocity_curve(radius, velocity, error=None, model='tanh',
                       one_way=True, plot=True):
    """
    fits a velocity curve to a given data.

    Parameters
    ----------
    radius : :obj:'~numpy.ndarray'
        radius values for the fit.

    velocity : :obj:'~numpy.ndarray'
        velocity values for the fit.

    error : :obj:'~numpy.ndarray'
        if not None, it will include the error in the model fit. Default is
        None.

    model : str
        model used to fit the velocity curve. Default is "tanh".

    one_way : boolean
        if True, it will assume the radius values are higher or equal to zero,
        and will mirror them on the other side of the plot. Default is True.

    plot : boolean
        if True, it will plot the fitted velocity curve and save it to the
        working directory. Default is True.

    Returns
    -------
    popt : :obj:'~numpy.ndarray'
        optimal parameters found by curve_fit. If the fit is unsuccesful, it
        will return 0.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix calculated by curve_fit. If the fit is unsuccesful,
        it will return 0.
    """
    if model == 'tanh':
        curve_model = tanh_model
    if one_way:
        radius = np.concatenate((radius, -radius))
        velocity = np.concatenate((velocity, -velocity))
        evel = np.concatenate((error, error))
    p0 = [np.max(velocity), 1]
    bounds = [[0, 0], [2000, 40]]
    try:
        if error is None:
            popt, pcov = curve_fit(curve_model, radius, velocity, p0=p0,
                                   bounds=bounds)
        else:
            popt, pcov = curve_fit(curve_model, radius, velocity, p0=p0,
                                   bounds=bounds, sigma=np.sqrt(evel))
        return popt, pcov
    except RuntimeError:
        print('Fit failed')
        return 0, 0


def polyfit_1_x1(*params, axis=0):
    """
    Applies a 1-degree polyfit to a given data.

    Parameters
    ----------
    *params : :obj:'~numpy.ndarray'
        parameters for the model. They must come in this order: x (x values or
        the fit), y (y values for the fit) and error (error of each data
        point).

    axis : int
        Default is 0.

    Returns
    -------
    x1 : float
        slope of the best fitted line.
    """
    x, y, error = params
    x1, x0 = np.polyfit(x, y, 1, w=1./error)
    return x1


def polyfit_1_x0(*params, axis=0):
    """
    Applies a 1-degree polyfit to a given data.

    Parameters
    ----------
    *params : :obj:'~numpy.ndarray'
        parameters for the model. They must come in this order: x (x values or
        the fit), y (y values for the fit) and error (error of each data
        point).

    axis : int
        Default is 0.

    Returns
    -------
    x0 : float
        y-axis interception of the best fitted line.
    """
    x, y, error = params
    x1, x0 = np.polyfit(x, y, 1, w=1./error)
    return x0


def bootstrap_polyfit_1(x, y, error):
    """
    applies bootstrap to a given data. It fits a 1-degree polynomial.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values to fit.

    y : :obj:'~numpy.ndarray'
        y-values to fit.

    error : :obj:'~numpy.ndarray'
        error of each data point.

    Returns
    -------
    res_x1 : :obj:'~numpy.ndarray'
        resampling of the slope of the best fitted lines.

    res_x0 : :obj:'~numpy.ndarray'
        resampling of the y-axis interception of the best fitted lines.
    """
    data = (x, y, error)
    random.seed(10)
    res_x1 = bootstrap(data, polyfit_1_x1, method='basic', paired=True,
                       vectorized=False, confidence_level=0.68)
    random.seed(10)
    res_x0 = bootstrap(data, polyfit_1_x0, method='basic', paired=True,
                       vectorized=False, confidence_level=0.68)
    return res_x1, res_x0


def get_confidence_levels_bootstrap_polyfit_1(x_values, dist_x1, dist_x0,
                                              confidence_level=0.95, num=1000):
    """
    Get the confidence level values of a boostrap resampling. DOESN'T WORK.

    Parameters
    ----------
    x_values : :obj:'~numpy.ndarray'
        x-values of the original sample.

    dist_x1 : :obj:'~numpy.ndarray'
        resampled distribution of the slope of the best fitted lines.

    dist_x0 : :obj:'~numpy.ndarray'
        resampled distribution of the y-axis interception of the best fitted
        lines.

    confidence_level : float
        confidence level to calculate. Must be a number between 0 and 1.
        Default is 0.95.

    num : int
        amount of values to resample. Default is 1000.

    Returns
    -------
    y_low : :obj:'~numpy.ndarray'
        lower y-values.

    y_high : :obj:'~numpy.ndarray'
        higher y-values.
    """
    y_high = np.zeros(x_values.shape)
    y_low = np.zeros(x_values.shape)
    # plt.figure()
    for i in range(len(x_values)):
        y_values = np.zeros(num)
        for j in range(num):
            dist_x1_j = random.choice(dist_x1)
            dist_x0_j = random.choice(dist_x0)
            y_values[j] = dist_x1_j * x_values[i] + dist_x0_j
        # print(y_values)
        # data = (y_values,)
        # res = bootstrap(data, np.mean, confidence_level=confidence_level,
        #                 method='basic', n_resamples=1000)
        # y_low[i] = res.confidence_interval.low
        # y_high[i] = res.confidence_interval.high
        # plt.scatter(np.full_like(x_values, x_values), y_values)
        y_low[i], y_high[i] = np.percentile(y_values, [34., 66.])
    return y_low, y_high


def bootstrap_polyfit_1_mine(x, y, error, num=1000):
    """
    own implementation of bootstrap using a 1-degree polyfit.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-values of the data points.

    y : :obj:'~numpy.ndarray'
        y-values of the data points.

    error : :obj:'~numpy.ndarray'
        error of the data points.

    num : int
        amount of resamples.

    Returns
    -------
    x1_dist : :obj:'~numpy.ndarray'
        resampling of the slope of the best fitted lines.

    x0_dist : :obj:'~numpy.ndarray'
        resampling of the y-axis interception of the best fitted lines.
    """
    x0_dist = np.zeros((num))
    x1_dist = np.zeros((num))
    for i in range(num):
        new_x = np.zeros((len(x)))
        new_y = np.zeros((len(x)))
        new_error = np.zeros((len(x)))
        for j in range(len(x)):
            rand_sample = random.randint(0, len(x)-1)
            new_x[j] = x[rand_sample]
            new_y[j] = y[rand_sample]
            new_error[j] = error[rand_sample]
        x1, x0 = np.polyfit(new_x, new_y, 1, w=1./new_error)
        x0_dist[i] = x0
        x1_dist[i] = x1
    return x1_dist, x0_dist
