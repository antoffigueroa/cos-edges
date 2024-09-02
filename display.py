#!/usr/bin/env python

#####################################
# Functions to display the results
#####################################

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from uncertainties import unumpy as unp
from astropy.visualization import simple_norm
from matplotlib import gridspec

import analysis
import useful
from constants_wavelengths import *

# Define constants

wl_oii_l = 3726.032
wl_oii_r = 3728.815
wl_hgamma = 4340.471
wl_hbeta = 4861.333
wl_oiii_l = 4958.911
wl_oiii_r = 5006.843

wratio2 = wl_oii_l/wl_oii_r


def plot_fit(wav, spec, popt, model, redshift, coord=None, save=False, wav_type='air'):
    """
    plots the fitted emission or absorption lines.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    popt : :obj:'~numpy.ndarray'
        optimal parameters of the data.

    model : function
        model used for the fitting.

    redshift : float
        redshift of the relevant object.

    coord : 2-tuple of int
        coordinate of the spaxel the spectrum belongs to. If given, the saved
        plot will include the coordinates in its name. Default is None.

    save : boolean
        if True, it will save the plot to a directory named model + _fits. This
        directory will be inside the working directory. Default is False.
    """
    file = model+"_fits/"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.tick_params(labelcolor='white', top='off', bottom='off', left='off',
                   right='off', color='white')
    plt.xlabel(r"Wavelength [$\AA$]")
    plt.ylabel(r"Flux [10$^{-16}$ erg/s/cm$^{2}$/arcsec$^{2}$/$\AA$]")
    if model == "hbeta_hgamma":
        fig = plot_hbeta_hgamma_fit(wav, spec, popt, fig, redshift, wav_type=wav_type)
        fig.suptitle(r"H$\gamma$ and H$\beta$ fits")
    elif model == "oii_doublet":
        fig = plot_oii_doublet(wav, spec, popt, fig, redshift, wav_type=wav_type)
        fig.suptitle(r"[O II] doublet fit")
    elif model == "all_lines":
        fig = plot_all_lines(wav, spec, popt, fig, redshift)
        fig.suptitle(r"Emission lines fits")
    if save:
        if coord is None:
            try:
                fig.savefig(file+"test.png")
            except Exception:
                os.system("mkdir "+file)
                fig.savefig(file+"test.png")
        else:
            try:
                fig.savefig(file+str(coord[0])+"-"+str(coord[1])+".png",
                            bbox_inches='tight')
            except Exception:
                os.system("mkdir "+file)
                fig.savefig(file+str(coord[0])+"-"+str(coord[1])+".png",
                            bbox_inches='tight')
    else:
        fig.show()


def plot_hbeta_hgamma_fit(wav, spec, popt, fig, redshift, coord=None,
                          save=False, file=None, wav_type='air'):
    """
    plots the Hbeta and Hgamma fits calculated by
    analysis.hbeta_hgamma_fitting

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    popt : :obj:'~numpy.ndarray'
        optimal parameters of the data.

    fig : matplotlib.figure.Figure
        matplotlib figure to plot the fits in.

    reshift : float
        redshift of the relevant object.

    coord : 2-tuple of int
        coordinate of the spaxel the spectrum belongs to. If given, the saved
        plot will include the coordinates in its name. Default is None.

    save : boolean
        if True, it will save the plot to a directory named model + _fits. This
        directory will be inside the working directory. Default is False.

    file : str
        name of the directory to save the fits in.

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure containing the plot.
    """
    if wav_type == 'air':
    	model = analysis.n_hgamma_hbeta_model_air
    elif wav_type == 'vac':
    	model = analysis.n_hgamma_hbeta_model_vac
    var = np.zeros(wav.shape)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    wav_finer = np.linspace(wav[0], wav[-1], num=10000)
    spec_fit = model(wav_finer, *popt)
    ax1.plot(wav, spec, label="Spectrum")
    ax1.plot(wav_finer, spec_fit, label=r"Best H$\gamma$ fit")
    ax1.set_xlim(wl_hgamma*(1+redshift) - 20, wl_hgamma*(1+redshift) + 20)
    ax1.legend()
    ax2.plot(wav, spec, label="Spectrum")
    ax2.plot(wav_finer, spec_fit, label=r"Best H$\beta$ fit")
    ax2.set_xlim(wl_hbeta*(1+redshift) - 20, wl_hbeta*(1+redshift) + 20)
    ax2.legend()
    return fig


def plot_oii_doublet(wav, spec, popt, fig, redshift, coord=None, save=False,
                     file=None, wav_type='air'):
    """
    plots the Hbeta and Hgamma fits calculated by
    analysis.two_gaussian_model

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    popt : :obj:'~numpy.ndarray'
        optimal parameters of the data.

    fig : matplotlib.figure.Figure
        matplotlib figure to plot the fits in.

    redshift : float
        redshift of the relevant object.

    coord : 2-tuple of int
        coordinate of the spaxel the spectrum belongs to. If given, the saved
        plot will include the coordinates in its name. Default is None.

    save : boolean
        if True, it will save the plot to a directory named model + _fits. This
        directory will be inside the working directory. Default is False.

    file : str
        name of the directory to save the fits in.

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure containing the plot.
    """
    line1_popt = np.array([popt[0], popt[1], popt[2]])
    line2_popt = np.array([popt[0]*popt[3], popt[1]/wratio2, popt[2]])
    ax1 = fig.add_subplot(111)
    wav_finer = np.linspace(wav[0], wav[-1], num=10000)
    line1_fit = analysis.gaussian_model(wav_finer, *line1_popt)
    line2_fit = analysis.gaussian_model(wav_finer, *line2_popt)
    if wav_type == 'air':
        model = two_gaussian_model_air
    else:
        model = two_gaussian_model_vac
    spec_fit = analysis.model(wav_finer, *popt)
    ax1.plot(wav, spec, label="Spectrum")
    ax1.plot(wav_finer, line1_fit, label=r"[O II]$\lambda$3727 line")
    ax1.plot(wav_finer, line2_fit, label=r"[O II]$\lambda$3729 line")
    ax1.plot(wav_finer, spec_fit, label=r"Best [O II] fit")
    ax1.set_xlim(wl_oii_l*(1+redshift) - 20, wl_oii_l*(1+redshift) + 20)
    ax1.legend()
    return fig


def plot_all_lines(wav, spec, popt, fig, redshift, coord=None, save=False,
                   file=None):
    """
    plots the Hbeta and Hgamma fits calculated by
    analysis.four_gaussian_model

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    popt : :obj:'~numpy.ndarray'
        optimal parameters of the data.

    fig : matplotlib.figure.Figure
        matplotlib figure to plot the fits in.

    redshift : float
        redshift of the relevant object.

    coord : 2-tuple of int
        coordinate of the spaxel the spectrum belongs to. If given, the saved
        plot will include the coordinates in its name. Default is None.

    save : boolean
        if True, it will save the plot to a directory named model + _fits. This
        directory will be inside the working directory. Default is False.

    file : str
        name of the directory to save the fits in.

    Returns
    -------
    fig : matplotlib.figure.Figure
        figure containing the plot.
    """
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    wav_finer = np.linspace(wav[0], wav[-1], num=10000)
    spec_fit = analysis.four_gaussian_model(wav_finer, *popt)
    ax1.step(wav, spec, label="Spectrum", color='grey', where='mid')
    ax1.plot(wav_finer, spec_fit, label=r"Best [O II]$\lambda$3727,3729 fit",
             color='red')
    ax1.set_xlim(wl_oii_l*(1+redshift) - 20, wl_oii_l*(1+redshift) + 20)
    ax1.legend(fontsize=7)
    ax2.step(wav, spec, label="Spectrum", color='grey', where='mid')
    ax2.plot(wav_finer, spec_fit, label=r"Best H$\beta$ fit", color='red')
    ax2.set_xlim(wl_hbeta*(1+redshift) - 20, wl_hbeta*(1+redshift) + 20)
    ax2.legend(fontsize=8)
    ax3.step(wav, spec, label="Spectrum", color='grey', where='mid')
    ax3.plot(wav_finer, spec_fit, label=r"Best [O III]$\lambda$5007 fit",
             color='red')
    ax3.set_xlim(wl_oiii_r*(1+redshift) - 20, wl_oiii_r*(1+redshift) + 20)
    ax3.legend(fontsize=8)
    [a.yaxis.set_ticks([]) for a in [ax2, ax3]]
    [a.yaxis.set_ticklabels([]) for a in [ax2, ax3]]
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    return fig


def plot_mgii_absorption(wav, spec, var, redshift, popt, pcov, n, system,
                         wav_type='air', save=False):
    """
    Plot Mg II absorption.

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

    popt : :obj:'~numpy.ndarray'
        optimal parameters of the model.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix of the optimal parameters of the model.

    n : int
        number of velocity components of the fit.

    system : str
        name of the system.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    save : boolean
        if True, it will save the plot to a directory named "mgii_absorption/".
        If system is not None, it will also include the name of the system in
        the name of the saved plot.
    """
    if wav_type == 'air':
        wobs = dict_wav_air['MgII_l'] * (1 + redshift)
        model = analysis.mgii_model_air
    elif wav_type == 'vac':
        wobs = dict_wav_vac['MgII_l'] * (1 + redshift)
        model = analysis.mgii_model_vac
    vel = useful.vel(wobs, wav)
    wav_finer = np.linspace(wav[0], wav[-1], num=10000)
    vel_finer = np.linspace(vel[0], vel[-1], num=10000)
    plt.figure(figsize=(10, 3))
    plt.step(vel, spec, 'black', lw=0.5)
    plt.step(vel, var, 'lightgrey')
    for i in range(n):
        component = np.array([popt[i*n], popt[i*n + 1], popt[i*n + 2],
                              popt[i*n + 3]])
        plt.plot(vel_finer, model(wav_finer, *component), label='Component '+n)
    plt.ylabel('Normalised flux', fontsize=20)
    plt.xlabel('Velocity [km/s]', fontsize=20)
    plt.text(-900, 0.2, system, fontsize=18)
    plt.xlim(-1000, 2000)
    plt.ylim(-0.1, 1.2)
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=15)
    if save:
        file = 'mgii_absorption/'
        try:
            fig.savefig(file+system+".png", bbox_inches='tight')
        except Exception:
            os.system("mkdir "+file)
            fig.savefig(file+system+".png", bbox_inches='tight')
    else:
        plt.show()


def plot_mgii_outflow(wav, spec, var, wav_mgii, spec_norm, var_mgii, popt_em,
                      popt, pcov, n, system, wav_type='air', save=False,
                      em_line='Hbeta', broad_component=False):
    """
    Plots a panel an emission line, along with its fitted components, and a
    second panel with Mg II absorption and its fitted components.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of theS data.

    var : :obj:'~numpy.ndarray'
        variance array of the data.

    wav_mgii : :obj:'~numpy.ndarray'
        wavelength array cut around the wavelength of the Mg II doublet.

    spec_norm : :obj:'~numpy.ndarray'
        normalised flux cut around the wavelength of the Mg II doublet.

    var_mgii : :obj:'~numpy.ndarray'
        variance array cut around the wavelength of the Mg II doublet.

    popt_em : :obj:'~numpy.ndarray'
        optimal parameters of the fit to the emission line.

    popt : :obj:'~numpy.ndarray'
        optimal parameters of the fit to the absorption.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix of the best fit model parameters.

    n : int
        number of velocity components in the absorption.

    system : str
        name of the relevant system.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    save : boolean
        if True, it will save the plot into a directory named "mgii_outflow/".
        Default is False.

    em_line : str
        name of the line used to fit the ISM component. Default is "Hbeta".

    broad_component : boolean
        if True, it will assume the emission has two velocity components.
        Default is False.
    """
    fig = plt.figure(figsize=(12, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    if wav_type == 'air':
        wl_em = dict_wav_air[em_line]
        if broad_component and popt_em[0] < popt_em[3]:
            z_em = popt_em[4] / wl_em - 1
        else:
            z_em = popt_em[1] / wl_em - 1
        wobs_em = dict_wav_air[em_line] * (1 + z_em)
        wl_abs = dict_wav_air['MgII_l']
        wobs_abs = wl_abs * (1 + z_em)
        model = analysis.mgii_model_air
    elif wav_type == 'vac':
        wl_em = dict_wav_vac[em_line]
        if broad_component and popt_em[0] < popt_em[3]:
            z_em = popt_em[4] / wl_em - 1
        else:
            z_em = popt_em[1] / wl_em - 1
        wobs_em = dict_wav_vac[em_line] * (1 + z_em)
        wl_abs = dict_wav_vac['MgII_l']
        wobs_abs = wl_abs * (1 + z_em)
        model = analysis.mgii_model_vac
    vel_em = useful.vel(wobs_em, wav)
    wav_finer_em = np.linspace(wav[0], wav[-1], num=10000)
    vel_finer_em = np.linspace(vel_em[0], vel_em[-1], num=10000)
    ax1.step(vel_em, spec, 'black', lw=0.5, where='mid', label='data')
    ax1.step(vel_em, var, 'lightgrey')
    if broad_component:
        if popt_em[0] > popt_em[3]:
            outflow_component = np.array([popt_em[3], popt_em[4], popt_em[5]])
            ism_component = np.array([popt_em[0], popt_em[1], popt_em[2]])
            wav_em = popt_em[1]
            sigma_em = popt_em[2]
        else:
            ism_component = np.array([popt_em[3], popt_em[4], popt_em[5]])
            outflow_component = np.array([popt_em[0], popt_em[1], popt_em[2]])
            wav_em = popt_em[4]
            sigma_em = popt_em[5]
        ax1.plot(vel_finer_em,
                 analysis.two_emissions_model(wav_finer_em, *popt_em),
                 color='grey', label=em_line+'\nemission')
        if ism_component[1] > outflow_component[1]:
            label_outflow = 'Broad\noutflow\ncomponent'
        else:
            label_outflow = 'Broad\ninflow\ncomponent'
        ax1.plot(vel_finer_em,
                 analysis.gaussian_model(wav_finer_em, *outflow_component),
                 color='lightblue', label=label_outflow, lw=2)
        ax1.plot(vel_finer_em,
                 analysis.gaussian_model(wav_finer_em, *ism_component),
                 color='#6AB374', label='ISM\ncomponent', lw=2)
    else:
        sigma_em = popt_em[2]
        ax1.plot(vel_finer_em, analysis.gaussian_model(wav_finer_em, *popt_em),
                 '#6AB374', label=em_line+' fit')
    ax1.legend(fontsize=8)
    ax1.set_ylabel(r'Flux [erg/s/cm$^2$/$\AA$]', fontsize=20)
    ax1.set_xlim(-500, 850)
    ax1.tick_params(labelsize=15)
    vel_abs = useful.vel(wobs_abs, wav_mgii)
    wav_finer_abs = np.linspace(wav_mgii[0], wav_mgii[-1], num=10000)
    vel_finer_abs = np.linspace(vel_abs[0], vel_abs[-1], num=10000)
    ax2.step(vel_abs, spec_norm, 'black', lw=0.5)
    ax2.step(vel_abs, var_mgii, 'lightgrey')
    all_components = np.zeros(wav_finer_abs.shape)
    n_outflows = 1
    n_inflows = 1
    coolors = ['#6AB374', '#797CDC', '#B56B45', '#C52184', '#2B3A67']
    for i in range(n):
        if i == 0:
            component = np.array([popt[0], wobs_abs, sigma_em,
                                  popt[0] * popt[1]])
            label_plot = 'ISM component'
        else:
            component = np.array([popt[4 * (i - 1) + 2], popt[4 * (i - 1) + 3],
                                  popt[4 * (i - 1) + 4],
                                  popt[4 * (i - 1) + 2] *
                                  popt[4 * (i - 1) + 5]])
            if wobs_abs < popt[4 * (i - 1) + 3]:
                label_plot = 'Inflow component '+str(n_inflows)
                n_inflows += 1
            else:
                label_plot = 'Outflow component '+str(i)
                n_outflows += 1
        ax2.plot(vel_finer_abs, model(wav_finer_abs, *component), coolors[i],
                 label=label_plot)
        all_components = all_components + model(wav_finer_abs, *component) - 1
    ax2.plot(vel_finer_abs, all_components + 1, 'grey')
    ax2.set_ylabel('Normalised flux', fontsize=20)
    ax2.set_xlabel('Velocity [km/s]', fontsize=20)
    ax2.text(-900, 0.2, system, fontsize=18)
    ax2.set_xlim(-1000, 2000)
    ax2.set_ylim(-0.1, 1.2)
    ax2.legend(fontsize=10)
    ax2.tick_params(labelsize=15)
    if save:
        file = 'mgii_outflow/'
        try:
            fig.savefig(file+system+".png", bbox_inches='tight')
        except Exception:
            os.system("mkdir "+file)
            fig.savefig(file+system+".png", bbox_inches='tight')
    else:
        plt.show()


def plot_emission_outflow(wav, spec, var, redshift, popt, pcov, wav_type='air',
                          em_line='Hbeta', system=None, save=False):
    """
    Plots emission lines with two velocity components: one ISM and one outflow
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

    popt : :obj:'~numpy.ndarray'
        optimal parameters of the broad emission model.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix of the optimal parameters.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    em_line : str
        name of the emission line used to perform the fit. Default is "Hbeta".

    system : str
        name of the relevant system.

    save : boolean
        if True, it will save the generated plot into a directory named
        "emission_outflow/".
    """
    if wav_type == 'air':
        wl_em = dict_wav_air[em_line]
        z_em = popt[1] / wl_em - 1
    elif wav_type == 'vac':
        wl_em = dict_wav_vac[em_line]
        z_em = popt[1] / wl_em - 1
    wav_finer = np.linspace(wav[0], wav[-1], num=10000)
    vel_finer = useful.vel(wl_em * (1 + z_em), wav_finer)
    vel_values = useful.vel(wl_em * (1 + z_em), wav)
    if popt[0] > popt[3]:
        outflow_component = np.array([popt[3], popt[4], popt[5]])
        ism_component = np.array([popt[0], popt[1], popt[2]])
    else:
        ism_component = np.array([popt[3], popt[4], popt[5]])
        outflow_component = np.array([popt[0], popt[1], popt[2]])
    plt.figure()
    plt.step(vel_values, spec, where='mid', color='black', lw=0.5)
    plt.plot(vel_finer, analysis.two_emissions_model(wav_finer, *popt),
             color='grey', label=em_line+' emission')
    plt.plot(vel_finer, analysis.gaussian_model(wav_finer, *outflow_component),
             color='lightblue', label='Broad outflow\ncomponent', lw=2)
    plt.plot(vel_finer, analysis.gaussian_model(wav_finer, *ism_component),
             color='#6AB374', label='ISM component', lw=2)
    plt.xlabel('Velocity [km/s]', fontsize=20)
    plt.ylabel(r'Flux [erg/s/cm$^2$/$\AA$]', fontsize=20)
    plt.legend(fontsize=15)
    plt.tick_params(labelsize=15)
    if save:
        file = 'emission_outflow/'
        try:
            plt.savefig(file+system+".png", bbox_inches='tight')
        except Exception:
            os.system("mkdir "+file)
            plt.savefig(file+system+".png", bbox_inches='tight')
    else:
        plt.show()


def plot_five_abs(wav, spec, var, redshift, popt, pcov, wav_type='air',
                  save=False, system=None):
    """
    Plots n sets of Mg II 2796,2803, Mg I and Fe II 2586,2600 absorption lines.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    var : :obj:'~numpy.ndarray'
        variance array of the data.

    redshift : float
        redshift pf the relevant object.

    popt : :obj:'~numpy.ndarray'
        optimal parameters of the fit.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix of the optimal parameters of the fit.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    save : boolean
        if True, it will save the generated plots into a directory named
        "abs_qso/".

    system : str
        name of the relevant system.
    """
    if wav_type == 'air':
        model = analysis.n_five_abs_model_air
        wl_abs_mgii = dict_wav_air['MgII_l']
        wl_abs_mgi = dict_wav_air['MgI']
        wl_abs_feii = dict_wav_air['FeII_l']
    elif wav_type == 'vac':
        model = analysis.n_five_abs_model_vac
        wl_abs_mgii = dict_wav_vac['MgII_l']
        wl_abs_mgi = dict_wav_vac['MgI']
        wl_abs_feii = dict_wav_vac['FeII_l']
    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(3, 1)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    wav_finer = np.linspace(wav[0], wav[-1], num=10000)
    vel_finer_mgii = useful.vel(wl_abs_mgii * (1 + redshift), wav_finer)
    vel_values_mgii = useful.vel(wl_abs_mgii * (1 + redshift), wav)
    ax1.step(vel_values_mgii, spec, 'black', lw=0.5, where='mid')
    ax1.step(vel_values_mgii, var, 'lightgrey', where='mid')
    ax1.plot(vel_finer_mgii, model(wav_finer, *popt), 'grey')
    ax1.set_ylabel('Normalised flux', fontsize=18)
    ax1.text(-900, 0.4, system, fontsize=18)
    ax1.text(1500, 0.2, 'Mg II', fontsize=18)
    ax1.set_xlim(-1000, 2000)
    ax1.set_ylim(-0.1, 1.2)
    vel_finer_mgi = useful.vel(wl_abs_mgi * (1 + redshift), wav_finer)
    vel_values_mgi = useful.vel(wl_abs_mgi * (1 + redshift), wav)
    ax2.plot(vel_finer_mgi, model(wav_finer, *popt), 'grey')
    ax2.step(vel_values_mgi, spec, 'black', lw=0.5, where='mid')
    ax2.step(vel_values_mgi, var, 'lightgrey', where='mid')
    ax2.set_ylabel('Normalised flux', fontsize=18)
    ax2.text(-900, 0.4, system, fontsize=18)
    ax2.text(1500, 0.2, 'Mg I', fontsize=18)
    ax2.set_xlim(-1000, 2000)
    ax2.set_ylim(-0.1, 1.2)
    vel_finer_feii = useful.vel(wl_abs_feii * (1 + redshift), wav_finer)
    vel_values_feii = useful.vel(wl_abs_feii * (1 + redshift), wav)
    ax3.plot(vel_finer_feii, model(wav_finer, *popt), 'grey')
    ax3.step(vel_values_feii, spec, 'black', lw=0.5, where='mid')
    ax3.step(vel_values_feii, var, 'lightgrey', where='mid')
    ax3.set_ylabel('Normalised flux', fontsize=18)
    ax3.text(-900, 0.4, system, fontsize=18)
    ax3.text(1500, 0.2, 'Fe II', fontsize=18)
    ax3.set_xlim(-1000, 2000)
    ax3.set_ylim(-0.1, 1.2)
    coolors = ['#9B5DE5', '#F15BB5', '#DCC638', '#00BBF9', '#00F5D4']
    n = int(len(popt) / 9)
    for i in range(n):
        A1 = popt[9*i]
        mu = popt[9*i + 1]
        sigma1 = popt[9*i + 2]
        A2 = popt[9*i + 3]
        A3 = popt[9*i + 4]
        sigma3 = popt[9*i + 5]
        A4 = popt[9*i + 6]
        sigma4 = popt[9*i + 7]
        A5 = popt[9*i + 8]
        comp = [A1, mu, sigma1, A2, A3, sigma3, A4, sigma4, A5]
        ax1.plot(vel_finer_mgii, model(wav_finer, *comp), coolors[i],
                 label='Component '+str(i))
        ax2.plot(vel_finer_mgi, model(wav_finer, *comp), coolors[i],
                 label='Component '+str(i))
        ax3.plot(vel_finer_feii, model(wav_finer, *comp), coolors[i],
                 label='Component '+str(i))
    ax1.legend()
    ax2.legend()
    ax3.legend()
    [a.xaxis.set_tick_params(labelsize=18) for a in [ax1, ax2, ax3]]
    [a.yaxis.set_tick_params(labelsize=18) for a in [ax1, ax2, ax3]]
    [a.xaxis.set_ticks([]) for a in [ax1, ax2]]
    [a.xaxis.set_ticklabels([]) for a in [ax1, ax2]]
    plt.xlabel('Velocity [km/s]', fontsize=20)
    plt.ylabel('Normalised flux', fontsize=20)
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    if save:
        file = 'abs_qso/'
        try:
            fig.savefig(file+system+".png", bbox_inches='tight')
        except Exception:
            os.system("mkdir "+file)
            fig.savefig(file+system+".png", bbox_inches='tight')
    else:
        plt.show()


def plot_feii(wav, spec, var, popt, pcov, z_em, sigma_em, n=1, wav_type='air',
              system=None, save=False):
    """
    Plots n sets of Fe II 2586,2600 absorption lines.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    var : :obj:'~numpy.ndarray'
        variance array of the data.

    popt : :obj:'~numpy.ndarray'
        optimal parameters of the fit.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix of the optimal parameters of the fit.

    z_em : float
        redshift of the ISM component.

    sigma_em : float
        standard deviation of the ISM component.

    n : int
        number of velocity components. Default is 1.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    system : str
        name of the relevant system. Default is None.

    save : boolean
        if True, it will save the generated plots into a directory named
        "feii_gal/".
    """
    if wav_type == 'air':
        model = analysis.feii_air
        wl_abs_mgii = dict_wav_air['MgII_l']
        wl_abs_feii = dict_wav_air['FeII_l']
    elif wav_type == 'vac':
        model = analysis.feii_vac
        wl_abs_mgii = dict_wav_vac['MgII_l']
        wl_abs_feii = dict_wav_vac['FeII_l']
    coolors = ['#6AB374', '#797CDC', '#B56B45', '#C52184', '#2B3A67']
    wav_finer = np.linspace(wav[0], wav[-1], num=10000)
    vel_finer_feii = useful.vel(wl_abs_feii * (1 + z_em), wav_finer)
    vel_values_feii = useful.vel(wl_abs_feii * (1 + z_em), wav)
    fig = plt.figure(figsize=(10, 3))
    plt.step(vel_values_feii, spec, 'black', lw=0.5, where='mid')
    plt.step(vel_values_feii, var, 'lightgrey', where='mid')
    plt.plot(vel_finer_feii, model(wav_finer, *[popt[0],
                                   wl_abs_feii * (1 + z_em),
                                   sigma_em, popt[1]]),
             label='ISM component', color=coolors[0])
    all_components = model(wav_finer, *[popt[0], wl_abs_feii * (1 + z_em),
                                        sigma_em, popt[1]])
    if n > 1:
        outflow_component = 1
        inflow_component = 1
        for i in range(n - 1):
            if (popt[4 * i + 3] / wl_abs_mgii * wl_abs_feii
                    > wl_abs_feii * (1 + z_em)):
                comp_label = 'Outflow component ' + str(outflow_component)
                outflow_component += 1
            else:
                comp_label = 'Inflow component ' + str(inflow_component)
                inflow_component += 1
            plt.plot(vel_finer_feii,
                     model(wav_finer,
                           *[popt[4 * i + 2],
                             popt[4 * i + 3] / wl_abs_mgii * wl_abs_feii,
                             popt[4 * i + 4], popt[4 * i + 5]]),
                     label=comp_label, color=coolors[i+1])
            all_components = (all_components +
                              model(wav_finer, *[popt[4 * i + 2],
                                    popt[4 * i + 3] / wl_abs_mgii *
                                    wl_abs_feii, popt[4 * i + 4],
                                    popt[4 * i + 5]])
                              - 1)
    plt.plot(vel_finer_feii, all_components, 'grey')
    plt.ylabel('Normalised flux', fontsize=18)
    plt.text(-900, 0.4, system, fontsize=18)
    plt.text(1500, 0.2, 'Fe II', fontsize=18)
    plt.xlim(-1000, 2000)
    plt.ylim(-0.1, 1.2)
    plt.legend()
    if save:
        file = 'feii_gal/'
        try:
            fig.savefig(file+system+".png", bbox_inches='tight')
        except Exception:
            os.system("mkdir "+file)
            fig.savefig(file+system+".png", bbox_inches='tight')
    else:
        plt.show()


def plot_caii(wav, spec, var, popt, pcov, z_em, sigma_em, n=1, wav_type='air',
              system=None, save=False):
    """
    Plots n sets of Ca II H&K absorption lines.

    Parameters
    ----------
    wav : :obj:'~numpy.ndarray'
        wavelength array of the data.

    spec : :obj:'~numpy.ndarray'
        flux array of the data.

    var : :obj:'~numpy.ndarray'
        variance array of the data.

    popt : :obj:'~numpy.ndarray'
        optimal parameters of the fit.

    pcov : :obj:'~numpy.ndarray'
        covariance matrix of the optimal parameters of the fit.

    z_em : float
        redshift of the ISM component.

    sigma_em : float
        standard deviation of the ISM component.

    n : int
        number of velocity components. Default is 1.

    wav_type : str
        type of wavelength. Supported types are "air" and "vac". Default is
        "air".

    system : str
        name of the relevant system. Default is None.

    save : boolean
        if True, it will save the generated plots into a directory named
        "caii_gal/".
    """
    if wav_type == 'air':
        model = analysis.caii_air
        wl_mgii = dict_wav_air['MgII_l']
        wl_caii_h = dict_wav_air['CaII_H']
        wl_caii_k = dict_wav_air['CaII_K']
    elif wav_type == 'vac':
        model = analysis.caii_vac
        wl_mgii = dict_wav_vac['MgII_l']
        wl_caii_h = dict_wav_vac['CaII_H']
        wl_caii_k = dict_wav_vac['CaII_K']
    coolors = ['#6AB374', '#797CDC', '#B56B45', '#C52184', '#2B3A67']
    wav_finer = np.linspace(wav[0], wav[-1], num=10000)
    vel_finer_caii_h = useful.vel(wl_caii_h * (1 + z_em), wav_finer)
    vel_values_caii_h = useful.vel(wl_caii_h * (1 + z_em), wav)
    vel_finer_caii_k = useful.vel(wl_caii_k * (1 + z_em), wav_finer)
    vel_values_caii_k = useful.vel(wl_caii_k * (1 + z_em), wav)
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 1)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.step(vel_values_caii_h, spec, 'black', lw=0.5, where='mid')
    ax1.step(vel_values_caii_h, var, 'lightgrey', where='mid')
    ax1.text(1500, 0.2, 'Ca II H', fontsize=18)
    ax2.step(vel_values_caii_k, spec, 'black', lw=0.5, where='mid')
    ax2.step(vel_values_caii_k, var, 'lightgrey', where='mid')
    ax2.text(1500, 0.2, 'Ca II K', fontsize=18)
    ax1.plot(vel_finer_caii_h, model(wav_finer, *[popt[0],
                                                  wl_caii_h * (1 + z_em),
                                                  sigma_em, popt[1]]),
             label='ISM component', color=coolors[0])
    ax2.plot(vel_finer_caii_k, model(wav_finer, *[popt[0],
                                                  wl_caii_h * (1 + z_em),
                                                  sigma_em, popt[1]]),
             label='ISM component', color=coolors[0])
    all_components = model(wav_finer, *[popt[0], wl_caii_h * (1 + z_em),
                                        sigma_em, popt[1]])
    if n > 1:
        outflow_component = 1
        inflow_component = 1
        for i in range(n - 1):
            if popt[4 * i + 3] / wl_mgii * wl_caii_h > wl_caii_h * (1 + z_em):
                comp_label = 'Outflow component ' + str(outflow_component)
                outflow_component += 1
            else:
                comp_label = 'Inflow component ' + str(inflow_component)
                inflow_component += 1
            ax1.plot(vel_finer_caii_h,
                     model(wav_finer, *[popt[4 * i + 2],
                           popt[4 * i + 3] / wl_mgii * wl_caii_h,
                           popt[4 * i + 4], popt[4 * i + 5]]),
                     label=comp_label, color=coolors[i+1])
            ax2.plot(vel_finer_caii_k,
                     model(wav_finer, *[popt[4 * i + 2],
                           popt[4 * i + 3] / wl_mgii * wl_caii_h,
                           popt[4 * i + 4], popt[4 * i + 5]]),
                     label=comp_label, color=coolors[i+1])
            all_components = (all_components +
                              model(wav_finer,
                                    *[popt[4 * i + 2],
                                      popt[4 * i + 3] / wl_mgii * wl_caii_h,
                                      popt[4 * i + 4], popt[4 * i + 5]])
                              - 1)
    ax1.plot(vel_finer_caii_h, all_components, 'grey')
    ax2.plot(vel_finer_caii_k, all_components, 'grey')
    [a.text(-900, 0.4, system, fontsize=18) for a in [ax1, ax2]]
    [a.set_xlim(-1000, 2000) for a in [ax1, ax2]]
    [a.set_ylim(-0.1, 1.2) for a in [ax1, ax2]]
    [a.legend() for a in [ax1, ax2]]
    plt.ylabel('Normalised flux', fontsize=18)
    if save:
        file = 'caii_gal/'
        try:
            fig.savefig(file+system+".png", bbox_inches='tight')
        except Exception:
            os.system("mkdir "+file)
            fig.savefig(file+system+".png", bbox_inches='tight')
    else:
        plt.show()


def plot_velocity_map(velocity_map, save=False, system_name=None):
    """
    plots a velocity map

    Parameters
    ----------
    velocity_map : :obj:'~numpy.ndarray'
        velocity map to be plotted.

    save : boolean
        if True, it will save the plot into maps/ + system_name + _velmap.png.
        Default is False.

    system_name
        if given, it will include the system name into the name of the saved
        plot. Default is None.
    """
    file = "maps/"
    plt.figure()
    max_vel = np.nanmax(np.abs(velocity_map))
    if max_vel > 400.:
        max_vel = 400
    ax = plt.gca()
    plt.imshow(velocity_map, cmap='seismic', vmax=max_vel, vmin=-max_vel)
    # plt.imshow(velocity_map, cmap='seismic', vmax=250, vmin=-250)
    y_total, x_total = velocity_map.shape
    if y_total > x_total:
        cbar = plt.colorbar(shrink=1., aspect=20*0.7)
    else:
        cbar = plt.colorbar(shrink=0.6, aspect=20*0.7)
    cbar.set_label('Velocity [km/s]', fontsize=18)
#    plt.xlim(4, 29)
#    plt.ylim(1, 21)
    # plt.gca().invert_yaxis()
    if y_total > x_total:
        plt.title('Velocity map', fontsize=18)
    else:
        plt.title('Velocity map', fontsize=18)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    if save:
        try:
            plt.savefig(file+system_name+'_velmap.png', bbox_inches='tight')
        except Exception:
            os.system("mkdir "+file)
            plt.savefig(file+system_name+'_velmap.png', bbox_inches='tight')
    else:
        plt.show()


def plot_sfr_map(sfr_map, save=False, system_name=None, oii=False):
    """
    plots a SFR map

    Parameters
    ----------
    sfr_map : :obj:'~numpy.ndarray'
        SFR map to be plotted.

    save : boolean
        if True, it will save the plot into maps/ + system_name + _sfrmap.png
        or maps/ + system_name + _OIIsfrmap.png. Default is False.

    system_name
        if given, it will include the system name into the name of the saved
        plot. Default is None.

    oii : boolean
        if True, it will assume the SFR is calculated from [O II]. If False, it
        will assume the SFR is calculated from Hbeta. Default is False.
    """
    file = "maps/"
    plt.figure()
    ax = plt.gca()
    plt.imshow(np.log10(sfr_map))
    y_total, x_total = sfr_map.shape
    if y_total > x_total:
        cbar = plt.colorbar(shrink=1., aspect=20*0.7)
    else:
        cbar = plt.colorbar(shrink=0.6, aspect=20*0.7)
    if oii:
        cbar.set_label(r'log(SFR$_{[O II]}$)', fontsize=18)
    else:
        cbar.set_label(r'log(SFR$_{H\beta}$)', fontsize=18)
    plt.xlim(4, 29)
    plt.ylim(1, 21)
    # plt.gca().invert_yaxis()
    if y_total > x_total:
        plt.title(system_name + ' system\nSFR map', fontsize=18)
    else:
        plt.title(system_name+' galaxy SFR map', fontsize=18)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    if save:
        try:
            if oii:
                plt.savefig(file+system_name+'_OIIsfrmap.png',
                            bbox_inches='tight')
            else:
                plt.savefig(file+system_name+'_sfrmap.png',
                            bbox_inches='tight')
        except Exception:
            os.system("mkdir "+file)
            if oii:
                plt.savefig(file+system_name+'_OIIsfrmap.png',
                            bbox_inches='tight')
            else:
                plt.savefig(file+system_name+'_sfrmap.png',
                            bbox_inches='tight')
    else:
        plt.show()


def plot_abundance_map(abundance_map, save=False, system_name=None):
    """
    plots an oxygen abundance map

    Parameters
    ----------
    abundance_map : :obj:'~numpy.ndarray'
        oxygen abundance map to be plotted.

    save : boolean
        if True, it will save the plot into maps/ + system_name +
        _abundancemap.png. Default is False.

    system_name
        if given, it will include the system name into the name of the saved
        plot. Default is None.
    """
    file = "maps/"
    plt.figure()
    ax = plt.gca()
    plt.imshow(abundance_map, vmin=8.3, vmax=9.)  # , vmax=9.5, vmin=8.5)
    y_total, x_total = abundance_map.shape
    if y_total > x_total:
        cbar = plt.colorbar(shrink=1., aspect=20*0.7)
    else:
        cbar = plt.colorbar(shrink=0.6, aspect=20*0.7)
    cbar.set_label('Abundance\n(log(O/H) + 12)', fontsize=18)
    plt.xlim(4, 29)
    plt.ylim(1, 21)
    # plt.gca().invert_yaxis()
    if y_total > x_total:
        plt.title('Abundance map', fontsize=18)
    else:
        plt.title('Abundance map', fontsize=18)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    if save:
        try:
            plt.savefig(file+system_name+'_abundancemap.png',
                        bbox_inches='tight')
        except Exception:
            os.system("mkdir "+file)
            plt.savefig(file+system_name+'_abundancemap.png',
                        bbox_inches='tight')
    else:
        plt.show()


def plot_ionization_map(ionization_map, save=False, system_name=None):
    """
    plots an ionisation parameter map

    Parameters
    ----------
    ionization_map : :obj:'~numpy.ndarray'
        ionisation parameter map to be plotted.

    save : boolean
        if True, it will save the plot into maps/ + system_name +
        _ionizationmap.png. Default is False.

    system_name
        if given, it will include the system name into the name of the saved
        plot. Default is None.
    """
    file = "maps/"
    plt.figure()
    ax = plt.gca()
    plt.imshow(ionization_map)
    y_total, x_total = ionization_map.shape
    if y_total > x_total:
        cbar = plt.colorbar(shrink=1., aspect=20*0.7)
    else:
        cbar = plt.colorbar(shrink=0.6, aspect=20*0.7)
    cbar.set_label('Ionization parameter\nlog(q)', fontsize=18)
    plt.gca().invert_yaxis()
    if y_total > x_total:
        plt.title('Ionization parameter map', fontsize=18)
    else:
        plt.title('Ionization parameter map', fontsize=18)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    if save:
        try:
            plt.savefig(file+system_name+'_ionizationmap.png',
                        bbox_inches='tight')
        except Exception:
            os.system("mkdir "+file)
            plt.savefig(file+system_name+'_ionizationmap.png',
                        bbox_inches='tight')
    else:
        plt.show()


def plot_sigma_sfr_map(sigma_sfr_map, save=False, system_name=None, oii=False):
    """
    plots a Sigma SFR map

    Parameters
    ----------
    sigma_sfr_map : :obj:'~numpy.ndarray'
        Sigma SFR map to be plotted.

    save : boolean
        if True, it will save the plot into maps/ + system_name +
        _sigmasfrmap.png or maps/ + system_name + _OIIsigmasfrmap.png. Default
        is False.

    system_name
        if given, it will include the system name into the name of the saved
        plot. Default is None.

    oii : boolean
        if True, it will assume the Sigma SFR is calculated from [O II]. If
        False, it will assume the Sigma SFR is calculated from Hbeta. Default
        is False.
    """
    file = "maps/"
    plt.figure()
    ax = plt.gca()
    plt.imshow(np.log10(sigma_sfr_map))
    y_total, x_total = sigma_sfr_map.shape
    if y_total > x_total:
        cbar = plt.colorbar(shrink=1., aspect=20*0.7)
    else:
        cbar = plt.colorbar(shrink=0.6, aspect=20*0.7)
    # cbar.set_label(r'log($\Sigma$SFR'+r'[M$_{\odot}$ yr$^{-1}$ kpc$^{-2}$])',
    #                fontsize=18)
    plt.xlim(4, 29)
    plt.ylim(1, 21)
    # plt.gca().invert_yaxis()
    if y_total > x_total:
        plt.title(r'$\Sigma$SFR map', fontsize=18)
    else:
        plt.title(r'$\Sigma$SFR map', fontsize=18)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    if save:
        try:
            if oii:
                plt.savefig(file+system_name+'_OIIsigmasfrmap.png',
                            bbox_inches='tight')
            else:
                plt.savefig(file+system_name+'_sigmasfrmap.png',
                            bbox_inches='tight')
        except Exception:
            os.system("mkdir "+file)
            if oii:
                plt.savefig(file+system_name+'_OIIsigmasfrmap.png',
                            bbox_inches='tight')
            else:
                plt.savefig(file+system_name+'_sigmasfrmap.png',
                            bbox_inches='tight')
    else:
        plt.show()


def mosaic_6maps_tight(map1, map2, map3, map4, map5, map6, xranges=None,
                       yranges=None, size=None, system_name=None,
                       im_ranges=None, im_xranges=None, im_yranges=None,
                       qa_slope=None, centers_img=None, delta=None):
    """
    plots a galaxy image and five maps: [O II] surface brightness, Sigma SFR,
    gas velocity, stellar velocity and oxygen abundance. It will display them
    into a tigh layout.

    Parameters
    ----------
    map1 : :obj:'~numpy.ndarray'
        image of the galaxy.

    map2 : :obj:'~numpy.ndarray'
        [O II] surface brightness map.

    map3 : :obj:'~numpy.ndarray'
        Sigma SFR map.

    map4 : :obj:'~numpy.ndarray'
        gas velocity map.

    map5 : :obj:'~numpy.ndarray'
        stellar velocity map.

    map6 : :obj:'~numpy.ndarray'
        oxygen abundance map.

    xranges : 2-tuple of int
        x range of the plotted maps. Default is None.

    yranges : 2-tuple of int
        y ranges of the plotted maps. Default is None.

    size : 2-tuple of int
        size of each of the plots. Default is None.

    system_name : str
        name of the system, to be included in the name of the saved plot.
        Default is None.

    im_ranges : 2-tuple of int
        coordinates of the letter (a) in the top left panel.

    im_xranges : 2-tuple of int
        x range of the plotted image. Default is None.

    im_yranges : 2-tuple of int
        y range of the plotted image. Default is None.

    qa_slope : float
        slope of the quasar angle. It must be in degrees. Default is None.

    centers_img : 2-tuple of float
        coordinates of the center of the galaxy in the image. An arrow coming
        from that center will be drawn. Default is None.

    delta : 2-tuple of float
        dx and dy of the arrow coming from the cente rof the galaxy. Default is
        None.
    """
    fig, ((p1, p2, p3), (p4, p5, p6)) = plt.subplots(2, 3,
                                                     figsize=(size[0] * 3,
                                                              size[1] * 2))
    # plot map 1
    im1 = p1.imshow(map1, aspect="auto")
    p1.text(im_xranges[0]+im_ranges[0], im_yranges[0]+im_ranges[1], '(a)',
            fontsize=18, c="white")
    p1.set_xlim(im_xranges[0], im_xranges[1])
    p1.set_ylim(map1.shape[1] - im_yranges[0], map1.shape[1] - im_yranges[1])
    p1.arrow(centers_img[0], centers_img[1], delta, -1 * qa_slope * delta,
             color='white', head_width=2.5, head_length=5)
    p1.text(centers_img[0] + delta + 5, centers_img[1] - qa_slope * delta - 3,
            'qso', fontsize=18, c="white")
    # plot map 2
    kcwinorm = simple_norm(map2 * 100.0, 'log', min_cut=0.0, max_cut=10)
    cbkcwi = plt.cm.ScalarMappable(norm=kcwinorm, cmap='inferno')
    im2 = p2.imshow(map2, aspect="auto", norm=kcwinorm,
                    cmap='inferno')  # , cmap='bwr')
    p2.text(xranges[0]+1, yranges[1]-4, '(b)', fontsize=18)
    divider = make_axes_locatable(p2)
    cax = divider.append_axes('top', size='5%', pad=0.0)
    cbar2 = fig.colorbar(cbkcwi, cax=cax, orientation='horizontal')
    cbar2.set_label('[OII] surface brightness\n[' + r'$\times$ 10$^{-16}$' +
                    r'erg/s/cm$^{2}$/arcsec$^{2}$]', fontsize=18)
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.xaxis.set_label_position('top')
    cbar2.set_ticks([0, 1, 2, 3, 4])
    cbar2.ax.tick_params(labelsize=18)
    # plot map 3
    im3 = p3.imshow(map3, aspect="auto", vmin=-2.5,
                    vmax=0.2, cmap='cividis')  # , cmap='autumn')
    p3.text(xranges[0]+1, yranges[1]-4, '(c)', fontsize=18)
    divider = make_axes_locatable(p3)
    cax = divider.append_axes('top', size='5%', pad=0.0)
    cbar3 = fig.colorbar(im3, cax=cax, orientation='horizontal')
    cbar3.set_label(r'log($\Sigma$SFR' + r'[M$_{\odot}$ yr$^{-1}$' +
                    r'kpc$^{-2}$])', fontsize=18)
    cbar3.ax.xaxis.set_ticks_position('top')
    cbar3.ax.xaxis.set_label_position('top')
    cbar3.ax.tick_params(labelsize=18)
    # plot map 4
    im4 = p4.imshow(map4, aspect="auto", vmin=-250,
                    vmax=250, cmap='bwr')  # , cmap='autumn')
    p4.text(xranges[0]+1, yranges[1]-4, '(d)', fontsize=18)
    divider = make_axes_locatable(p4)
    cax = divider.append_axes('bottom', size='5%', pad=0.0)
    cbar4 = fig.colorbar(im4, cax=cax, orientation='horizontal')
    cbar4.set_label('Gas velocity [km/s]',
                    fontsize=18)
    cbar4.ax.tick_params(labelsize=18)
    # plot map 5
    im5 = p5.imshow(map5, aspect="auto", vmin=-250, vmax=250, cmap='bwr')
    p5.text(xranges[0]+1, yranges[1]-4, '(e)', fontsize=18)
    divider = make_axes_locatable(p5)
    cax = divider.append_axes('bottom', size='5%', pad=0.0)
    cbar5 = fig.colorbar(im5, cax=cax, orientation='horizontal')
    cbar5.set_label('Stellar velocity [km/s]', fontsize=18)
    cbar5.ax.tick_params(labelsize=18)
    # plot map 6
    im6 = p6.imshow(map6, aspect="auto", vmin=8.45,
                    vmax=9)  # , cmap='autumn')
    p6.text(xranges[0]+1, yranges[1]-4, '(f)', fontsize=18)
    divider = make_axes_locatable(p6)
    cax = divider.append_axes('bottom', size='5%', pad=0.0)
    cbar6 = fig.colorbar(im6, cax=cax, orientation='horizontal')
    cbar6.set_label('12 + log(O/H)', fontsize=18)
    cbar6.ax.tick_params(labelsize=18)
    # Finish figure
    [a.set_xlim(xranges[0], xranges[1]) for a in [p2, p3, p4, p5, p6]]
    [a.set_ylim(yranges[0], yranges[1]) for a in [p2, p3, p4, p5, p6]]
    [a.yaxis.set_ticks([]) for a in [p1, p2, p3, p4, p5, p6]]
    [a.xaxis.set_ticks([]) for a in [p1, p2, p3, p4, p5, p6]]
    [a.yaxis.set_ticklabels([]) for a in [p1, p2, p3, p4, p5, p6]]
    [a.xaxis.set_ticklabels([]) for a in [p1, p2, p3, p4, p5, p6]]
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.savefig('maps_paper_'+system_name+'.png', bbox_inches='tight')
    plt.savefig('maps_paper_'+system_name+'.pdf', bbox_inches='tight')
    plt.show()


def bpt_diagram(oiii_hbeta, nii_halpha, redshift, save=False):
    """
    calculates a BPT diagram at a given redshift and includes the provided
    point or series of points.

    Paramters
    ---------
    oiii_hbeta : :obj:'~numpy.ndarray'
        [O III]/Hbeta ratio. It must be in log scale.

    nii_halpha : :obj:'~numpy.ndarray'
        N II/Halpha ratio. It must be in log scale.

    redshift : float
        redshift of the relevant object.

    save : boolean
        if True, it will save the plot in the working directory.
    """
    log_nii_halpha, log_oiii_hbeta = useful.bpt_curve(redshift)
    file = 'bpt_plots/'
    plt.figure()
    ax = plt.gca()
    plt.plot(log_nii_halpha, log_oiii_hbeta)
    plt.xlim(-2., 1.)
    plt.ylim(-1.3, 1.5)
    plt.errorbar(unp.nominal_values(nii_halpha),
                 unp.nominal_values(oiii_hbeta), fmt='o',
                 yerr=unp.std_devs(oiii_hbeta), xerr=unp.std_devs(nii_halpha),
                 ecolor='gray', elinewidth=0.5, zorder=1)
    plt.text(-1.7, -1., 'Star-forming', fontsize=18)
    plt.text(0.3, 1.3, 'AGN', fontsize=18)
    plt.xlabel(r'log([NII]/H$\alpha$)', fontsize=18)
    plt.ylabel(r'log([OIII]/H$\beta$)', fontsize=18)
    ax.tick_params(labelsize=18)
    if save:
        try:
            plt.savefig(file+'face-on_sdss.png',
                        bbox_inches='tight')
        except Exception:
            os.system("mkdir "+file)
            plt.savefig(file+'face-on_sdss.png',
                        bbox_inches='tight')
    else:
        plt.show()


def mosaic_4maps_tight(map1, map2, map3, map4, mask1, mask2, mask3, mask4,
                       xranges=None, yranges=None, size=None, system_name=None,
                       im_ranges=None, im_xranges=None, im_yranges=None,
                       qa_slope=None, centers_img=None, delta=None):
    """
    creates a figure with the image of a galaxy, [O II] surface brightness map,
    Sigma SFR map and oxygen abundance map. It also separates up to four
    galaxies depending on the provided masks.

    Parameters
    ----------
    map1 : :obj:'~numpy.ndarray'
        image of the galaxy.

    map2 : :obj:'~numpy.ndarray'
        [O II] surface brightness map.

    map3 : :obj:'~numpy.ndarray'
        Sigma SFR map.

    map4 : :obj:'~numpy.ndarray'
        oxygen abundance map.

    mask1 : :obj:'~numpy.ndarray'
        mask delimiting the spaxels of the first galaxy.

    mask2 : :obj:'~numpy.ndarray'
        mask delimiting the spaxels of the second galaxy.

    mask3 : :obj:'~numpy.ndarray'
        mask delimiting the spaxels of the third galaxy.

    mask4 : :obj:'~numpy.ndarray'
        mask delimiting the spaxels of the fourth galaxy.

    xranges : 2-tuple of int
        x range of the plotted maps. Default is None.

    yranges : 2-tuple of int
        y ranges of the plotted maps. Default is None.

    size : 2-tuple of int
        size of each of the plots. Default is None.

    system_name : str
        name of the system, to be included in the name of the saved plot.
        Default is None.

    im_ranges : 2-tuple of int
        coordinates of the letter (a) in the top left panel.

    im_xranges : 2-tuple of int
        x range of the plotted image. Default is None.

    im_yranges : 2-tuple of int
        y range of the plotted image. Default is None.

    qa_slope : float
        slope of the quasar angle. It must be in degrees. Default is None.

    centers_img : 2-tuple of float
        coordinates of the center of the galaxy in the image. An arrow coming
        from that center will be drawn. Default is None.

    delta : 2-tuple of float
        dx and dy of the arrow coming from the cente rof the galaxy. Default is
        None.
    """
    fig, ((p1, p2), (p3, p4)) = plt.subplots(2, 2, figsize=(size[0] * 2,
                                                            size[1] * 2))
    # plot map 1
    im1 = p1.imshow(map1, aspect="auto")
    circle1 = plt.Circle((69, 53), 15, color='white', fill=False,
                         linestyle='--')
    circle2 = plt.Circle((55, 25), 17, color='white', fill=False,
                         linestyle='--')
    circle3 = plt.Circle((32, 52), 14, color='white', fill=False,
                         linestyle='--')
    circle4 = plt.Circle((25, 75), 12, color='white', fill=False,
                         linestyle='--')
    p1.add_patch(circle1)
    p1.add_patch(circle2)
    p1.add_patch(circle3)
    p1.add_patch(circle4)
    p1.text(im_xranges[0]+im_ranges[0], im_yranges[0]+im_ranges[1], '(a)',
            fontsize=18, c="white")
    p1.set_xlim(im_xranges[0], im_xranges[1])
    p1.set_ylim(map1.shape[1] - im_yranges[0], map1.shape[1] - im_yranges[1])
    p1.arrow(centers_img[0], centers_img[1], delta, -1 * qa_slope * delta,
             color='white', head_width=2.5, head_length=5)
    p1.text(centers_img[0] + delta + 5, centers_img[1] - qa_slope * delta - 3,
            'qso', fontsize=18, c="white")
    p1.arrow(83, 75, -2.5, -5,
             color='white', head_width=2.5, head_length=5)
    p1.text(77, 80, 'G2a',
            fontsize=18, c="white")
    p1.arrow(28.26679947, 17., 5, 2.5,
             color='white', head_width=2.5, head_length=5)
    p1.text(16.26679947, 16., 'G2c',
            fontsize=18, c="white")
    p1.arrow(51.79795897, 72, -5, -5,
             color='white', head_width=2.5, head_length=5)
    p1.text(51.79795897, 77, 'G2b',
            fontsize=18, c="white")
    p1.arrow(46.63324958, 85, -5, -4.,
             color='white', head_width=2.5, head_length=5)
    p1.text(48.63324958, 88, 'G2d',
            fontsize=18, c="white")
    # plot map 2
    kcwinorm = simple_norm(map2 * 100.0, 'log', min_cut=0.0, max_cut=10)
    cbkcwi = plt.cm.ScalarMappable(norm=kcwinorm, cmap='inferno')
    im2 = p2.imshow(map2, aspect="auto", norm=kcwinorm,
                    cmap='inferno')  # , cmap='bwr')
    p2.text(xranges[0]+1, yranges[1]-4, '(b)', fontsize=18)
    divider = make_axes_locatable(p2)
    cax = divider.append_axes('top', size='5%', pad=0.0)
    cbar2 = fig.colorbar(cbkcwi, cax=cax, orientation='horizontal')
    cbar2.set_label('[OII] surface brightness\n['+r'$\times$ 10$^{-16}$' +
                    r'erg/s/cm$^{2}$/arcsec$^{2}$]', fontsize=18)
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.xaxis.set_label_position('top')
    cbar2.set_ticks([0, 1, 2, 3, 4])
    cbar2.ax.tick_params(labelsize=18)
    p2.contour(mask1, levels=[1])
    p2.contour(mask2, levels=[1])
    p2.contour(mask3, levels=[1])
    p2.contour(mask4, levels=[1])
    # plot map 3
    im3 = p3.imshow(map3, aspect="auto", vmin=-2.5,
                    vmax=0.2, cmap='cividis')  # , cmap='autumn')
    p3.text(xranges[0]+1, yranges[1]-4, '(c)', fontsize=18)
    divider = make_axes_locatable(p3)
    cax = divider.append_axes('bottom', size='5%', pad=0.0)
    cbar3 = fig.colorbar(im3, cax=cax, orientation='horizontal')
    cbar3.set_label(r'log($\Sigma$SFR'+r'[M$_{\odot}$ yr$^{-1}$' +
                    r'kpc$^{-2}$])', fontsize=18)
    # cbar3.ax.xaxis.set_ticks_position('top')
    # cbar3.ax.xaxis.set_label_position('top')
    cbar3.ax.tick_params(labelsize=18)
    p3.contour(mask1, levels=[1])
    p3.contour(mask2, levels=[1])
    p3.contour(mask3, levels=[1])
    p3.contour(mask4, levels=[1])
    # plot map 6
    im4 = p4.imshow(map4, aspect="auto", vmin=7.9,
                    vmax=9)  # , cmap='autumn')
    p4.text(xranges[0]+1, yranges[1]-4, '(d)', fontsize=18)
    divider = make_axes_locatable(p4)
    cax = divider.append_axes('bottom', size='5%', pad=0.0)
    cbar4 = fig.colorbar(im4, cax=cax, orientation='horizontal')
    cbar4.set_label('12 + log(O/H)', fontsize=18)
    cbar4.ax.tick_params(labelsize=18)
    p4.contour(mask1, levels=[1])
    p4.contour(mask2, levels=[1])
    p4.contour(mask3, levels=[1])
    p4.contour(mask4, levels=[1])
    # Finish figure
    [a.set_xlim(xranges[0], xranges[1]) for a in [p2, p3, p4]]
    [a.set_ylim(yranges[0], yranges[1]) for a in [p2, p3, p4]]
    [a.yaxis.set_ticks([]) for a in [p1, p2, p3, p4]]
    [a.xaxis.set_ticks([]) for a in [p1, p2, p3, p4]]
    [a.yaxis.set_ticklabels([]) for a in [p1, p2, p3, p4]]
    [a.xaxis.set_ticklabels([]) for a in [p1, p2, p3, p4]]
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.savefig('maps_paper_others_'+system_name+'.png', bbox_inches='tight')
    plt.savefig('maps_paper_others_'+system_name+'.pdf', bbox_inches='tight')
    plt.show()


def mosaic_4kinmaps_tight(map1, map2, map3, map4, mask1, mask2, mask3, mask4,
                          xranges=None, yranges=None, size=None,
                          system_name=None, im_ranges=None, im_xranges=None,
                          im_yranges=None, qa_slope=None, centers_img=None,
                          delta=None):
    """
    creates a figure with the image of a galaxy, a gas velocity map, a stellar
    velocity map and a relative velocity map. It also separates up to four
    galaxies depending on the provided masks.

    Parameters
    ----------
    map1 : :obj:'~numpy.ndarray'
        image of the galaxy.

    map2 : :obj:'~numpy.ndarray'
        gas velocity map.

    map3 : :obj:'~numpy.ndarray'
        stellar velocity map.

    map4 : :obj:'~numpy.ndarray'
        relative velocity map.

    mask1 : :obj:'~numpy.ndarray'
        mask delimiting the spaxels of the first galaxy.

    mask2 : :obj:'~numpy.ndarray'
        mask delimiting the spaxels of the second galaxy.

    mask3 : :obj:'~numpy.ndarray'
        mask delimiting the spaxels of the third galaxy.

    mask4 : :obj:'~numpy.ndarray'
        mask delimiting the spaxels of the fourth galaxy.

    xranges : 2-tuple of int
        x range of the plotted maps. Default is None.

    yranges : 2-tuple of int
        y ranges of the plotted maps. Default is None.

    size : 2-tuple of int
        size of each of the plots. Default is None.

    system_name : str
        name of the system, to be included in the name of the saved plot.
        Default is None.

    im_ranges : 2-tuple of int
        coordinates of the letter (a) in the top left panel.

    im_xranges : 2-tuple of int
        x range of the plotted image. Default is None.

    im_yranges : 2-tuple of int
        y range of the plotted image. Default is None.

    qa_slope : float
        slope of the quasar angle. It must be in degrees. Default is None.

    centers_img : 2-tuple of float
        coordinates of the center of the galaxy in the image. An arrow coming
        from that center will be drawn. Default is None.

    delta : 2-tuple of float
        dx and dy of the arrow coming from the cente rof the galaxy. Default is
        None.
    """
    fig, ((p1, p2), (p3, p4)) = plt.subplots(2, 2, figsize=(size[0] * 2,
                                                            size[1] * 2))
    # plot map 1
    im1 = p1.imshow(map1, aspect="auto")
    circle1 = plt.Circle((69, 53), 15, color='white', fill=False,
                         linestyle='--')
    circle2 = plt.Circle((55, 25), 17, color='white', fill=False,
                         linestyle='--')
    circle3 = plt.Circle((32, 52), 14, color='white', fill=False,
                         linestyle='--')
    circle4 = plt.Circle((25, 75), 12, color='white', fill=False,
                         linestyle='--')
    p1.add_patch(circle1)
    p1.add_patch(circle2)
    p1.add_patch(circle3)
    p1.add_patch(circle4)
    p1.text(im_xranges[0]+im_ranges[0], im_yranges[0]+im_ranges[1], '(a)',
            fontsize=18, c="white")
    p1.set_xlim(im_xranges[0], im_xranges[1])
    p1.set_ylim(map1.shape[1] - im_yranges[0], map1.shape[1] - im_yranges[1])
    p1.arrow(centers_img[0], centers_img[1], delta, -1 * qa_slope * delta,
             color='white', head_width=2.5, head_length=5)
    p1.text(centers_img[0] + delta + 5, centers_img[1] - qa_slope * delta - 3,
            'qso', fontsize=18, c="white")
    p1.arrow(83, 75, -2.5, -5,
             color='white', head_width=2.5, head_length=5)
    p1.text(77, 80, 'G2a',
            fontsize=18, c="white")
    p1.arrow(28.26679947, 17., 5, 2.5,
             color='white', head_width=2.5, head_length=5)
    p1.text(16.26679947, 16., 'G2c',
            fontsize=18, c="white")
    p1.arrow(51.79795897, 72, -5, -5,
             color='white', head_width=2.5, head_length=5)
    p1.text(51.79795897, 77, 'G2b',
            fontsize=18, c="white")
    p1.arrow(46.63324958, 85, -5, -4.,
             color='white', head_width=2.5, head_length=5)
    p1.text(48.63324958, 88, 'G2d',
            fontsize=18, c="white")
    # plot map 2
    im2 = p2.imshow(map2, aspect="auto", cmap='bwr', vmin=-250,
                    vmax=250)
    p2.text(xranges[0]+1, yranges[1]-4, '(b)', fontsize=18)
    divider = make_axes_locatable(p2)
    cax = divider.append_axes('top', size='5%', pad=0.0)
    cbar2 = fig.colorbar(im2, cax=cax, orientation='horizontal')
    cbar2.set_label('Gas velocity [km/s]', fontsize=18)
    cbar2.ax.xaxis.set_ticks_position('top')
    cbar2.ax.xaxis.set_label_position('top')
    cbar2.ax.tick_params(labelsize=18)
    p2.contour(mask1, levels=[1])
    p2.contour(mask2, levels=[1])
    p2.contour(mask3, levels=[1])
    p2.contour(mask4, levels=[1])
    # plot map 3
    im3 = p3.imshow(map3, aspect="auto", vmin=-650,
                    vmax=650, cmap='bwr')
    p3.text(xranges[0]+1, yranges[1]-4, '(c)', fontsize=18)
    divider = make_axes_locatable(p3)
    cax = divider.append_axes('bottom', size='5%', pad=0.0)
    cbar3 = fig.colorbar(im3, cax=cax, orientation='horizontal')
    cbar3.set_label('Stellar velocity [km/s]', fontsize=18)
    # cbar3.ax.xaxis.set_ticks_position('top')
    # cbar3.ax.xaxis.set_label_position('top')
    cbar3.ax.tick_params(labelsize=18)
    p3.contour(mask1, levels=[1])
    p3.contour(mask2, levels=[1])
    p3.contour(mask3, levels=[1])
    p3.contour(mask4, levels=[1])
    # plot map 6
    im4 = p4.imshow(map4, aspect="auto", vmin=-90,
                    vmax=90, cmap='bwr')
    p4.text(xranges[0]+1, yranges[1]-4, '(d)', fontsize=18)
    divider = make_axes_locatable(p4)
    cax = divider.append_axes('bottom', size='5%', pad=0.0)
    cbar4 = fig.colorbar(im4, cax=cax, orientation='horizontal')
    cbar4.set_label('Relative velocity [km/s]', fontsize=18)
    cbar4.ax.tick_params(labelsize=18)
    p4.contour(mask1, levels=[1])
    p4.contour(mask2, levels=[1])
    p4.contour(mask3, levels=[1])
    p4.contour(mask4, levels=[1])
    p4.text(17.5, 29.5, 'z=0.09897',
            fontsize=12, c="indigo")
    p4.arrow(20.5, 29.4, -0.7, -1.0,
             color='indigo', head_width=0.5, head_length=1)
    p4.text(-1, 22, 'z=0.09875',
            fontsize=12, c="indigo")
    p4.arrow(2, 21.8, 0.7, -2.0,
             color='indigo', head_width=0.5, head_length=1)
    p4.text(17.5, 6, 'z=0.09840',
            fontsize=12, c="indigo")
    p4.arrow(21.5, 7, -2.0, 1.5,
             color='indigo', head_width=0.5, head_length=1)
    p4.text(8, 6, 'z=0.09782',
            fontsize=12, c="indigo")
    p4.arrow(11.5, 7, -2.5, 1.0,
             color='indigo', head_width=0.5, head_length=1)
    # Finish figure
    [a.set_xlim(xranges[0], xranges[1]) for a in [p2, p3, p4]]
    [a.set_ylim(yranges[0], yranges[1]) for a in [p2, p3, p4]]
    [a.yaxis.set_ticks([]) for a in [p1, p2, p3, p4]]
    [a.xaxis.set_ticks([]) for a in [p1, p2, p3, p4]]
    [a.yaxis.set_ticklabels([]) for a in [p1, p2, p3, p4]]
    [a.xaxis.set_ticklabels([]) for a in [p1, p2, p3, p4]]
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.savefig('maps_paper_kin_'+system_name+'.png', bbox_inches='tight')
    plt.savefig('maps_paper_kin_'+system_name+'.pdf', bbox_inches='tight')
    plt.show()
