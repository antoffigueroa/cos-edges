#!/usr/bin/env python

import numpy as np

import prepare
import useful
import apply_ppxf
import analysis

cubeid = '00045'

redshift_gal = {'00045': 0.0432, '00046': 0.04308, '00047': 0.098485, '00047b': 0.09881}

data_path = "../../group_project/modules/no_cont_"+cubeid+"data.fits"
var_path = "../../group_project/modules/no_cont_"+cubeid+"var.fits"
redshift = redshift_gal[cubeid]

# data_path = "../data/kb220304_"+cubeid+"_icubes_proj.fits"
# var_path = "../data/kb220304_"+cubeid+"_vcubes_proj.fits"
# redshift = 0.09881

gal_names = {'00045': 'Face-on', '00046': 'Edge-on', '00047': 'Merging'}

# cube, var, wav_raw, hdr = useful.read_cube(data_path, var_path)
# wav = prepare.bary_correction(wav_raw, hdr)

# Bin cube
# rebinned_cube, rebinned_var = prepare.binning(cube, var, wav, hdr, save=False, bin_size=3)

# MW extinction correction
# mw_corrected_cube, mw_corrected_var = prepare.milky_way_extinction_correction(wav, rebinned_cube.data, rebinned_var.data)

# Continuum subtraction
# no_cont_cube, no_cont_var = apply_ppxf.remove_continuum_cube(wav, mw_corrected_cube, mw_corrected_var, redshift)
no_cont_cube, no_cont_var, wav_raw, hdr = useful.read_cube(data_path, var_path)
wav = prepare.bary_correction(wav_raw, hdr)


# Dust extinction correction
snr_map_hbeta = useful.snr_line_cube(wav, no_cont_cube, no_cont_var, (1+redshift)*4861.333)
mask_hbeta = useful.create_mask(snr_map_hbeta, 3.)
snr_map_hgamma = useful.snr_line_cube(wav, no_cont_cube, no_cont_var, (1+redshift)*4340.471)
mask_hgamma = useful.create_mask(snr_map_hgamma, 3.)
mask_hgamma_hbeta = np.logical_and(mask_hbeta, mask_hgamma)
dcorr_data, dcorr_var = prepare.dust_extinction_correction(wav, no_cont_cube, no_cont_var, redshift, mask=mask_hgamma_hbeta)

# make masks
snr_map_oii = useful.snr_line_cube(wav, dcorr_data, dcorr_var, (1+redshift)*3726.032)
mask_oii = useful.create_mask(snr_map_oii, 5.)
snr_map_hbeta = useful.snr_line_cube(wav, dcorr_data, dcorr_var, (1+redshift)*4861.333)
mask_hbeta = useful.create_mask(snr_map_hbeta, 3.)

# Velocity maps
velocity_map = analysis.velocity_map(wav, dcorr_data, dcorr_var, redshift, mask_oii=mask_oii, mask_hbeta=mask_hbeta, plot=True, cubeid=cubeid)

# Make flux maps
# oii_flux_map, oii_error_map, hbeta_flux_map, hbeta_error_map, oiii_flux_map, oiii_error_map = analysis.flux_maps(wav, dcorr_data, dcorr_var, redshift, mask_oii, mask_hbeta, error=True, plot=True)
"""
np.save("maps_arrays/"+gal_names[cubeid]+"oii_flux_map", oii_flux_map)

# Calculate abundances
R23_map, R23_error_map = analysis.R23(oii_flux_map, oiii_flux_map, hbeta_flux_map, error=True, oii_flux_error=oii_error_map, oiii_flux_error=oiii_error_map, hbeta_flux_error=hbeta_error_map)
O23_map, O23_error_map = analysis.O23(oii_flux_map, oiii_flux_map, error=True, oii_flux_error=oii_error_map, oiii_flux_error=oiii_error_map)
abundance_map, abundance_error_map, ionization_map, ionization_error_map = analysis.abundance_ionization_map_KK(R23_map, O23_map, plot=True, cubeid=cubeid, error=True, R23_error=R23_error_map, O32_error=O23_error_map)
"""
# SFR maps

# SFR_map, SFR_map_error = analysis.sfr_map(hbeta_flux_map, redshift, plot=True, cubeid=cubeid, error=True, hbeta_error=hbeta_error_map)

# print('Total SFR (Mo/yr):')
# print(np.nansum(SFR_map))
"""
only_oii = np.logical_and(mask_oii, np.invert(mask_hbeta))
empty_mask = np.full_like(only_oii, False, dtype=bool)

only_oii_flux_map = oii_flux_map
only_oii_flux_map[~ only_oii] = np.nan

SFR_map_oii, SFR_map_oii_error = analysis.sfr_oii_map(only_oii_flux_map, redshift, plot=True, cubeid=cubeid, error=True, oii_error=oii_error_map)

print('Total SFR (Mo/yr):')
print(np.nansum(SFR_map)+np.nansum(SFR_map_oii))


wav_map = analysis.mu_map(wav, dcorr_data, dcorr_var, redshift,mask_oii=mask_oii, mask_hbeta=mask_hbeta)

# Sigma SFR maps

spaxel_size = 0.29 * 3
sigma_sfr_map, sigma_sfr_map_error_low, sigma_sfr_map_error_high = analysis.sigma_sfr_map(hbeta_flux_map, redshift, spaxel_size, plot=True, cubeid=cubeid, error=True, hbeta_error=hbeta_error_map, redshift_error=((analysis.wl_oii_l * (1 + redshift) - np.nanstd(wav_map)) / analysis.wl_oii_l - 1,(analysis.wl_oii_l * (1 + redshift) + np.nanstd(wav_map)) / analysis.wl_oii_l - 1))

number_of_spaxels = np.sum(~np.isnan(sigma_sfr_map))
number_outflow = np.sum(sigma_sfr_map > 0.1)

SFR_map_total = np.nansum(np.dstack((SFR_map,SFR_map_oii)),2)
SFR_map_total[SFR_map_total == 0] = np.nan

blue_blob_mask = np.load('../separate_galaxies/blue_blob.npy')

print('G2a SFR (Mo/yr):')
print(np.nansum(SFR_map[~blue_blob_mask]))
number_of_spaxels = np.sum(~np.isnan(sigma_sfr_map[~blue_blob_mask]))
print('G2a SigmaSFR (Mo/yr/kpc^2):')
print(np.nansum(sigma_sfr_map[~blue_blob_mask])/number_of_spaxels)

blue_blurb_mask = np.load('../separate_galaxies/blue_blurb.npy')

print('G2c SFR (Mo/yr):')
print(np.nansum(SFR_map[~blue_blurb_mask]))
number_of_spaxels = np.sum(~np.isnan(sigma_sfr_map[~blue_blurb_mask]))
print('G2c SigmaSFR (Mo/yr/kpc^2):')
print(np.nansum(sigma_sfr_map[~blue_blurb_mask])/number_of_spaxels)

sigma_sfr_map, sigma_sfr_map_error_low, sigma_sfr_map_error_high = analysis.sigma_sfr_map_oii(SFR_map_total, redshift, spaxel_size, plot=False, cubeid=cubeid, error=True, sfr_map_error=SFR_map_oii_error, redshift_error=((analysis.wl_oii_l * (1 + redshift) - np.nanstd(wav_map)) / analysis.wl_oii_l - 1,(analysis.wl_oii_l * (1 + redshift) + np.nanstd(wav_map)) / analysis.wl_oii_l - 1))

red_blob_mask = np.load('../separate_galaxies/red_blob.npy')

print('G2b SFR (Mo/yr):')
print(np.nansum(SFR_map_oii[~red_blob_mask]))
number_of_spaxels = np.sum(~np.isnan(sigma_sfr_map[~red_blob_mask]))
print('G2b SigmaSFR (Mo/yr/kpc^2):')
print(np.nansum(sigma_sfr_map[~red_blob_mask])/number_of_spaxels)

# number_of_spaxels = np.sum(~np.isnan(sigma_sfr_map))
# number_outflow = np.sum(sigma_sfr_map > 0.1)

print('Total SigmaSFR (Mo/yr/kpc^2):')
print(np.nansum(sigma_sfr_map)/number_of_spaxels)
print('Spaxels with SigmaSFR > 0.1:')
print(str(number_outflow)+'/'+str(number_of_spaxels))

"""
"""
wav_map = useful.wav_obs(3726.032 * (1 + redshift), velocity_map)
vel_map_qso = useful.vel(3726.032 * (1 + 0.04318), wav_map)

print('Max velocity:')
print(np.nanmax(vel_map_qso))
print('Min velocity: ')
print(np.nanmin(vel_map_qso))
print('Central velocity: ')
print(np.nanmean(vel_map_qso))


print('Metallicity percentiles: ')
print(useful.abundance2metallicity_KK(np.nanpercentile(abundance_map, [2.9, 5, 18.6, 50, 81.4, 95, 97.1])))

# Save the maps

# np.save("maps_arrays/"+gal_names[cubeid]+"velocity_map", velocity_map)
# np.save("maps_arrays/"+gal_names[cubeid]+"abundance_map", abundance_map)
np.save("maps_arrays/"+gal_names[cubeid]+"sigma_sfr_map", sigma_sfr_map)
# np.save("maps_arrays/"+gal_names[cubeid]+"abundance_error", abundance_error_map)



np.save("maps_arrays/"+gal_names[cubeid]+"velocity_map", velocity_map)
"""

