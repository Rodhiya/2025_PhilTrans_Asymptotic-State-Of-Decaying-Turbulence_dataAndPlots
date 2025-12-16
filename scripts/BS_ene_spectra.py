import numpy as np
import matplotlib.pyplot as plt
import script as sc
import matplotlib as mpl
import plotting_BS
mpl.rcParams['text.usetex'] = True


ene_spect_k2_re105 = np.load('data/BS_spectra_array_re105.npy')
karr_re105_k2 = np.load('data/BS_karr_re105.npy')
time_re105_k2 = np.load('data/BS_time_norm_re105.npy')
intLen_re105_k2 = np.load('data/BS_intLen_re105.npy')

cols = ['time', 'totE', 'decay_exp', 'intLen', 'Lambda', 'kol', 'reLam']


fig_ene_spect, ax_ene_spect = plotting_BS.ene_spect(karr_re105_k2, ene_spect_k2_re105, time_re105_k2, intLen_re105_k2)

nu = 5e-5
fig_ene_spect_norm, ax_ene_spect_norm = plotting_BS.ene_spect_norm(karr_re105_k2, ene_spect_k2_re105, time_re105_k2, intLen_re105_k2, nu)

spectral_slope_raw = np.load('data/BS_spectralSlope_re105.npz')
spectral_slope_karr_raw = np.load('data/BS_spectralSlope_karr_re105.npz')
spectral_slope_list = [spectral_slope_raw[key] for key in spectral_slope_raw]
spectral_slope_karr_list = [spectral_slope_karr_raw[key] for key in spectral_slope_karr_raw]

fig_ene_spect_slope, ax_ene_spect_slope = plotting_BS.ene_spect_slope(spectral_slope_karr_list, spectral_slope_list, time_re105_k2)
plt.show()
