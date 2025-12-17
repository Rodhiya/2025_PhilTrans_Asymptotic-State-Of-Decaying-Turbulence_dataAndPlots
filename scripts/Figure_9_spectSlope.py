import numpy as np
import matplotlib.pyplot as plt
import script as sc
import matplotlib as mpl
import plotting_BS
import plotting_LKB
mpl.rcParams['text.usetex'] = True


time_re105_k2 = np.load('../data/BS_time_norm_re105.npy')

nu = 5e-5

spectral_slope_raw_k2 = np.load('../data/BS_spectralSlope_re105.npz')
spectral_slope_karr_raw_k2 = np.load('../data/BS_spectralSlope_karr_re105.npz')
spectral_slope_list_k2 = [spectral_slope_raw_k2[key] for key in spectral_slope_raw_k2]
spectral_slope_karr_list_k2 = [spectral_slope_karr_raw_k2[key] for key in spectral_slope_karr_raw_k2]

fig_ene_spect_slope_k2, ax_ene_spect_slope_k2 = plotting_BS.ene_spect_slope(spectral_slope_karr_list_k2, spectral_slope_list_k2, time_re105_k2)


time_re105_k4 = np.load('../data/LKB_time_norm_re105.npy')

spectral_slope_raw_k4 = np.load('../data/LKB_spectralSlope_re105.npz')
spectral_slope_karr_raw_k4 = np.load('../data/LKB_spectralSlope_karr_re105.npz')
spectral_slope_list_k4 = [spectral_slope_raw_k4[key] for key in spectral_slope_raw_k4]
spectral_slope_karr_list_k4 = [spectral_slope_karr_raw_k4[key] for key in spectral_slope_karr_raw_k4]

fig_ene_spect_slope_k4, ax_ene_spect_slope_k4 = plotting_LKB.ene_spect_slope(spectral_slope_karr_list_k4, spectral_slope_list_k4, time_re105_k4)
plt.show()
