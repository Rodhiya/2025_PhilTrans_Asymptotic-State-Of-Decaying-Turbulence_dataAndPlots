import numpy as np
import matplotlib.pyplot as plt
import script as sc
import matplotlib as mpl
import plotting_BS
mpl.rcParams['text.usetex'] = True


dissipation_spectra = np.load('data/spectra_diss_spectra.npy')
karr_list = np.load('data/karr_diss_spectra.npy')
time_list = np.load('data/time_diss_spectra.npy')
Re_list = np.load('data/Re_diss_spectra.npy')

slope_raw = np.load('data/slope_diss_spectra.npy.npz')
slope_karr_raw = np.load('data/keta_diss_spectra.npy.npz')
slope_keta = np.load('data/k_eta_diss_spectra.npy')

slope_list = [slope_raw[key] for key in slope_raw]
slope_karr_list = [slope_karr_raw[key] for key in slope_karr_raw]

fig, ax_diss_spect = plotting_BS.dissipation_spectra(karr_list, dissipation_spectra, time_list, Re_list)

fig, ax_diss_spect_slope = plotting_BS.dissipation_spectra_slope(slope_karr_list, slope_list, time_list, slope_keta)
plt.show()