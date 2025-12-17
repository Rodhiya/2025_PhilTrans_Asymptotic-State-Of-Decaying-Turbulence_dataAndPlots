import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotting_BS
import plotting_LKB

mpl.rcParams['text.usetex'] = True


spect_list_raw_k2 = np.load('../data/BS_spect_init_spectra.npy.npz')
karr_list_raw_k2 = np.load('../data/BS_karr_init_spectra.npy.npz')

spect_list_k2 = [spect_list_raw_k2[key] for key in spect_list_raw_k2]
karr_list_k2 = [karr_list_raw_k2[key] for key in karr_list_raw_k2]

fig_ene_spect, ax_ene_spect = plotting_BS.ene_spect_t0(karr_list_k2, spect_list_k2)

spect_list_raw_k4 = np.load('../data/LKB_spect_init_spectra.npy.npz')
karr_list_raw_k4 = np.load('../data/LKB_karr_init_spectra.npy.npz')

spect_list_k4 = [spect_list_raw_k4[key] for key in spect_list_raw_k4]
karr_list_k4 = [karr_list_raw_k4[key] for key in karr_list_raw_k4]

fig_ene_spect, ax_ene_spect = plotting_LKB.ene_spect_t0(karr_list_k4, spect_list_k4)

plt.show()