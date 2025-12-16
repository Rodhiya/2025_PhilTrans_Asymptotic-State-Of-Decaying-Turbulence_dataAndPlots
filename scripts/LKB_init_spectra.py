import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotting_LKB

mpl.rcParams['text.usetex'] = True


spect_list_raw = np.load('data/LKB_spect_init_spectra.npy.npz')
karr_list_raw = np.load('data/LKB_karr_init_spectra.npy.npz')

spect_list = [spect_list_raw[key] for key in spect_list_raw]
karr_list = [karr_list_raw[key] for key in karr_list_raw]

fig_ene_spect, ax_ene_spect = plotting_LKB.ene_spect_t0(karr_list, spect_list)
plt.show()