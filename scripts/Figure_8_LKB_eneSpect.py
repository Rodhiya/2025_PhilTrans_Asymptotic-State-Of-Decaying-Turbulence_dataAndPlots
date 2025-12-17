import numpy as np
import matplotlib.pyplot as plt
import script as sc
import matplotlib as mpl
import plotting_LKB
mpl.rcParams['text.usetex'] = True


ene_spect_k4_re105 = np.load('../data/LKB_spectra_array_re105.npy')
karr_re105_k4 = np.load('../data/LKB_karr_re105.npy')
time_re105_k4 = np.load('../data/LKB_time_norm_re105.npy')
intLen_re105_k4 = np.load('../data/LKB_intLen_re105.npy')

cols = ['time', 'totE', 'decay_exp', 'intLen', 'Lambda', 'kol', 'reLam']


fig_ene_spect, ax_ene_spect = plotting_LKB.ene_spect(karr_re105_k4, ene_spect_k4_re105, time_re105_k4, intLen_re105_k4)

nu = 5e-5
fig_ene_spect_norm, ax_ene_spect_norm = plotting_LKB.ene_spect_norm(karr_re105_k4, ene_spect_k4_re105, time_re105_k4, intLen_re105_k4, nu)

plt.show()