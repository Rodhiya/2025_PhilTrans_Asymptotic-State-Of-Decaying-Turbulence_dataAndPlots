import numpy as np
import matplotlib.pyplot as plt
import plotting_BS
import plotting_LKB
import pandas as pd
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

BS_migdal_len_re145_k2 = np.load('../data/BS_Migdal_len_re145.npy')
cols = ['time', 'MigdalLen']
df_re145_k2 = pd.DataFrame(BS_migdal_len_re145_k2, columns=cols)

time_re145_k2 = df_re145_k2['time']; MigdalLen_re145_k2 = df_re145_k2['MigdalLen']

BS_results_data_re145_k2 = np.load('../data/BS_results_data_re145.npy', allow_pickle=True)
cols_ene = ['time', 'totE', 'decay_exp', 'intLen', 'Lambda', 'kol', 'reLam']
df_re145_ene_k2 = pd.DataFrame(BS_results_data_re145_k2, columns=cols_ene)
time_re145_ene_k2 = df_re145_ene_k2['time']; totE_re145_ene_k2 = df_re145_ene_k2['totE']

fig_migdal_lenVSene, ax_migdal_lenVSene = plotting_BS.migdal_lenVSene(time_re145_ene_k2[:], MigdalLen_re145_k2[:], totE_re145_ene_k2[:])


LKB_migdal_len_re145 = np.load('../data/LKB_Migdal_len_re145.npy')
df_re145_k4 = pd.DataFrame(LKB_migdal_len_re145, columns=cols)
time_re145_k4 = df_re145_k4['time']; MigdalLen_re145_k4 = df_re145_k4['MigdalLen']
LKB_results_data_re145 = np.load('../data/LKB_results_data_re145.npy', allow_pickle=True)
cols_ene = ['time', 'totE', 'decay_exp', 'intLen']
df_re145_ene_k4 = pd.DataFrame(LKB_results_data_re145, columns=cols_ene)
time_re145_ene_k4 = df_re145_ene_k4['time']; totE_re145_ene_k4 = df_re145_ene_k4['totE']

fig_migdal_lenVSene, ax_migdal_lenVSene = plotting_LKB.migdal_lenVSene(time_re145_ene_k4[:], MigdalLen_re145_k4[:], totE_re145_ene_k4[:])

plt.show()