import numpy as np
import matplotlib.pyplot as plt
import plotting_BS
import pandas as pd
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True


BS_migdal_len_re30 = np.load('data/BS_Migdal_len_re30.npy')
BS_migdal_len_re45 = np.load('data/BS_Migdal_len_re45.npy')
BS_migdal_len_re70 = np.load('data/BS_Migdal_len_re70.npy')
BS_migdal_len_re93 = np.load('data/BS_Migdal_len_re93.npy')
BS_migdal_len_re105 = np.load('data/BS_Migdal_len_re105.npy')
BS_migdal_len_re145 = np.load('data/BS_Migdal_len_re145.npy')

cols = ['time', 'MigdalLen']

df_re30 = pd.DataFrame(BS_migdal_len_re30, columns=cols)
df_re45 = pd.DataFrame(BS_migdal_len_re45, columns=cols)
df_re70 = pd.DataFrame(BS_migdal_len_re70, columns=cols)
df_re93 = pd.DataFrame(BS_migdal_len_re93, columns=cols)
df_re105 = pd.DataFrame(BS_migdal_len_re105, columns=cols)
df_re145 = pd.DataFrame(BS_migdal_len_re145, columns=cols)

time_re30 = df_re30['time']; MigdalLen_re30 = df_re30['MigdalLen']
time_re45 = df_re45['time']; MigdalLen_re45 = df_re45['MigdalLen']
time_re70 = df_re70['time']; MigdalLen_re70 = df_re70['MigdalLen']
time_re93 = df_re93['time']; MigdalLen_re93 = df_re93['MigdalLen']
time_re105 = df_re105['time']; MigdalLen_re105 = df_re105['MigdalLen']
time_re145 = df_re145['time']; MigdalLen_re145 = df_re145['MigdalLen']


time_list = [time_re30, time_re45, time_re70, time_re93, time_re105, time_re145]
migdal_len_list = [MigdalLen_re30, MigdalLen_re45, MigdalLen_re70, MigdalLen_re93, MigdalLen_re105, MigdalLen_re145]


fig_migdal_len, ax_migdal_len = plotting_BS.migdal_len(time_list, migdal_len_list)

fig_parabolaFit, ax_parabolaFit = plotting_BS.migdalLen_parabolaFit(df_re145['time'], df_re145['MigdalLen'])

BS_results_data_re145 = np.load('data/BS_results_data_re145.npy', allow_pickle=True)
cols_ene = ['time', 'totE', 'decay_exp', 'intLen', 'Lambda', 'kol', 'reLam']
df_re145_ene = pd.DataFrame(BS_results_data_re145, columns=cols_ene)
time_re145_ene = df_re145_ene['time']; totE_re145_ene = df_re145_ene['totE']


fig_migdal_lenVSene, ax_migdal_lenVSene = plotting_BS.migdal_lenVSene(time_re145_ene[:], MigdalLen_re145[:], totE_re145_ene[:])


plt.show()