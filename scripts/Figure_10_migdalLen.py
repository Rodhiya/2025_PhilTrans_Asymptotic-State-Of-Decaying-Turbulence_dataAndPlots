import numpy as np
import matplotlib.pyplot as plt
import plotting_BS
import plotting_LKB
import pandas as pd
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True


BS_migdal_len_re30 = np.load('../data/BS_Migdal_len_re30.npy')
BS_migdal_len_re45 = np.load('../data/BS_Migdal_len_re45.npy')
BS_migdal_len_re70 = np.load('../data/BS_Migdal_len_re70.npy')
BS_migdal_len_re93 = np.load('../data/BS_Migdal_len_re93.npy')
BS_migdal_len_re105 = np.load('../data/BS_Migdal_len_re105.npy')
BS_migdal_len_re145 = np.load('../data/BS_Migdal_len_re145.npy')

cols = ['time', 'MigdalLen']

df_re30_k2 = pd.DataFrame(BS_migdal_len_re30, columns=cols)
df_re45_k2 = pd.DataFrame(BS_migdal_len_re45, columns=cols)
df_re70_k2 = pd.DataFrame(BS_migdal_len_re70, columns=cols)
df_re93_k2 = pd.DataFrame(BS_migdal_len_re93, columns=cols)
df_re105_k2 = pd.DataFrame(BS_migdal_len_re105, columns=cols)
df_re145_k2 = pd.DataFrame(BS_migdal_len_re145, columns=cols)

time_re30_k2 = df_re30_k2['time']; MigdalLen_re30_k2 = df_re30_k2['MigdalLen']
time_re45_k2 = df_re45_k2['time']; MigdalLen_re45_k2 = df_re45_k2['MigdalLen']
time_re70_k2 = df_re70_k2['time']; MigdalLen_re70_k2 = df_re70_k2['MigdalLen']
time_re93_k2 = df_re93_k2['time']; MigdalLen_re93_k2 = df_re93_k2['MigdalLen']
time_re105_k2 = df_re105_k2['time']; MigdalLen_re105_k2 = df_re105_k2['MigdalLen']
time_re145_k2 = df_re145_k2['time']; MigdalLen_re145_k2 = df_re145_k2['MigdalLen']


time_list_k2 = [time_re30_k2, time_re45_k2, time_re70_k2, time_re93_k2, time_re105_k2, time_re145_k2]
migdal_len_list_k2 = [MigdalLen_re30_k2, MigdalLen_re45_k2, MigdalLen_re70_k2, MigdalLen_re93_k2, MigdalLen_re105_k2, MigdalLen_re145_k2]


fig_migdal_len, ax_migdal_len = plotting_BS.migdal_len(time_list_k2, migdal_len_list_k2)

fig_parabolaFit, ax_parabolaFit = plotting_BS.migdalLen_parabolaFit(df_re145_k2['time'], df_re145_k2['MigdalLen'])

LKB_migdal_len_re93 = np.load('../data/LKB_Migdal_len_re93.npy')
LKB_migdal_len_re105 = np.load('../data/LKB_Migdal_len_re105.npy')
LKB_migdal_len_re145 = np.load('../data/LKB_Migdal_len_re145.npy')

cols = ['time', 'MigdalLen']

df_re93_k4 = pd.DataFrame(LKB_migdal_len_re93, columns=cols)
df_re105_k4 = pd.DataFrame(LKB_migdal_len_re105, columns=cols)
df_re145_k4 = pd.DataFrame(LKB_migdal_len_re145, columns=cols)

time_re93_k4 = df_re93_k4['time']; MigdalLen_re93_k4 = df_re93_k4['MigdalLen']
time_re105_k4 = df_re105_k4['time']; MigdalLen_re105_k4 = df_re105_k4['MigdalLen']
time_re145_k4 = df_re145_k4['time']; MigdalLen_re145_k4 = df_re145_k4['MigdalLen']


time_list_k4 = [time_re93_k4, time_re105_k4, time_re145_k4]
migdal_len_list_k4 = [MigdalLen_re93_k4, MigdalLen_re105_k4, MigdalLen_re145_k4]


fig_migdal_len, ax_migdal_len = plotting_LKB.migdal_len(time_list_k4, migdal_len_list_k4)

fig_parabolaFit, ax_parabolaFit = plotting_LKB.migdalLen_parabolaFit(df_re145_k4['time'], df_re145_k4['MigdalLen'])

plt.show()