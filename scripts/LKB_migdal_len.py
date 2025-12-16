import numpy as np
import matplotlib.pyplot as plt
import plotting_LKB
import pandas as pd
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True


LKB_migdal_len_re93 = np.load('data/LKB_Migdal_len_re93.npy')
LKB_migdal_len_re105 = np.load('data/LKB_Migdal_len_re105.npy')
LKB_migdal_len_re145 = np.load('data/LKB_Migdal_len_re145.npy')

cols = ['time', 'MigdalLen']

df_re93 = pd.DataFrame(LKB_migdal_len_re93, columns=cols)
df_re105 = pd.DataFrame(LKB_migdal_len_re105, columns=cols)
df_re145 = pd.DataFrame(LKB_migdal_len_re145, columns=cols)

time_re93 = df_re93['time']; MigdalLen_re93 = df_re93['MigdalLen']
time_re105 = df_re105['time']; MigdalLen_re105 = df_re105['MigdalLen']
time_re145 = df_re145['time']; MigdalLen_re145 = df_re145['MigdalLen']


time_list = [time_re93, time_re105, time_re145]
migdal_len_list = [MigdalLen_re93, MigdalLen_re105, MigdalLen_re145]


fig_migdal_len, ax_migdal_len = plotting_LKB.migdal_len(time_list, migdal_len_list)

fig_parabolaFit, ax_parabolaFit = plotting_LKB.migdalLen_parabolaFit(df_re145['time'], df_re145['MigdalLen'])

plt.show()
