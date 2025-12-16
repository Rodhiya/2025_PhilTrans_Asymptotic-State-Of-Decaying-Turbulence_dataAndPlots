import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import plotting_BS
mpl.rcParams['text.usetex'] = True

# load data
BS_results_data_re30 = np.load('data/BS_results_data_re30.npy', allow_pickle=True)
BS_results_data_re45 = np.load('data/BS_results_data_re45.npy', allow_pickle=True)
BS_results_data_re70 = np.load('data/BS_results_data_re70.npy', allow_pickle=True)
BS_results_data_re93 = np.load('data/BS_results_data_re93.npy', allow_pickle=True)
BS_results_data_re105 = np.load('data/BS_results_data_re105.npy', allow_pickle=True)
BS_results_data_re145 = np.load('data/BS_results_data_re145.npy', allow_pickle=True)

# Define your original columns
cols = ['time', 'totE', 'decay_exp', 'intLen', 'Lambda', 'kol', 'reLam']

# Create the DataFrame
df_re30 = pd.DataFrame(BS_results_data_re30, columns=cols)
df_re45 = pd.DataFrame(BS_results_data_re45, columns=cols)
df_re70 = pd.DataFrame(BS_results_data_re70, columns=cols)
df_re93 = pd.DataFrame(BS_results_data_re93, columns=cols)
df_re105 = pd.DataFrame(BS_results_data_re105, columns=cols)
df_re145 = pd.DataFrame(BS_results_data_re145, columns=cols)


time_re30 = df_re30['time']; totE_re30 = df_re30['totE']; decay_exp_re30 = df_re30['decay_exp']
time_re45 = df_re45['time']; totE_re45 = df_re45['totE']; decay_exp_re45 = df_re45['decay_exp']
time_re70 = df_re70['time']; totE_re70 = df_re70['totE']; decay_exp_re70 = df_re70['decay_exp']
time_re93 = df_re93['time']; totE_re93 = df_re93['totE']; decay_exp_re93 = df_re93['decay_exp']
time_re105 = df_re105['time']; totE_re105 = df_re105['totE']; decay_exp_re105 = df_re105['decay_exp']
time_re145 = df_re145['time']; totE_re145 = df_re145['totE']; decay_exp_re145 = df_re145['decay_exp']

time_list = [time_re30, time_re45, time_re70, time_re93, time_re105, time_re145]
totE_list = [totE_re30, totE_re45, totE_re70, totE_re93, totE_re105, totE_re145]
decay_exp_list = [decay_exp_re30, decay_exp_re45, decay_exp_re70, decay_exp_re93, decay_exp_re105, decay_exp_re145]



fig_totE, ax_totE = plotting_BS.totE(time_list, totE_list)

fig_eff, ax_eff = plotting_BS.decay_exp(time_list, decay_exp_list)

plt.show()