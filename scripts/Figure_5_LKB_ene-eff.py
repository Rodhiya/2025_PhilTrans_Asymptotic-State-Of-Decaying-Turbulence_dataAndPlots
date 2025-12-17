import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import plotting_LKB
mpl.rcParams['text.usetex'] = True

# load data
LKB_results_data_re93 = np.load('../data/LKB_results_data_re93.npy', allow_pickle=True)
LKB_results_data_re105 = np.load('../data/LKB_results_data_re105.npy', allow_pickle=True)
LKB_results_data_re145 = np.load('../data/LKB_results_data_re145.npy', allow_pickle=True)

# Define your original columns
cols = ['time', 'totE', 'decay_exp', 'intLen']


# Create the DataFrame
df_re93 = pd.DataFrame(LKB_results_data_re93, columns=cols)
df_re105 = pd.DataFrame(LKB_results_data_re105, columns=cols)
df_re145 = pd.DataFrame(LKB_results_data_re145, columns=cols)


time_re93 = df_re93['time']; totE_re93 = df_re93['totE']; decay_exp_re93 = df_re93['decay_exp']
time_re105 = df_re105['time']; totE_re105 = df_re105['totE']; decay_exp_re105 = df_re105['decay_exp']
time_re145 = df_re145['time']; totE_re145 = df_re145['totE']; decay_exp_re145 = df_re145['decay_exp']

time_list = [time_re93, time_re105, time_re145]
totE_list = [totE_re93, totE_re105, totE_re145]
decay_exp_list = [decay_exp_re93, decay_exp_re105, decay_exp_re145]

fig_totE, ax_totE = plotting_LKB.totE(time_list, totE_list)

fig_eff, ax_eff = plotting_LKB.eff(time_list, decay_exp_list)

plt.show()



