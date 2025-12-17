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


time_re93 = df_re93['time']; intLen_re93 = df_re93['intLen']
time_re105 = df_re105['time']; intLen_re105 = df_re105['intLen']
time_re145 = df_re145['time']; intLen_re145 = df_re145['intLen']

time_list = [time_re93, time_re105, time_re145]
intLen_list = [intLen_re93, intLen_re105, intLen_re145]

fig_intLen, ax_intLen = plotting_LKB.intLen(time_list, intLen_list)
plt.show()