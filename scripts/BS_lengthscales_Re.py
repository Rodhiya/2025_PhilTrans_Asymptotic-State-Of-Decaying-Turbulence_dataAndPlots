import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import plotting_BS
mpl.rcParams['text.usetex'] = True

# load data
print('1. Loading data...')
BS_results_data_re30 = np.load('data/BS_results_data_re30.npy', allow_pickle=True)
BS_results_data_re45 = np.load('data/BS_results_data_re45.npy', allow_pickle=True)
BS_results_data_re70 = np.load('data/BS_results_data_re70.npy', allow_pickle=True)
BS_results_data_re93 = np.load('data/BS_results_data_re93.npy', allow_pickle=True)
BS_results_data_re105 = np.load('data/BS_results_data_re105.npy', allow_pickle=True)
BS_results_data_re145 = np.load('data/BS_results_data_re145.npy', allow_pickle=True)
print('  Data loaded successfully!')

# Define your original columns
cols = ['time', 'totE', 'decay_exp', 'intLen', 'Lambda', 'kol', 'reLam']

# Create the DataFrame
print('2. Creating DataFrames...')
df_re30 = pd.DataFrame(BS_results_data_re30, columns=cols)
df_re45 = pd.DataFrame(BS_results_data_re45, columns=cols)
df_re70 = pd.DataFrame(BS_results_data_re70, columns=cols)
df_re93 = pd.DataFrame(BS_results_data_re93, columns=cols)
df_re105 = pd.DataFrame(BS_results_data_re105, columns=cols)
df_re145 = pd.DataFrame(BS_results_data_re145, columns=cols)

print('3(a) Creating time and Integral length scale lists...')
time_list = [df_re30['time'], df_re45['time'], df_re70['time'], df_re93['time'], df_re105['time'], df_re145['time']]
intLen_list = [df_re30['intLen'], df_re45['intLen'], df_re70['intLen'], df_re93['intLen'], df_re105['intLen'], df_re145['intLen']]
print('3(b) Plotting Integral length scale...')
fig_intLen, ax_intLen = plotting_BS.intLen(time_list, intLen_list)

print('4(a) Creating time and Lambda lists...')
Lambda_list = [df_re30['Lambda'], df_re45['Lambda'], df_re70['Lambda'], df_re93['Lambda'], df_re105['Lambda'], df_re145['Lambda']]
print('4(b) Plotting Lambda...')
fig_lambda, ax_lambda = plotting_BS.lambda_plot(time_list, Lambda_list)

print('5(a) Creating time and Kolmogorov length scale lists...')
kol_list = [df_re30['kol'], df_re45['kol'], df_re70['kol'], df_re93['kol'], df_re105['kol'], df_re145['kol']]
print('5(b) Plotting Kolmogorov length scale...')
fig_kol, ax_kol = plotting_BS.kol(time_list, kol_list)

print('6(a) Creating time and Reynolds length scale lists...')
reLam_list = [df_re30['reLam'], df_re45['reLam'], df_re70['reLam'], df_re93['reLam'], df_re105['reLam'], df_re145['reLam']]
print('6(b) Plotting Reynolds length scale...')
fig_reLam, ax_reLam = plotting_BS.reLam(time_list, reLam_list)

plt.show()

