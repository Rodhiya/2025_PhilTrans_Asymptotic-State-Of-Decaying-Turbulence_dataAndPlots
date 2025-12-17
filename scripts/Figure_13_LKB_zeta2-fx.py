import numpy as np
import plotting_LKB as pl
import matplotlib.pyplot as plt

zeta2_array = np.load('../data/LKB_zeta2_array_re145.npy')
time_array = np.load('../data/LKB_time_zeta_re145.npy')
x_array = np.load('../data/LKB_x_arr_re145.npy')
f_re145_value_all = np.load('../data/LKB_f_re145_value_all.npy')
f_re145_err_all = np.load('../data/LKB_f_re145_err_all.npy')

fig, ax = pl.zeta2(zeta2_array, time_array, x_array, f_re145_value_all)

fig_fx, ax_fx = pl.fx(f_re145_value_all, f_re145_err_all, x_array)
plt.show()

