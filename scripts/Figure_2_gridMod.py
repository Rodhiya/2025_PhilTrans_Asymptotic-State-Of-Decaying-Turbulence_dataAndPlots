import numpy as np
import plotting_BS
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True


eff_ind_list_raw = np.load('../data/gridModification_effList.npy.npz')
time_list_raw = np.load('../data/gridModification_timeList.npy.npz')

eff_ind_list = [eff_ind_list_raw[key] for key in eff_ind_list_raw]
time_list = [time_list_raw[key] for key in time_list_raw]


fig, ax_eff = plotting_BS.gridMod_eff(time_list, eff_ind_list)
plt.show()