import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
import matplotlib as mpl
import script as sc
mpl.rcParams['text.usetex'] = True

def totE(time_list, totE_list):
    fig = plt.figure(figsize=(6.5,5))
    ax = fig.add_subplot(1,1,1)
    time_re93 = time_list[0]; totE_re93 = totE_list[0]
    time_re105 = time_list[1]; totE_re105 = totE_list[1]
    time_re145 = time_list[2]; totE_re145 = totE_list[2]

    ax.loglog(time_re93, totE_re93/totE_re93[0], label='$Re_{\lambda}=93$', color = '#1f77b4')
    ax.loglog(time_re105, totE_re105/totE_re105[0], label='$Re_{\lambda}=105$', color = '#ff7f0e')
    ax.loglog(time_re145, totE_re145/totE_re145[0], label='$Re_{\lambda}=145$', color = '#2ca02c')


    cl=5
    s=50
    f=90000
    ax.loglog(time_re93[s:f], cl*(time_re93[s:f])**(-10/7), 'k-.')
    ax.text(0.6, 0.6, 'Slope = -10/7', fontsize=20, color='k', transform=ax.transAxes)
    # --- MODIFIED TICK PARAMETERS START ---

    # 1. Set major ticks to be visible on all four sides (top, bottom, left, right)
    ax.tick_params(axis='both', which='major', labelsize=16,
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=6, width=1.2) # Optional: customize major tick appearance

    # 2. Add minor ticks and ensure they are also on all four sides
    ax.tick_params(which='minor',
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=3, width=0.75) # Optional: customize minor tick appearance


    ax.set_xlabel('$t/T_{eddy,0}$', fontsize=20)
    ax.set_ylabel('$E_n/E_{n(t=0)}$', fontsize=20)
    ax.set_yticks([1e-6,1e-4,1e-2,1e0])
    ax.set_title('(a) Kinetic energy $E_n(t)$', fontsize=22)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=20, loc='lower left')
    ax.set_xlim(1e-1,2.5e4)
    ax.set_ylim(1e-6,1e1)
    plt.tight_layout()
    plt.savefig('figures/Figure_5a_totE_LKB.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax

def eff(time_list, eff_ind_list):
    fig = plt.figure(figsize=(6.5,5))
    ax = fig.add_subplot(1,1,1)

    time_re93 = time_list[0]; eff_ind_re93 = eff_ind_list[0]
    time_re105 = time_list[1]; eff_ind_re105 = eff_ind_list[1]
    time_re145 = time_list[2]; eff_ind_re145 = eff_ind_list[2]

    # --- MODIFIED TICK PARAMETERS START ---

    # 1. Set major ticks to be visible on all four sides (top, bottom, left, right)
    ax.tick_params(axis='both', which='major', labelsize=14,
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=6, width=1.2) # Optional: customize major tick appearance


    # print('====================================================')
    # print('Type of eff_ind_re30:', type(eff_ind_re30))
    # print('Shape of eff_ind_re30:', np.shape(eff_ind_re30))
    # print('Type of time_re30:', type(time_re30))
    # print('Shape of time_re30:', np.shape(time_re30))
    # print('=====================================================')
    ax.plot(time_re93, eff_ind_re93, label='$Re_{\\lambda}=93$', color = '#1f77b4')
    ax.plot(time_re105, eff_ind_re105, label='$Re_{\\lambda}=105$', color = '#ff7f0e')
    ax.plot(time_re145, eff_ind_re145, label='$Re_{\\lambda}=145$', color = '#2ca02c')

    ax.set_title('(b) Decay exponent ($n$)', fontsize=22)
    ax.set_xlabel('$t/T_{eddy,0}$', fontsize=18)
    ax.set_ylabel('$n$', fontsize=18)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.tick_params(axis='both', which='major', labelsize=18)
    # ax.legend(fontsize=18)
    ax.set_xlim(-10,5000)
    ax.axhline(y=10/7, linestyle='-.', color='grey')
    ax.text(0.75, 0.55, '$n = 10/7$', fontsize=20, color='tab:grey', transform=ax.transAxes)
    ax.set_ylim(0.5,2)

    # Create an inset axes at a specific position and size
    # inset_ax = ax.inset_axes([0.40, 0.1, 0.55, 0.55]) # [left, bottom, width, height] as fractions
    # # Plot data on the inset axes, for example:
    # inset_ax.plot(time_re93, eff_ind_re93, label = '$Re_{\\lambda} = 93$', color = '#1f77b4')
    # inset_ax.plot(time_re105, eff_ind_re105, label = '$Re_{\\lambda} = 105$', color = '#ff7f0e')
    # inset_ax.plot(time_re145, eff_ind_re145, label = '$Re_{\\lambda} = 145$', color = '#2ca02c')
    # inset_ax.set_yticks([1, 1.25, 1.43])
    # inset_ax.set_xticks([0, 250, 500])
    # inset_ax.tick_params(axis='both', which='major', labelsize=18)
    # inset_ax.axhline(y=10/7, linestyle='-.', color='grey')
    # inset_ax.set_xlim(-10,500)
    # inset_ax.set_ylim(1.0,1.5)


    plt.tight_layout()
    plt.savefig('figures/Figure_5b_eff_ind_LKB.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax

def intLen(time_list, intLen_list):
    fig = plt.figure(figsize=(6.5,5))
    ax = fig.add_subplot(1,1,1)
    L_box = 2*np.pi
    time_re93 = time_list[0]; intLen_re93 = intLen_list[0]
    time_re105 = time_list[1]; intLen_re105 = intLen_list[1]
    time_re145 = time_list[2]; intLen_re145 = intLen_list[2]

    ax.loglog(time_re93, intLen_re93/L_box, label='$Re_{\\lambda}=93$', color = '#1f77b4')
    ax.loglog(time_re105, intLen_re105/L_box, label='$Re_{\\lambda}=105$', color = '#ff7f0e')
    ax.loglog(time_re145, intLen_re145/L_box, label='$Re_{\\lambda}=145$', color = '#2ca02c')

    cl=0.005
    s=100
    f=10000
    ax.loglog(time_re145[s:f], cl*(time_re145[s:f])**(0.36), 'k-.')
    ax.text(0.55, 0.1, 'Slope = 0.36', fontsize=20, color='k', transform=ax.transAxes)
    # --- MODIFIED TICK PARAMETERS START ---

    # 1. Set major ticks to be visible on all four sides (top, bottom, left, right)
    ax.tick_params(axis='both', which='major', labelsize=18,
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=6, width=1.2) # Optional: customize major tick appearance

    # 2. Add minor ticks and ensure they are also on all four sides
    ax.tick_params(which='minor',
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=3, width=0.75) # Optional: customize minor tick appearance


    ax.legend(fontsize=18)
    ax.set_title('Integral length ($L$)', fontsize=22)  
    ax.set_xlabel('$t/T_{eddy,0}$', fontsize=20)
    ax.set_ylabel('$L/L_{box}$', fontsize=20)
    ax.set_xlim(1e-1,2.5e4)
    ax.set_ylim(5e-3,1e0)
    plt.tight_layout()
    plt.savefig('figures/Figure_6_intLen_LKB.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax
    
def ene_spect(karr_list, ene_spect_list, time_arr, intLen_list):
    time_list = [0, 5.0, 25, 75, 200, 500, 1000, 2000]
    time_list_id = [sc.find_element(time_arr, t)[0] for t in time_list]

    fig = plt.figure(figsize=(8,6))
    ax_spect = fig.add_subplot(1,1,1)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i in range(len(ene_spect_list)):
        ax_spect.loglog(karr_list, ene_spect_list[i], label='t/T= {:.2f}'.format(time_arr[time_list_id[i]]), color=colors[i])
    
    ax_spect.set_xlabel('$k$', fontsize=20)
    ax_spect.set_ylabel(r'$E(k)$', fontsize=22, rotation=0, labelpad=15)
    ax_spect.set_title('(a) Energy Spectrum for LKB spectra ($Re_{\\lambda}=105$)', fontsize=22)
    ax_spect.tick_params(axis='both', which='major', labelsize=16)
    ax_spect.legend(fontsize=14)

    # 1. Set major ticks to be visible on all four sides (top, bottom, left, right)
    ax_spect.tick_params(axis='both', which='major', labelsize=18,
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=6, width=1.2) # Optional: customize major tick appearance

    # 2. Add minor ticks and ensure they are also on all four sides
    ax_spect.tick_params(which='minor',
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=3, width=0.75) # Optional: customize minor tick appearance

    # c1 = 0.05
    # ax_spect.loglog(karr[1:20], c1*(karr[1:20]*intLen[0])**4, '-.', color='black')
    # ax_spect.text(2e0, 1e-3, '$k^4$', fontsize=20)
    c1 = 0.03
    ax_spect.loglog(karr_list[70:1000], c1*(karr_list[70:1000]*intLen_list[0])**(-5/3), '-.', color='black')
    ax_spect.text(3e2, 1e-3, '$k^{-5/3}$', fontsize=20)
    # c2 = 0.002
    # ax_spect.loglog(karr[10:80], c2*np.power(karr[10:80], -1), '-.', color='black')
    # ax_spect.text(2e1, 2e-4, 'Slope = $-1$', fontsize=14)
    ax_spect.set_xlim([1, 2e3])
    ax_spect.set_ylim([1e-15, 1e-1])
    plt.savefig('figures/Figure_8a_ene_spectra_LKB.pdf', dpi=200, bbox_inches = 'tight')
    fig.tight_layout()
    return fig, ax_spect

def ene_spect_norm(karr_list, ene_spect_list, time_arr, intLen_list, nu):

    time_list = [0, 5.0, 25, 75, 200, 500, 1000, 2000]
    time_list_id = [sc.find_element(time_arr, t)[0] for t in time_list]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig = plt.figure(figsize=(8,6))
    ax_spect_norm = fig.add_subplot(1,1,1)

    for i in range(1,len(ene_spect_list)):
        tke, diss = sc.num_int(karr_list, ene_spect_list[i], nu)
        idx = time_list_id[i]
        ax_spect_norm.loglog(karr_list[:]*intLen_list[idx], ene_spect_list[i,:]/(tke*intLen_list[idx]), label='t/T= {:.2f}'.format(time_arr[idx]), color=colors[i])

    ax_spect_norm.set_xlabel('$kL$', fontsize=20)
    ax_spect_norm.set_ylabel(r'$\frac{E(k)}{E_n L}$', fontsize=22, rotation=0, labelpad=20)
    ax_spect_norm.set_title('(b) Normalized Energy Spectrum for LKB spectra ($Re_{\lambda}=105$)', fontsize=20)
    ax_spect_norm.tick_params(axis='both', which='major', labelsize=16)
    # ax_spect_norm.legend(fontsize=14)
    # 1. Set major ticks to be visible on all four sides (top, bottom, left, right)
    ax_spect_norm.tick_params(axis='both', which='major', labelsize=18,
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=6, width=1.2) # Optional: customize major tick appearance

    # 2. Add minor ticks and ensure they are also on all four sides
    ax_spect_norm.tick_params(which='minor',
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=3, width=0.75) # Optional: customize minor tick appearance
    print('=====================================')
    print('Initial intLen: ', intLen_list[0])
    c1 = 5
    ax_spect_norm.loglog(karr_list[0:20]*intLen_list[0], c1*(karr_list[0:20]*intLen_list[0])**4, '-.', color='black')
    ax_spect_norm.text(1e-1, 1e-2, '$k^4$', fontsize=20)
    c2 = 1.2
    ax_spect_norm.loglog(karr_list[0:20], c2*np.power(karr_list[0:20], -1), '-.', color='black')
    ax_spect_norm.text(4, 1, '$k^{-1}$', fontsize=20)
    ax_spect_norm.set_xlim([1e-2, 3e2])
    ax_spect_norm.set_ylim([1e-10, 1e1])
    fig.tight_layout()
    plt.savefig('figures/Figure_8b_ene_spectra_norm_LKB.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax_spect_norm
    
def ene_spect_slope(karr_list, spect_slope_list, time_arr):
    time_list = [0, 5.0, 25, 75, 200, 500, 1000, 2000]
    time_list_id = [sc.find_element(time_arr, t)[0] for t in time_list]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig = plt.figure(figsize=(8,6))
    ax_slope = fig.add_subplot(1,1,1)

    for i in range(len(time_list_id)):
        ax_slope.semilogx(karr_list[i], spect_slope_list[i], label='t/T= {:.2f}'.format(time_arr[time_list_id[i]]))
        
    ax_slope.set_xlabel('$kL$', fontsize=20)
    ax_slope.set_ylabel(r'$\frac{d\ln E(k)}{d\ln k}$', fontsize=22, rotation=0, labelpad=25)
    ax_slope.set_xlim([-6e-2,1e2])
    ax_slope.set_ylim([-4, 5])
    ax_slope.axhline(y=-1, color='black', linestyle='--')
    ax_slope.set_title('(b) Spectral Slope for LKB spectra ($Re_{\\lambda}=105$)', fontsize=20)
    ax_slope.tick_params(axis='both', which='major', labelsize=16)
    ax_slope.legend(fontsize=14)   
    fig.tight_layout()
    plt.savefig('figures/Figure_9b_ene_slope_LKB.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax_slope