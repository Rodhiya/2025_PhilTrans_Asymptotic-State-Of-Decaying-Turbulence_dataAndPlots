import numpy as np
import matplotlib.pyplot as plt
import script as sc
import matplotlib as mpl
# import matplotlib.colors as mcolors
mpl.rcParams['text.usetex'] = True


def totE(time_list, totE_list):
    fig = plt.figure(figsize=(6.5,5))
    ax = fig.add_subplot(1,1,1)

    time_re30 = time_list[0]; totE_re30 = totE_list[0]
    time_re45 = time_list[1]; totE_re45 = totE_list[1]
    time_re70 = time_list[2]; totE_re70 = totE_list[2]
    time_re93 = time_list[3]; totE_re93 = totE_list[3]
    time_re105 = time_list[4]; totE_re105 = totE_list[4]
    time_re145 = time_list[5]; totE_re145 = totE_list[5]
    ax.loglog(time_re30, totE_re30/totE_re30[0], label='$Re_{\\lambda}=30$', color = '#1f77b4')
    ax.loglog(time_re45, totE_re45/totE_re45[0], label='$Re_{\\lambda}=45$', color = '#ff7f0e')
    ax.loglog(time_re70, totE_re70/totE_re70[0], label='$Re_{\\lambda}=70$', color = '#2ca02c')
    ax.loglog(time_re93, totE_re93/totE_re93[0], label='$Re_{\\lambda}=93$', color = '#d62728')
    ax.loglog(time_re105, totE_re105/totE_re105[0], label='$Re_{\\lambda}=105$', color = '#8c564b')
    ax.loglog(time_re145, totE_re145/totE_re145[0], label='$Re_{\\lambda}=145$', color = '#9467bd')

    cl=5
    s=100
    f=10000
    ax.loglog(time_re145[s:f], cl*(time_re145[s:f])**(-1.25), 'k-.')
    ax.text(0.5, 0.7, 'Slope = -5/4' , rotation=0, fontsize=20, color='k', transform=ax.transAxes)

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
    ax.set_title('(a) Kinetic energy $E_n(t)$', fontsize=22)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=18, loc='lower left')
    ax.set_xlim(1e-1,1e4)
    ax.set_ylim(1e-6,1e1)
    ax.set_yticks([1e-6, 1e-4, 1e-2, 1e0])
    # ax.set_yticklabels([1e-6, 1e-4, 1e-2, 1])
    plt.tight_layout()
    plt.savefig('figures/Figure_3a_kinetic_energy_BS.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax

def decay_exp(time_list, decay_exp_list):
    time_re30 = time_list[0]; decay_exp_re30 = decay_exp_list[0]
    time_re45 = time_list[1]; decay_exp_re45 = decay_exp_list[1]
    time_re70 = time_list[2]; decay_exp_re70 = decay_exp_list[2]
    time_re93 = time_list[3]; decay_exp_re93 = decay_exp_list[3]
    time_re105 = time_list[4]; decay_exp_re105 = decay_exp_list[4]
    time_re145 = time_list[5]; decay_exp_re145 = decay_exp_list[5]
    fig = plt.figure(figsize=(6.5,5))
    ax_eff = fig.add_subplot(1,1,1)

    # --- MODIFIED TICK PARAMETERS START ---

    # 1. Set major ticks to be visible on all four sides (top, bottom, left, right)
    ax_eff.tick_params(axis='both', which='major', labelsize=16,
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=6, width=1.2) # Optional: customize major tick appearance


    # print('====================================================')
    # print('Type of eff_ind_re30:', type(eff_ind_re30))
    # print('Shape of eff_ind_re30:', np.shape(eff_ind_re30))
    # print('Type of time_re30:', type(time_re30))
    # print('Shape of time_re30:', np.shape(time_re30))
    # print('=====================================================')
    ax_eff.plot(time_re30, decay_exp_re30, label='$Re_{\\lambda}=30$', color = '#1f77b4')
    ax_eff.plot(time_re45, decay_exp_re45, label='$Re_{\\lambda}=45$', color = '#ff7f0e')
    ax_eff.plot(time_re70, decay_exp_re70, label='$Re_{\\lambda}=70$', color = '#2ca02c')
    ax_eff.plot(time_re93, decay_exp_re93, label='$Re_{\\lambda}=93$', color = '#d62728')
    ax_eff.plot(time_re105, decay_exp_re105, label='$Re_{\\lambda}=105$', color = '#8c564b')
    ax_eff.plot(time_re145, decay_exp_re145, label='$Re_{\\lambda}=145$', color = '#9467bd')

    # Create an inset axes at a specific position and size
    inset_ax = ax_eff.inset_axes([0.41, 0.09, 0.55, 0.55]) # [left, bottom, width, height] as fractions
    # Plot data on the inset axes, for example:
    inset_ax.plot(time_re30, decay_exp_re30, label = '$Re_{\\lambda} = 30$', color = '#1f77b4')
    inset_ax.plot(time_re45, decay_exp_re45, label = '$Re_{\\lambda} = 45$', color = '#ff7f0e')
    inset_ax.plot(time_re70, decay_exp_re70, label = '$Re_{\\lambda} = 70$', color = '#2ca02c')
    inset_ax.plot(time_re93, decay_exp_re93, label = '$Re_{\\lambda} = 93$', color = '#d62728')
    inset_ax.plot(time_re105, decay_exp_re105, label = '$Re_{\\lambda} = 105$', color = '#8c564b')
    inset_ax.plot(time_re145, decay_exp_re145, label = '$Re_{\\lambda} = 145$', color = '#9467bd')
    inset_ax.set_yticks([1, 1.25, 1.5])
    inset_ax.set_xticks([0, 250, 500])
    inset_ax.axhline(y=1.25, linestyle='-.', color='grey')
    inset_ax.tick_params(axis='both', which='major', labelsize=16)
    inset_ax.set_xlim(-10,500)
    inset_ax.set_ylim(1.0,1.5)

    ax_eff.set_xlabel('$t/T_{eddy,0}$', fontsize=20)
    ax_eff.set_ylabel('$n$', fontsize=20)
    ax_eff.set_yticks([0.25, 0.75, 1.25, 1.75, 2.5, 5])
    # ax.legend(fontsize=20)
    ax_eff.set_title('(b) Decay exponent ($n$)', fontsize=22)
    ax_eff.set_xlim(-10,5000)
    ax_eff.axhline(y=1.25, linestyle='-.', color='grey')
    # ax.axhline(y=1.5, linestyle='-.', color='grey')
    # ax.axhline(y=2, linestyle='-.', color='grey')
    # ax.axhline(y=2.5, linestyle='-.', color='grey')
    # ax.set_ylim(0.0,1.8)
    ax_eff.set_ylim(0.0,1.8)
    plt.tight_layout()
    plt.savefig('figures/Figure_3b_decay_exponent_BS.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax_eff

def intLen(time_list, intLen_list):
    fig = plt.figure(figsize=(6.5,5))
    ax = fig.add_subplot(1,1,1)
    L_box = 2*np.pi
    # print('====================================================')
    # print('Type of eff_ind_re30:', type(eff_ind_re30))
    # print('Shape of eff_ind_re30:', np.shape(eff_ind_re30))
    # print('Type of time_re30:', type(time_re30))
    # print('Shape of time_re30:', np.shape(time_re30))
    # print('=====================================================')
    time_re30 = time_list[0]; intLen_re30 = intLen_list[0]
    time_re45 = time_list[1]; intLen_re45 = intLen_list[1]
    time_re70 = time_list[2]; intLen_re70 = intLen_list[2]
    time_re93 = time_list[3]; intLen_re93 = intLen_list[3]
    time_re105 = time_list[4]; intLen_re105 = intLen_list[4]
    time_re145 = time_list[5]; intLen_re145 = intLen_list[5]
    
    ax.loglog(time_re30, intLen_re30/L_box, label='$Re_{\\lambda}=30$', color = '#1f77b4')
    ax.loglog(time_re45, intLen_re45/L_box, label='$Re_{\\lambda}=45$', color = '#ff7f0e')
    ax.loglog(time_re70, intLen_re70/L_box, label='$Re_{\\lambda}=70$', color = '#2ca02c')
    ax.loglog(time_re93, intLen_re93/L_box, label='$Re_{\\lambda}=93$', color = '#d62728')
    ax.loglog(time_re105, intLen_re105/L_box, label='$Re_{\\lambda}=105$', color = '#8c564b')
    ax.loglog(time_re145, intLen_re145/L_box, label='$Re_{\\lambda}=145$', color = '#9467bd')

    cl=0.003
    s=100
    f=10000
    ax.loglog(time_re145[s:f], cl*(time_re145[s:f])**(2/5), 'k--')
    ax.text(0.55, 0.1, 'Slope = 0.40', fontsize=20, color='k', transform=ax.transAxes)

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

    ax.set_title('(a) Integral length ($L$)', fontsize=22)  
    ax.set_xlabel('$t/T_{eddy,0}$', fontsize=20)
    ax.set_ylabel('$L/L_{box}$', fontsize=20)
    ax.legend(fontsize=18)
    ax.set_xlim(1e-1,1e4)
    ax.set_ylim(5e-3,1e0)
    plt.tight_layout()
    plt.savefig('figures/Figure_4a_intLen_BS.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax

def lambda_plot(time_list, lambda_list):
    fig = plt.figure(figsize=(6.5,5))
    ax = fig.add_subplot(1,1,1)
    L_box = 2*np.pi

    time_re30 = time_list[0]; Lambda_re30 = lambda_list[0]
    time_re45 = time_list[1]; Lambda_re45 = lambda_list[1]
    time_re70 = time_list[2]; Lambda_re70 = lambda_list[2]
    time_re93 = time_list[3]; Lambda_re93 = lambda_list[3]
    time_re105 = time_list[4]; Lambda_re105 = lambda_list[4]
    time_re145 = time_list[5]; Lambda_re145 = lambda_list[5]
    # print('====================================================')
    # print('Type of eff_ind_re30:', type(eff_ind_re30))
    # print('Shape of eff_ind_re30:', np.shape(eff_ind_re30))
    # print('Type of time_re30:', type(time_re30))
    # print('Shape of time_re30:', np.shape(time_re30))
    # print('=====================================================')
    ax.loglog(time_re30, Lambda_re30/L_box, label='$Re_{\\lambda}=30$', color = '#1f77b4')
    ax.loglog(time_re45, Lambda_re45/L_box, label='$Re_{\\lambda}=45$', color = '#ff7f0e')
    ax.loglog(time_re70, Lambda_re70/L_box, label='$Re_{\\lambda}=70$', color = '#2ca02c')
    ax.loglog(time_re93, Lambda_re93/L_box, label='$Re_{\\lambda}=93$', color = '#d62728')
    ax.loglog(time_re105, Lambda_re105/L_box, label='$Re_{\\lambda}=105$', color = '#8c564b')
    ax.loglog(time_re145, Lambda_re145/L_box, label='$Re_{\\lambda}=145$', color = '#9467bd')

    cl=0.0006
    s=100
    f=10000
    ax.loglog(time_re145[s:f], cl*(time_re145[s:f])**(1/2), 'k--')
    ax.text(0.55, 0.1, 'Slope = 1/2', fontsize=20, color='k', transform=ax.transAxes)


    # --- MODIFIED TICK PARAMETERS START ---

    # 1. Set major ticks to be visible on all four sides (top, bottom, left, right)
    ax.tick_params(axis='both', which='major', labelsize=14,
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=6, width=1.2) # Optional: customize major tick appearance

    # 2. Add minor ticks and ensure they are also on all four sides
    ax.tick_params(which='minor',
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=3, width=0.75) # Optional: customize minor tick appearance


    ax.set_xlabel('$t/T_{eddy,0}$', fontsize=20)
    ax.set_ylabel('$\\lambda/L_{box}$', fontsize=20)
    ax.set_title('(b) Taylor microscale ($\\lambda$)', fontsize=22)
    ax.set_xlim(1e-1,1e4)
    ax.set_ylim(1e-3,1e0)
    ax.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig('figures/Figure_4b_lambda_BS.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax

def kol(time_list, kol_list):
    fig = plt.figure(figsize=(6.5,5))
    ax = fig.add_subplot(1,1,1)
    L_box = 2*np.pi
    time_re30 = time_list[0]; kol_re30 = kol_list[0]
    time_re45 = time_list[1]; kol_re45 = kol_list[1]
    time_re70 = time_list[2]; kol_re70 = kol_list[2]
    time_re93 = time_list[3]; kol_re93 = kol_list[3]
    time_re105 = time_list[4]; kol_re105 = kol_list[4]
    time_re145 = time_list[5]; kol_re145 = kol_list[5]
    # print('====================================================')
    # print('Type of eff_ind_re30:', type(eff_ind_re30))
    # print('Shape of eff_ind_re30:', np.shape(eff_ind_re30))
    # print('Type of time_re30:', type(time_re30))
    # print('Shape of time_re30:', np.shape(time_re30))
    # print('=====================================================')
    ax.loglog(time_re30, kol_re30/L_box, label='$Re_{\\lambda}=30$', color = '#1f77b4')
    ax.loglog(time_re45, kol_re45/L_box, label='$Re_{\\lambda}=45$', color = '#ff7f0e')
    ax.loglog(time_re70, kol_re70/L_box, label='$Re_{\\lambda}=70$', color = '#2ca02c')
    ax.loglog(time_re93, kol_re93/L_box, label='$Re_{\\lambda}=93$', color = '#d62728')
    ax.loglog(time_re105, kol_re105/L_box, label='$Re_{\\lambda}=105$', color = '#8c564b')
    ax.loglog(time_re145, kol_re145/L_box, label='$Re_{\\lambda}=145$', color = '#9467bd')


    cl=0.00003
    s=100
    f=10000
    ax.loglog(time_re145[s:f], cl*(time_re145[s:f])**(9/16), 'k--')
    ax.text(0.55, 0.25, 'Slope = 9/16', fontsize=20, color='k', transform=ax.transAxes)

    # --- MODIFIED TICK PARAMETERS START ---

    # 1. Set major ticks to be visible on all four sides (top, bottom, left, right)
    ax.tick_params(axis='both', which='major', labelsize=14,
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=6, width=1.2) # Optional: customize major tick appearance

    # 2. Add minor ticks and ensure they are also on all four sides
    ax.tick_params(which='minor',
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=3, width=0.75) # Optional: customize minor tick appearance


    ax.set_xlabel('$t/T_{eddy,0}$', fontsize=20)
    ax.set_ylabel('$\\eta/L_{box}$', fontsize=20)
    ax.set_title('(c) Kolmogorov lengthscale ($\\eta$)', fontsize=22)
    ax.set_xlim(1e-1,1e4)
    ax.set_ylim(1e-5,1e-1)
    # ax.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig('figures/Figure_4c_kol_BS.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax

def reLam(time_list, reLam_list):
    fig = plt.figure(figsize=(6.5,5))
    ax = fig.add_subplot(1,1,1)
    L_box = 2*np.pi
    time_re30 = time_list[0]; reLam_re30 = reLam_list[0]
    time_re45 = time_list[1]; reLam_re45 = reLam_list[1]
    time_re70 = time_list[2]; reLam_re70 = reLam_list[2]
    time_re93 = time_list[3]; reLam_re93 = reLam_list[3]
    time_re105 = time_list[4]; reLam_re105 = reLam_list[4]
    time_re145 = time_list[5]; reLam_re145 = reLam_list[5]

    # print('====================================================')
    # print('Type of eff_ind_re30:', type(eff_ind_re30))
    # print('Shape of eff_ind_re30:', np.shape(eff_ind_re30))
    # print('Type of time_re30:', type(time_re30))
    # print('Shape of time_re30:', np.shape(time_re30))
    # print('=====================================================')
    ax.loglog(time_re30, reLam_re30, label='$Re_{\\lambda}=30$', color = '#1f77b4')
    ax.loglog(time_re45, reLam_re45, label='$Re_{\\lambda}=45$', color = '#ff7f0e')
    ax.loglog(time_re70, reLam_re70, label='$Re_{\\lambda}=70$', color = '#2ca02c')
    ax.loglog(time_re93, reLam_re93, label='$Re_{\\lambda}=93$', color = '#d62728')
    ax.loglog(time_re105, reLam_re105, label='$Re_{\\lambda}=105$', color = '#8c564b')
    ax.loglog(time_re145, reLam_re145, label='$Re_{\\lambda}=145$', color = '#9467bd')
    c=150;s=100;f=20000
    ax.loglog(time_re145[s:f], c*(time_re145[s:f])**(-0.125), 'k--')
    ax.text(0.65, 0.85, 'Slope = -1/8', fontsize=20, color='k', transform=plt.gca().transAxes)
    # --- MODIFIED TICK PARAMETERS START ---

    # 1. Set major ticks to be visible on all four sides (top, bottom, left, right)
    ax.tick_params(axis='both', which='major', labelsize=14,
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=6, width=1.2) # Optional: customize major tick appearance

    # 2. Add minor ticks and ensure they are also on all four sides
    ax.tick_params(which='minor',
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=3, width=0.75) # Optional: customize minor tick appearance


    ax.set_xlabel('$t/T_{eddy,0}$', fontsize=20)
    ax.set_ylabel('$Re_{\\lambda}$', fontsize=20)
    ax.set_title('(d) Taylor Reynolds number ($Re_{\\lambda}$)', fontsize=22)
    ax.set_xlim(1e-1,1e4)
    ax.set_ylim(1e0,2e2)
    # ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/Figure_4d_reLam_BS.pdf', dpi=200, bbox_inches = 'tight')
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
    ax_spect.set_ylabel(r'$E(k)$', fontsize=22, rotation=0, labelpad=20)
    ax_spect.set_title('(a) Energy Spectrum for BS spectra ($Re_{\\lambda}=105$)', fontsize=22)
    ax_spect.tick_params(axis='both', which='major', labelsize=16)
    ax_spect.legend(fontsize=16)
    
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

    c1 = 0.01
    ax_spect.loglog(karr_list[1:20], c1*(karr_list[1:20]*intLen_list[0])**2, '-.', color='black')
    ax_spect.text(2e0, 1e-4, '$k^2$', fontsize=20)
    c1 = 0.03
    ax_spect.loglog(karr_list[70:1000], c1*(karr_list[70:1000]*intLen_list[0])**(-5/3), '-.', color='black')
    ax_spect.text(3e2, 1e-3, '$k^{-5/3}$', fontsize=20)
    c2 = 0.005
    ax_spect.loglog(karr_list[15:80], c2*np.power(karr_list[15:80], -1), '--', color='black')
    ax_spect.text(2.5e1, 2.5e-4, '$k^{-1}$', fontsize=20)
    ax_spect.set_xlim([1e0, 2e3])
    ax_spect.set_ylim([1e-14, 1e0])
    plt.savefig('figures/Figure_7a_ene_spectra_BS.pdf', dpi=200, bbox_inches = 'tight')
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
        ax_spect_norm.loglog(karr_list[1:]*intLen_list[idx], ene_spect_list[i, 1:]/(tke*intLen_list[idx]), label='t/T= {:.2f}'.format(time_arr[idx]), color=colors[i])

    ax_spect_norm.set_xlabel('$kL$', fontsize=20)
    ax_spect_norm.set_ylabel(r'$\frac{E(k)}{E_n L}$', fontsize=22, rotation=0, labelpad=15)
    ax_spect_norm.set_title('(b) Normalized Energy Spectrum for BS spectra ($Re_{\lambda}=105$)', fontsize=20)
    ax_spect_norm.tick_params(axis='both', which='major', labelsize=18)
    ax_spect_norm.legend(fontsize=14)

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

    c1 = 1.2
    ax_spect_norm.loglog(karr_list[1:20]*intLen_list[0], c1*(karr_list[1:20]*intLen_list[0])**2, '-.', color='black')
    ax_spect_norm.text(1e-1, 1e-1, '$k^2$', fontsize=20)
    c2 = 1
    ax_spect_norm.loglog(karr_list[2:20], c2*np.power(karr_list[2:20], -1), '-.', color='black')
    ax_spect_norm.text(4, 0.5, '$k^{-1}$', fontsize=20)   
    ax_spect_norm.set_xlim([1e-2, 3e2])
    ax_spect_norm.set_ylim([1e-8, 1e1])
    fig.tight_layout()
    plt.savefig('figures/Figure_7b_ene_spectra_norm_BS.pdf', dpi=200, bbox_inches = 'tight')
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
    ax_slope.set_ylabel(r'$\frac{d\ln E(k)}{d\ln k}$', fontsize=22, rotation=0, labelpad=20)
    ax_slope.set_xlim([-6e-2,1e2])
    ax_slope.set_ylim([-4, 3])
    ax_slope.axhline(y=-1, color='black', linestyle='--')
    ax_slope.set_title('(a) Spectral Slope for BS spectra ($Re_{\\lambda}=105$)', fontsize=20)
    ax_slope.tick_params(axis='both', which='major', labelsize=16)
    ax_slope.legend(fontsize=14)   
    fig.tight_layout()
    plt.savefig('figures/Figure_9a_ene_slope_BS.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax_slope

def migdal_len(time_list, migdal_len_list):
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8c564b', '#9467bd']
    label_list = ['$Re_{\\lambda}=30$', '$Re_{\\lambda}=45$', '$Re_{\\lambda}=70$', '$Re_{\\lambda}=93$', '$Re_{\\lambda}=105$', '$Re_{\\lambda}=145$']
    fig = plt.figure(figsize=(6.5,5))
    ax = fig.add_subplot(1,1,1)
    L_box = 2*np.pi

    for i in range(len(time_list)):
        ax.plot(time_list[i], migdal_len_list[i]/L_box, color=colors_list[i], label=label_list[i])
    
    cl=0.00012
    s=500
    f=50000

    ax.loglog(time_list[5][s:f], cl*(time_list[5][s:f])**(0.53), color='grey', linestyle='-.')
    ax.text(0.555, 0.375,f"Slope =  {0.53:.2f}", fontsize=16, rotation=32, color='k', transform=ax.transAxes)


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


    ax.set_xlabel('$t/T_{eddy,0}$', fontsize=16)
    ax.set_ylabel('$L_M/L_{box}$', fontsize=16)
    ax.set_title('(a) Migdal length for BS spectra', fontsize=18)
    ax.legend(fontsize=14)

    # ax.legend(fontsize=16,bbox_to_anchor=(1., 1), loc='upper left')


    ax.set_xlim(1e-1,1e4)
    ax.set_ylim(1e-4,1e-1)
    plt.tight_layout()
    plt.savefig('figures/Figure_10a_migdal_len_BS.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax

def migdalLen_parabolaFit(time_re145, migdal_len_re145):
    print("time_re145.shape: ", time_re145.shape)
    print("migdal_len_re145.shape: ", migdal_len_re145.shape)
    migdal_len_re145 = migdal_len_re145[time_re145>=20]; time_re145 = time_re145[time_re145>=20]
    migdal_len_re145 = migdal_len_re145[time_re145<=2000]; time_re145 = time_re145[time_re145<=2000]



    print("time_re145.shape: ", time_re145.shape)
    print("migdal_len_re145.shape: ", migdal_len_re145.shape)

    coefficients = np.polyfit(migdal_len_re145[:], time_re145[:], 2)
    print('coefficients: ', coefficients)


    # 3. Create a polynomial function from the coefficients
    parabola_model = np.poly1d(coefficients)
    time_fit = parabola_model(migdal_len_re145[:])

    fig = plt.figure(figsize=(6.5,5))

    ax = fig.add_subplot(1,1,1)

    ax.plot(migdal_len_re145, time_re145, label='Simulation: $Re_{\lambda}=145$')
    ax.plot(migdal_len_re145, time_fit, '--', label="$\\frac{t}{T_{eddy,0}} = 1.65 - 832.2 L + 237791.8 L^2$")




    ax.tick_params(axis='both', which='major', labelsize=18,
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=6, width=1.2) # Optional: customize major tick appearance

    # 2. Add minor ticks and ensure they are also on all four sides
    ax.tick_params(which='minor',
                direction='in', # Optional: makes ticks point inwards
                top=True, right=True, bottom=True, left=True,
                length=3, width=0.75) # Optional: customize minor tick appearance

    ax.set_ylabel('$t/T_{eddy,0}$', fontsize=18)
    ax.set_xlabel('$L_M(t)$', fontsize=18)
    ax.set_title('(b) Parabolic fit of $L_M(t)$ for BS spectra ($Re_{\lambda}=145$)', fontsize=18)

    ax.set_xlim(0, 0.09)


    ax.set_ylim(20,2000)
    ax.legend(fontsize=16,loc='upper left')

    plt.tight_layout()
    plt.savefig('figures/Figure_10b_migdal_len_parabolaFit_BS.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax

def migdal_lenVSene(time_re145, migdal_len_re145, totE_re145):
    log_E_L = np.genfromtxt('data/theoretical_MIG_logLlogE.csv', delimiter=',')

    totE_ln = np.log(totE_re145)
    migdal_len_ln = np.log(migdal_len_re145)

    time_st = 20
    time_end = 2000
    s, _ = sc.find_element(time_re145, time_st)
    f, _ = sc.find_element(time_re145, time_end)
    # print("Time at s and f: ", time_re145[s], time_re145[f])
    
    result = sc.optimize_migdal(totE_re145[s:f], migdal_len_re145[s:f], log_E_L)
    # print("after optimization: ", result.x)
    # fig = plt.figure(figsize=(8,6))
    # ax_raw = fig.add_subplot(1,1,1)
    # ax_raw.plot(migdal_len_ln[s:f], totE_ln[s:f], label = "Simulation at $Re_{\\lambda} = 145$: $1/L_M^{2.36}$")
    # ax_raw.plot(log_E_L[:,0], log_E_L[:,1], label = "Complete theory including all exponents")
    # ax_raw.set_xlabel('$\ln (L_M)$', fontsize=20)
    # ax_raw.set_ylabel(r'$\ln (E_n)$', fontsize=22, rotation=0, labelpad=40)
    # ax_raw.set_title('Migdal Scaling', fontsize=20)
    # ax_raw.tick_params(axis='both', which='major', labelsize=16)
    # ax_raw.legend(fontsize=14)
    # ax_raw.quiver(migdal_len_ln[2500], totE_ln[2500], result.x[1], -1*result.x[0], angles='xy', scale_units='xy', scale=1, color='k')
    # fig.tight_layout()

    fig = plt.figure(figsize=(8,6))
    ax_shifted = fig.add_subplot(1,1,1)
    ax_shifted.plot(migdal_len_ln[s:f]+result.x[1], totE_ln[s:f]-result.x[0], label = "Simulation at $Re_{\lambda} = 145$: $1/L_M^{2.36}$")
    ax_shifted.plot(log_E_L[:,0], log_E_L[:,1], label = "Complete theory including all exponents")
    ax_shifted.set_xlabel('$\ln (L_M)$', fontsize=20)
    ax_shifted.set_ylabel(r'$\ln (E_n)$', fontsize=22, rotation=0, labelpad=40)
    ax_shifted.set_title('(a) Energy vs Migdal length for BS spectra ($Re_{\\lambda}=145$)', fontsize=20)
    ax_shifted.text(0.05, 0.6, "Migdal's theory", fontsize=18, color='#ff7f0e', transform=ax_shifted.transAxes)
    ax_shifted.text(0.45, 0.6, "Simulation: $Re_{\\lambda}=145$", fontsize=18, color='#1f77b4', transform=ax_shifted.transAxes)
    ax_shifted.tick_params(axis='both', which='major', labelsize=16)
    ax_shifted.set_ylim(totE_ln[f]-result.x[0],totE_ln[s]-result.x[0])
    ax_shifted.set_xlim(migdal_len_ln[s]+result.x[1],migdal_len_ln[f]+result.x[1])  
    # ax_shifted.legend(fontsize=14)
    fig.tight_layout()
    plt.savefig('figures/Figure_11a_migdal_lenVSene_BS.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax_shifted

def zeta2(zeta2_array, time_array, x_arr, f_re145_value_all):
    
    shash_theory_long = np.genfromtxt("data/theoretical_strFun_slope.csv",delimiter=",")
    diff = sc.mean_of_curves(np.log(x_arr),f_re145_value_all,shash_theory_long[:,1]-0.0,shash_theory_long[:,0])
    fig = plt.figure()
    ax_zeta2 = fig.add_subplot(111)

    for i in range(time_array.shape[0]):
        ax_zeta2.plot(np.log(x_arr), zeta2_array[i], '--', label="Time = {:.2f}".format(time_array[i]))
    ax_zeta2.plot(shash_theory_long[shash_theory_long[:,0]>np.min(f_re145_value_all),1]+diff, shash_theory_long[shash_theory_long[:,0]>np.min(f_re145_value_all),0], label="Migdal's theory")
    ax_zeta2.set_xlabel('$\log x$', fontsize=18)
    ax_zeta2.set_ylabel('$\zeta_2(x, t)$', fontsize=18)
    ax_zeta2.set_title('(a) Time evolution of $\zeta_2(x,t)$ for BS spectra ($Re_{\lambda}=145$)', fontsize=18)
    ax_zeta2.tick_params(axis='both', which='major', labelsize=16)

    # ax_zeta2.set_xlim([0,22])
    # ax_zeta2.set_ylim([0,1])
    ax_zeta2.legend(fontsize=16)
    fig.tight_layout()
    plt.savefig('figures/Figure_12a_zeta2_BS.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax_zeta2

def fx(f_re145_value_all, f_re145_err_all, x_arr):
    shash_theory_long = np.genfromtxt("data/theoretical_strFun_slope.csv",delimiter=",")
    diff = sc.mean_of_curves(np.log(x_arr),f_re145_value_all,shash_theory_long[:,1]-0.0,shash_theory_long[:,0])

    fig = plt.figure()
    ax_fx = fig.add_subplot(111)
    ax_fx.plot(np.log(x_arr), f_re145_value_all, color='#ff7f0e', label="Simulation: $Re_{\\lambda} = 145$")
    ax_fx.fill_between(np.log(x_arr), f_re145_value_all-f_re145_err_all, f_re145_value_all+f_re145_err_all, color='#1f77b4', alpha=0.3)
    ax_fx.plot(shash_theory_long[shash_theory_long[:,0]>np.min(f_re145_value_all),1]+diff, shash_theory_long[shash_theory_long[:,0]>np.min(f_re145_value_all),0], label="Migdal's theory")
    ax_fx.text(-2.5, 1.7, "Migdal's theory", fontsize=18, color='#ff7f0e')
    ax_fx.text(0.4, 1.7, "Simulation: $Re_{\\lambda}=145$", fontsize=18, color='#1f77b4')
    ax_fx.set_xlabel('$\log x$', fontsize=20)
    ax_fx.set_ylabel('$f(x)$', fontsize=20)
    ax_fx.set_title('(b) $f(x,t)$ for BS spectra ($Re_{\\lambda}=145$)', fontsize=20)
    ax_fx.tick_params(axis='both', which='major', labelsize=18)

    fig.tight_layout()    
    plt.savefig('figures/Figure_12b_fx_BS.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax_fx

def dissipation_spectra(karr_list, dissipation_spectra, time_list, Re_list):
    fig = plt.figure(figsize=(8,5))
    ax_72norm = fig.add_subplot(1,1,1)

    for i in range(time_list.shape[0]):
        ax_72norm.loglog(karr_list[i], dissipation_spectra[i], label = '$t/T= {:.2f}, Re_{{\lambda}}= {:.1f}$'.format(time_list[i], Re_list[i]))

    ax_72norm.set_title('(a) Compensated Energy Spectrum for BS spectra ($Re_{\lambda}=93$)', fontsize=18)
    ax_72norm.set_xlabel('$k\eta$', fontsize=22)
    ax_72norm.set_ylabel(r'$\frac{E(k)}{\epsilon^{2/3} k^{-5/3}}$', fontsize=22, rotation=0, labelpad=50)
    # plt.xlim(1e0*kol_length_li[0], 1024*kol_length_li[-1])
    plt.ylim(1e-27, 1e2)
    ax_72norm.tick_params(axis='both', which='major', labelsize=15)
    ax_72norm.legend(fontsize=15)
    fig.tight_layout()
    plt.savefig('figures/Figure_14a_dissipation_spectra_BS.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax_72norm
    
def dissipation_spectra_slope(slope_karr_list, slope_list, time_list, k_eta_list):
    fig = plt.figure(figsize=(8,5))
    ax_local_slope = fig.add_subplot(1,1,1)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in range(time_list.shape[0]):
        ax_local_slope.plot(slope_karr_list[i], slope_list[i], label = '$t/T= {:.2f}, k_{{max}}\eta= {:.1f}$'.format(time_list[i], k_eta_list[i]), color=colors[i])

    ax_local_slope.set_title('(b) Spectral Slope for BS spectra ($Re_{\\lambda}=93$)', fontsize=18)
    ax_local_slope.set_xlabel('$k\eta$', fontsize=22)
    ax_local_slope.set_ylabel(r'$\frac{d\ln E(k)}{d\ln k}$', fontsize=22, rotation=0, labelpad=40)
    plt.xlim(0, 12)
    plt.ylim(-60, 0)
    ax_local_slope.tick_params(axis='both', which='major', labelsize=15)
    ax_local_slope.legend(fontsize=14)
    fig.tight_layout()
    plt.savefig('figures/Figure_14b_local_slope_dissipation_spectra_BS.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax_local_slope

def gridMod_eff(time_list, eff_ind_list):
    fig = plt.figure(figsize=(6.5,5))
    ax_eff = fig.add_subplot(1,1,1)

    ax_eff.plot(time_list[0], eff_ind_list[0], color='k', label = 'Without grid modification')
    ax_eff.plot(time_list[1], eff_ind_list[1], '--', color='grey', label = 'Grid modification ($k_{max}\\eta = 1.8$)')
    ax_eff.plot(time_list[2], eff_ind_list[2], '--', color='skyblue', label = 'Grid modification ($k_{max}\\eta = 2.0$)')
    ax_eff.plot(time_list[3], eff_ind_list[3], '--', color='gold', label = 'Grid modification ($k_{max}\\eta = 2.2$)')
    ax_eff.plot(time_list[4], eff_ind_list[4], '--', color='cornflowerblue', label = 'Grid modification ($k_{max}\\eta = 3.0$)')
    ax_eff.plot(time_list[5], eff_ind_list[5], '--', color='lawngreen', label = 'Grid modification ($k_{max}\\eta = 5.0$)')
    ax_eff.plot(time_list[6], eff_ind_list[6], '--', color='crimson', label = 'Grid modification ($k_{max}\\eta = 6.0$)')


    ax_eff.set_title('Decay exponent (n) for $Re_{\\lambda}=93$', fontsize=22)
    ax_eff.set_xlabel('$t/T_{eddy,0}$', fontsize=22)
    ax_eff.set_ylabel('$n$', fontsize=22)
    ax_eff.set_xticks([0, 1000, 2000, 3000])
    ax_eff.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6])
    ax_eff.set_xlim(-100,3.5e3)
    ax_eff.set_ylim(0.75, 1.7)
    ax_eff.tick_params(axis='both', which='major', labelsize=18)
    ax_eff.legend(fontsize=14)
    fig.tight_layout()
    plt.savefig('figures/Figure_2_gridMod_eff_BS.pdf', dpi=200, bbox_inches = 'tight')
    return fig, ax_eff
