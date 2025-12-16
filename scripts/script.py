import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
import linecache

def line_no(fpath):
    f = open(fpath, 'r')
    tot_l = len(f.readlines())
    print("File name: ", fpath)
    print("Total lines: ", tot_l)
    return tot_l

def eulstat_variables(fpath, k=42):
    tot_l = line_no(fpath)
    time_line = np.arange(2,tot_l,k)
    totE_line = np.arange(10,tot_l,k)
    diss_line = np.arange(11,tot_l,k)
    #dissipation = np.array([float(linecache.getline(fpath, li)[26:36]) for li in diss_line])
    #totE = np.array([float(linecache.getline(fpath, li)[26:36]) for li in totE_line])
    #time = np.array([float(linecache.getline(fpath, li)[21:32]) for li in time_line])

    dissipation = []
    for li in diss_line:
        try:
            line = linecache.getline(fpath, li)
            dissipation.append(float(line[26:36]))
        except ValueError:
            print(f"ValueError: Dissipation variable not found on line {li}")
            dissipation.append(np.nan)

    totE = []
    for li in totE_line:
        try:
            line = linecache.getline(fpath, li)
            totE.append(float(line[26:36]))
        except ValueError:
            print(f"ValueError: totE variable not found on line {li}")
            totE.append(np.nan)

    time = []
    for li in time_line:
        try:
            line = linecache.getline(fpath, li)
            time.append(float(line[21:32]))
        except ValueError:
            print(f"ValueError: Time variable not found on line {li}")
            time.append(np.nan)

    return np.array(time), np.array(totE), np.array(dissipation)

def eddyTurnover_time(fpath):
    #print('Checking eddy turn over time')
    #print(linecache.getline(fpath, 14)[26:60])
    print("File name: ", fpath)
    eddytime_array_x = float(linecache.getline(fpath, 14)[26:36])
    eddytime_array_y = float(linecache.getline(fpath, 14)[38:48]) 
    eddytime_array_z = float(linecache.getline(fpath, 14)[50:60])
    eddytime_array = (eddytime_array_x+eddytime_array_y+eddytime_array_z)/3
    
    return eddytime_array

def get_slope(x,y):
    slope = np.zeros_like(x)
    
    slope[1:-1] = (y[2:]-y[:-2])/(x[2:]-x[:-2])
    slope[0] = (y[1]-y[0])/(x[1]-x[0])
    slope[-1] = (y[-1]-y[-2])/(x[-1]-x[-2])

    return slope, x

def initial_spectra(fpath):
    f = open(fpath, 'r')  
    lines = f.readlines()[1:]
    spectra = []
    for l in lines:
        temp = l.split()
        temp = [float(no) for no in temp]
        spectra += temp
    np_spectra = np.array(spectra)
    
    return np_spectra

def spect_eulstat(fpath, k=42, spect_l=13, start=16):
    spect = []
    tot_l = line_no(fpath)
    for i in range(start,tot_l,k):
        ins_spec = []
        for j in range(spect_l):
            n = i+j
            lst = linecache.getline(fpath, n).split()
            lst = [float(l) for l in lst]
            ins_spec = ins_spec + lst 
        spect.append(ins_spec)
    Espectrum = np.array(spect)

    return Espectrum

def spect_eulstat0(fpath, spect_l=13, start=16):
    spect = []
    ins_spec = []
    for j in range(spect_l):
        lst = linecache.getline(fpath, j+start).split()
        #print('lst at', j, lst)
        lst = [float(l) for l in lst]
        ins_spec = ins_spec + lst
    spectrum_0 = np.array(ins_spec)
    return spectrum_0

def dissSpect_eulstat(fpath, k=42, spect_l=13, start=13+16+1):
    spect = []
    tot_l = line_no(fpath)
    for i in range(start,tot_l,k):
        ins_spec = []
        for j in range(spect_l):
            n = i+j
            lst = linecache.getline(fpath, n).split()
            lst = [float(l) for l in lst]
            ins_spec = ins_spec + lst 
        spect.append(ins_spec)
    Espectrum = np.array(spect)

    return Espectrum

def get_integral(fpath, size=42):
    tot_l = line_no(fpath)
    len_line = np.arange(3,tot_l,size)
    length = np.array([float(linecache.getline(fpath, li)[62:72]) for li in len_line])

    return length

def get_longInt(fpath, size=42):
    tot_l = line_no(fpath)
    len_line_u = np.arange(3,tot_l,size)
    len_line_v = np.arange(4,tot_l,size)
    len_line_w = np.arange(5,tot_l,size)

    len_u = np.array([float(linecache.getline(fpath, li)[26:37]) for li in len_line_u])
    len_v = np.array([float(linecache.getline(fpath, li)[38:49]) for li in len_line_v])
    len_w = np.array([float(linecache.getline(fpath, li)[50:72]) for li in len_line_w])

    long_len = np.sqrt(len_u**2 + len_v**2 + len_w**2)

    return long_len

def eulstat_time(fpath, k=42):
    tot_l = line_no(fpath)
    time_line = np.arange(2,tot_l,k)
    time = []
    for li in time_line:
        try:
            line = linecache.getline(fpath, li)
            time.append(float(line[21:32]))
        except ValueError:
            print(f"ValueError: Time variable not found on line {li}")
            time.append(np.nan)

    return np.array(time)

def eulstat_totE(fpath, k=42):
    tot_l = line_no(fpath)
    totE_line = np.arange(10,tot_l,k)
    totE = np.array([float(linecache.getline(fpath, li)[26:36]) for li in totE_line])

    return totE

def eulstat_diss(fpath, k=42):
    tot_l = line_no(fpath)
    diss_line = np.arange(11,tot_l,k)
    dissipation = np.array([float(linecache.getline(fpath, li)[26:36]) for li in diss_line])

    return dissipation

def eulstat_urms(fpath, k=42):
    tot_l = line_no(fpath)
    urms_line = np.arange(9,tot_l,k)
    urms_list_list = [(linecache.getline(fpath, li)[26:60]).split('  ') for li in urms_line]
    urms_arr = np.array([np.array(li, dtype=float) for li in urms_list_list])
    urms = np.linalg.norm(urms_arr, axis=1)/np.sqrt(3)
    #print('urms: ',urms)

    return urms

def eulstat_intL(fpath, k=42):
    tot_l = line_no(fpath)
    length_line = np.arange(3,tot_l,k)
    length = np.array([float(linecache.getline(fpath, li)[26:36]) for li in length_line])

    return length

def eulstat_kol(fpath, size=42):
    tot_l = line_no(fpath)
    kol_line = np.arange(13,tot_l,size)
    kol = np.array([float(linecache.getline(fpath, li)[26:36]) for li in kol_line])

    return kol

def eulstat_un(fpath, size=42): # kolmogorov velocity
    tot_l = line_no(fpath)
    un_line = np.arange(13,tot_l,size)
    un = np.array([float(linecache.getline(fpath, li)[38:48]) for li in un_line])

    return un

def u3_L(u_p, L, eps):
    u3_l = np.power(u_p,3)/L
    u3_l_norm = eps/u3_l

    return u3_l, u3_l_norm

def get_L11(k_arr, ek):
    tot_k = 0
    L11 = 0
    for i,k_i in enumerate(k_arr[1:]):
        tot_k += ek[i]
        L11 += ek[i]/k_i

    L11 = L11*3*np.pi/(4*tot_k)
    return L11

def num_int(k, ek, nu):
    tot_k = 0
    epsilon = 0
    
    tot_k = np.sum(ek)
    epsilon = 2*nu*np.sum(k*k*ek)

    #for i,k_i in enumerate(k):
    #    tot_k += ek[i]
    #    epsilon += k_i*k_i*ek[i]

    #epsilon = 2*nu*epsilon
    return tot_k, epsilon

def get_eta(nu, epsilon):
    eta = (nu**3/epsilon)**0.25
    return eta

def eulstat_Re(fpath, size=42):
    tot_l = line_no(fpath)
    Re_line = np.arange(8,tot_l,size)
    Re = np.array([float(linecache.getline(fpath, li)[62:72]) for li in Re_line])

    return Re

def eulstat_lam(fpath, size=42):
    tot_l = line_no(fpath)
    lam_line = np.arange(6,tot_l,size)
    lam = np.array([float(linecache.getline(fpath, li)[26:36]) for li in lam_line])

    return lam

def eulstate_uvw_rms(fpath, size=42):
    tot_l = line_no(fpath)
    rms_line = np.arange(9,tot_l,size)
    urms = np.array([float(linecache.getline(fpath, li)[26:36]) for li in rms_line])
    vrms = np.array([float(linecache.getline(fpath, li)[37:48]) for li in rms_line])
    wrms = np.array([float(linecache.getline(fpath, li)[49:60]) for li in rms_line])
    n = rms_line.shape[0]
    rms_array = np.zeros([n,3])
    rms_array[:,0] = urms
    rms_array[:,1] = vrms
    rms_array[:,2] = wrms

    return rms_array

# U_rms
def get_urms(ek):
    k = np.sum(ek)
    urms = 2*np.sqrt(k)/3

    return urms

#eddy turnover time
def get_eddyT(karr, ek):
    l11 = get_L11(karr, ek)
    u_rms = get_urms(ek)
    t_eddy = l11/u_rms

    return t_eddy

# taylor microscale
def get_taylormicroscale(karr, ek, nu):
    urms = get_urms(ek)
    eps = num_int(karr, ek, nu)[1]
    lambda_micro = np.sqrt(15*nu*np.power(urms, 2)/eps)

    return lambda_micro

def get_Re(karr, ek, nu):
    k, eps = num_int(karr, ek, nu)
    Re_L = k**2/(eps*nu)
    Re_lam = (20*Re_L/3)**0.5

    return Re_L, Re_lam

def slope_leastSQ(x,y):
    x = np.log(x)
    y = np.log(y)
    A = np.vstack([x, np.ones(len(x))]).T
    sol, residual  = np.linalg.lstsq(A, y)[0:2]
    #    print('slope is: ', sol[0])
    #    print('residual is: ', residual[0])
    return sol[0], residual[0]

def eulstat_eddytime(fpath, k):
    tot_l = line_no(fpath)
    eddy_line = np.arange(14,tot_l,k)
    eddytime_array_x = np.array([float(linecache.getline(fpath, li)[26:36]) for li in eddy_line])
    eddytime_array_y = np.array([float(linecache.getline(fpath, li)[38:48]) for li in eddy_line])
    eddytime_array_z = np.array([float(linecache.getline(fpath, li)[50:60]) for li in eddy_line])
    eddytime_array = (eddytime_array_x+eddytime_array_y+eddytime_array_z)/3
    
    return eddytime_array

def diss_from_spectra(ek, k, nu):
    n = ek.shape[0]
    diss_arr = np.zeros(n)
    diss_arr = 2*nu*np.sum(ek*np.power(k,2), axis=1)

    return diss_arr

def u_x_corr_eulstat(fpath, k=122, corr_l=13, start=4):
    u_x_corr_lst = []
    tot_l = line_no(fpath)
    for i in range(start, tot_l, k):
        ins_corr_lst = []
        for j in range(corr_l):
            n = i+j
            lst = linecache.getline(fpath, n).split()
            lst = [float(l) for l in lst]
            ins_corr_lst = ins_corr_lst + lst
        u_x_corr_lst.append(ins_corr_lst[1:])
    u_x_corr = np.array(u_x_corr_lst)
    return u_x_corr

def u_y_corr_eulstat(fpath, k=122, corr_l=13, start=17):
    u_y_corr_lst = []
    tot_l = line_no(fpath)
    for i in range(start, tot_l, k):
        ins_corr_lst = []
        for j in range(corr_l):
            n = i+j
            lst = linecache.getline(fpath, n).split()
            lst = [float(l) for l in lst]
            ins_corr_lst = ins_corr_lst + lst
        u_y_corr_lst.append(ins_corr_lst[1:])
    u_y_corr = np.array(u_y_corr_lst)
    return u_y_corr

def u_z_corr_eulstat(fpath, k=122, corr_l=13, start=30):
    u_z_corr_lst = []
    tot_l = line_no(fpath)
    for i in range(start, tot_l, k):
        ins_corr_lst = []
        for j in range(corr_l):
            n = i+j
            lst = linecache.getline(fpath, n).split()
            lst = [float(l) for l in lst]
            ins_corr_lst = ins_corr_lst + lst
        u_z_corr_lst.append(ins_corr_lst[1:])
    u_z_corr = np.array(u_z_corr_lst)
    return u_z_corr

def v_x_corr_eulstat(fpath, k=122, corr_l=13, start=44):
    v_x_corr_lst = []
    tot_l = line_no(fpath)
    for i in range(start, tot_l, k):
        ins_corr_lst = []
        for j in range(corr_l):
            n = i+j
            lst = linecache.getline(fpath, n).split()
            lst = [float(l) for l in lst]
            ins_corr_lst = ins_corr_lst + lst
        v_x_corr_lst.append(ins_corr_lst[1:])
    v_x_corr = np.array(v_x_corr_lst)
    return v_x_corr

def v_y_corr_eulstat(fpath, k=122, corr_l=13, start=57):
    v_y_corr_lst = []
    tot_l = line_no(fpath)
    for i in range(start, tot_l, k):
        ins_corr_lst = []
        for j in range(corr_l):
            n = i+j
            lst = linecache.getline(fpath, n).split()
            lst = [float(l) for l in lst]
            ins_corr_lst = ins_corr_lst + lst
        v_y_corr_lst.append(ins_corr_lst[1:])
    v_y_corr = np.array(v_y_corr_lst)
    return v_y_corr

def v_z_corr_eulstat(fpath, k=122, corr_l=13, start=70):
    v_z_corr_lst = []
    tot_l = line_no(fpath)
    for i in range(start, tot_l, k):
        ins_corr_lst = []
        for j in range(corr_l):
            n = i+j
            lst = linecache.getline(fpath, n).split()
            lst = [float(l) for l in lst]
            ins_corr_lst = ins_corr_lst + lst
        v_z_corr_lst.append(ins_corr_lst[1:])
    v_z_corr = np.array(v_z_corr_lst)
    return v_z_corr

def w_x_corr_eulstat(fpath, k=122, corr_l=13, start=84):
    w_x_corr_lst = []
    tot_l = line_no(fpath)
    for i in range(start, tot_l, k):
        ins_corr_lst = []
        for j in range(corr_l):
            n = i+j
            lst = linecache.getline(fpath, n).split()
            lst = [float(l) for l in lst]
            ins_corr_lst = ins_corr_lst + lst
        w_x_corr_lst.append(ins_corr_lst[1:])
    w_x_corr = np.array(w_x_corr_lst)
    return w_x_corr

def w_y_corr_eulstat(fpath, k=122, corr_l=13, start=97):
    w_y_corr_lst = []
    tot_l = line_no(fpath)
    for i in range(start, tot_l, k):
        ins_corr_lst = []
        for j in range(corr_l):
            n = i+j
            lst = linecache.getline(fpath, n).split()
            lst = [float(l) for l in lst]
            ins_corr_lst = ins_corr_lst + lst
        w_y_corr_lst.append(ins_corr_lst[1:])
    w_y_corr = np.array(w_y_corr_lst)
    return w_y_corr

def w_z_corr_eulstat(fpath, k=122, corr_l=13, start=110):
    w_z_corr_lst = []
    tot_l = line_no(fpath)
    for i in range(start, tot_l, k):
        ins_corr_lst = []
        for j in range(corr_l):
            n = i+j
            lst = linecache.getline(fpath, n).split()
            lst = [float(l) for l in lst]
            ins_corr_lst = ins_corr_lst + lst
        w_z_corr_lst.append(ins_corr_lst[1:])
    w_z_corr = np.array(w_z_corr_lst)
    return w_z_corr

def read_avg_strfun(fpath, nx, norder):
    spect_l = int((nx/2)/6) +1
    strfun = np.zeros((norder-1,nx//2))

    for i in range(norder-1):
        ins_str = []
        for j in range(spect_l):
            lst = linecache.getline(fpath, i*(spect_l+1)+j+3).split()
            lst = [float(l) for l in lst]
            ins_str += lst
        #print('=========================================================')
        #print('i is: ', i)
        #print('ins_str: ', ins_str)
        strfun[i,:] = np.array(ins_str)

    return strfun
    
def velocity_signal(fpath, num_valus=256):
    signal = np.memmap(fpath, dtype=np.float64 , mode='r+', shape=(num_valus), order='F')
    return signal

def find_xcomm(num_points, max_val = 200):
    overall_max = max_val
    overall_min = 0
    x_comm = np.linspace(overall_min, overall_max, num_points)

    return x_comm

def y_interp_avg(x_list, y_list, x_comm):
    
    n = len(x_list)
    N = np.shape(x_comm)[0]
    y_interp = np.zeros((n,N))
    
    print('Lengths of both list: ', len(x_list), len(y_list))
    for i in range(n):
        temp_y_intrep = np.interp(x_comm, x_list[i], y_list[i])
        y_interp[i,:] += temp_y_intrep

    
    y_avg = np.sum(y_interp, axis=0)/n
    y_std = np.std(y_interp, axis=0, ddof=1)

    return y_avg, y_std

def find_element(array, value):
    idx = np.abs(array-value).argmin()
    return idx, array[idx]

def plotting_energy_spect(karr, spectra, time_id_li, time_val_li, kol_length_li, kol_vel_li, kmax):
    fig = plt.figure(figsize=(8,5))
    ax_ene = fig.add_subplot(1,1,1)

    for i in range(len(time_id_li)):
        x_norm = 1/kol_length_li[i]
        y_norm = kol_length_li[i] * np.power(kol_vel_li[i],2)
        k_eta = kmax*kol_length_li[i]
        ax_ene.loglog(karr[1:kmax]/x_norm, spectra[time_id_li[i],1:kmax]/y_norm, label = '$t/T= {:.2f}, k_{{max}}\eta= {:.1f}$'.format(time_val_li[i], k_eta))

    
    ax_ene.set_title('Energy Spectrum ($Re_{\lambda,t=0}=93$)', fontsize=18)
    ax_ene.set_xlabel('$k\eta$', fontsize=22)
    ax_ene.set_ylabel(r'$\frac{E(k)}{\eta u_{\eta}^2}$', fontsize=22, rotation=0, labelpad=20)
    # plt.xlim(1e0*kol_length_li[0], 1024*kol_length_li[-1])
    plt.ylim(1e-31, 1e4)
    ax_ene.tick_params(axis='both', which='major', labelsize=15)
    ax_ene.legend(fontsize=15)
    fig.tight_layout()
    fig.savefig('figures/energy_spect.pdf')
    return fig, ax_ene

def plotting_energy_spect_norm(karr, spectra, time_id_li, time_val_li, kol_length_li, kol_vel_li, kmax):
    fig = plt.figure(figsize=(8,5))
    ax_72norm = fig.add_subplot(1,1,1)

    for i in range(len(time_id_li)):
        x_norm = 1/kol_length_li[i]
        y_norm = kol_length_li[i] * np.power(kol_vel_li[i],2) / np.power(karr[1:kmax]/x_norm,7/2)
        k_eta = kmax*kol_length_li[i]
        ax_72norm.loglog(karr[1:kmax]/x_norm, spectra[time_id_li[i],1:kmax]/y_norm, label = '$t/T= {:.2f}, k_{{max}}\eta= {:.1f}$'.format(time_val_li[i], k_eta))

    
    ax_72norm.set_title('Energy Spectrum ($Re_{\lambda,t=0}=93$)', fontsize=18)
    ax_72norm.set_xlabel('$k\eta$', fontsize=22)
    ax_72norm.set_ylabel(r'$\frac{E(k)}{\eta u_{\eta}^2 (k\eta)^{-7/2}}$', fontsize=22, rotation=0, labelpad=50)
    # plt.xlim(1e0*kol_length_li[0], 1024*kol_length_li[-1])
    plt.ylim(1e-27, 1e2)
    ax_72norm.tick_params(axis='both', which='major', labelsize=15)
    ax_72norm.legend(fontsize=15)
    fig.tight_layout()
    fig.savefig('figures/energy_spect_norm.pdf')
    return fig, ax_72norm

def plotting_energy_spect_norm2(karr, spectra, time_id_li, time_val_li, kol_length_li, kol_vel_li, kmax, Relam_li):
    fig = plt.figure(figsize=(8,5))
    ax_72norm = fig.add_subplot(1,1,1)
    nu = 0.0001
    
    for i in range(len(time_id_li)):
        x_norm = 1/kol_length_li[i]
        
        k_eta = kmax*kol_length_li[i]
        k, eps = num_int(karr[1:kmax], spectra[time_id_li[i],1:kmax], nu)
        y_norm = np.power(eps,2/3) * np.power(karr[1:kmax],-5/3)
        ax_72norm.loglog(karr[1:kmax]/x_norm, spectra[time_id_li[i],1:kmax]/y_norm, label = '$t/T= {:.2f}, Re_{{\lambda}}= {:.1f}$'.format(time_val_li[i], Relam_li[i]))

    
    ax_72norm.set_title('(a) Compensated Energy Spectrum for BS spectra ($Re_{\lambda}=93$)', fontsize=18)
    ax_72norm.set_xlabel('$k\eta$', fontsize=22)
    ax_72norm.set_ylabel(r'$\frac{E(k)}{\epsilon^{2/3} k^{-5/3}}$', fontsize=22, rotation=0, labelpad=50)
    # plt.xlim(1e0*kol_length_li[0], 1024*kol_length_li[-1])
    plt.ylim(1e-27, 1e2)
    ax_72norm.tick_params(axis='both', which='major', labelsize=15)
    ax_72norm.legend(fontsize=15)
    fig.tight_layout()
    fig.savefig('figures/energy_spect_norm2.pdf')
    return fig, ax_72norm

def local_slope(x,y):
    slope = np.zeros_like(x)
    
    slope[1:-1] = (y[2:]-y[:-2])/(x[2:]-x[:-2])
    slope[0] = (y[1]-y[0])/(x[1]-x[0])
    slope[-1] = (y[-1]-y[-2])/(x[-1]-x[-2])

    return slope

def local_log_slope(karr, spectra):
    k_log = np.log(karr)
    e_log = np.log(spectra)
    # print("k_log: \n")
    # print(k_log)
    # print("e_log: \n")
    # print(e_log)
    # sl_n = np.shape(k_log)[0]
    # k_new = np.linspace(k_log[0], k_log[-1], sl_n)
    # e_new = np.interp(k_new, k_log[:], e_log[:])
    # slope = local_slope(k_new, e_new)
    slope = local_slope(k_log, e_log)

    return slope, karr
    # return slope, np.exp(k_new)

def plotting_local_slope(karr, spectra, time_id_li, time_val_li, kol_length_li, kol_vel_li, k_eta_st, k_eta_end, kmax):
    fig = plt.figure(figsize=(8,5))
    ax_local_slope = fig.add_subplot(1,1,1)
    # ref_line = np.linspace(0.1, 10, 100)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in range(len(time_id_li)):
        x_norm = 1/kol_length_li[i]
        y_norm = 1
        slope, k_new = local_log_slope(karr[1:kmax], spectra[time_id_li[i],1:kmax])
        k_eta = kmax*kol_length_li[i]
        k_start_idx = find_element(k_new[:]/x_norm, k_eta_st)[0]
        k_end_idx = find_element(k_new[:]/x_norm, k_eta_end[i])[0]
        slope_smoothed = savgol_filter(slope, window_length=51, polyorder=3)
        ax_local_slope.plot(k_new[k_start_idx:k_end_idx]/x_norm, slope_smoothed[k_start_idx:k_end_idx], label = '$t/T= {:.2f}, k_{{max}}\eta= {:.1f}$'.format(time_val_li[i], k_eta), color=colors[i])
        # ax_local_slope.loglog(ref_line, 10*ref_line**(2/3), "--", color='black', linewidth=0.5)
        # ax_local_slope.plot(k_new[k_start_idx:k_end_idx]/x_norm, slope[k_start_idx:k_end_idx], linewidth=0.5, color=colors[i])

    
    ax_local_slope.set_title('(b) Spectral Slope for BS spectra ($Re_{\\lambda}=93$)', fontsize=18)
    ax_local_slope.set_xlabel('$k\eta$', fontsize=22)
    ax_local_slope.set_ylabel(r'$\frac{d\ln E(k)}{d\ln k}$', fontsize=22, rotation=0, labelpad=40)
    plt.xlim(0, 12)
    plt.ylim(-60, 0)
    ax_local_slope.tick_params(axis='both', which='major', labelsize=15)
    ax_local_slope.legend(fontsize=14)
    fig.tight_layout()
    fig.savefig('figures/local_slope_BS.pdf')
    return fig, ax_local_slope

def moving_avg(y, ord):
    n = y.shape[0]
    if ord ==3:
        # y_mod = np.zeros(n)
        y_mod = y.copy()
        y_n= np.roll(y, 1)
        y_p = np.roll(y, -1) 
        y_mod[1:-1] = (y_n[1:-1] + y[1:-1] + y_p[1:-1])/3
    if ord ==5:
        y_mod = np.zeros(n)
        y_mod = y.copy()
        y_n1 = np.roll(y, 2); y_n2= np.roll(y, 1)
        y_p1 = np.roll(y, -1); y_p2 = np.roll(y, -2); 
        y_mod[2:-2] = (y_n2[2:-2] + y_n1[2:-2] + y[2:-2] + y_p1[2:-2]+ y_p2[2:-2])/5
    if ord > 5:
        y_arr = np.array(y, dtype=float)
        window_size = 2 * ord + 1
        kernel = np.ones(window_size)
        sums = np.convolve(y_arr, kernel, mode='same')
        counts = np.convolve(np.ones_like(y_arr), kernel, mode='same')
        y_mod = sums / counts
        
    return y_mod

def shasha_length(ek, k):
    num = sp.integrate.simpson(y= ek*k, x=k)
    den = sp.integrate.simpson(y=ek*k**2, x=k)
    len = num/den
    return len

def compute_zeta2(ek, k, r):
    cos_arr = np.cos(k*r)
    sin_arr = np.sin(k*r)
    num_int = (-cos_arr + sin_arr/(k*r))*ek
    num = sp.integrate.simpson(y=num_int, x=k)
    den_int = (1 - sin_arr/(k*r))*ek
    den = sp.integrate.simpson(y=den_int, x=k)
    # print('den: ', den)
    zeta_2 = num/den
    return zeta_2

def second_moment_f(ek, k, x_arr):
    M_len = shasha_length(ek[1:], k[1:])
    # print("M_len: ", M_len)
    zeta_2r = np.zeros_like(x_arr)
    for i, x_i in enumerate(x_arr):
        zeta_2r[i] = compute_zeta2(ek[1:],k[1:],x_i*M_len)
        # print('zeta_2r', zeta_2r[i])

    return zeta_2r

def get_zeta2(karr, spectrum_re145, time_re145, time_st, time_end, xarr):
    zeta_2 = []
    time = []
    print("Getting zeta2")
    for i in range(len(time_re145)):
        print('Realization: ', i)
        start_idx, _ = find_element(time_re145[i], time_st)
        end_idx, _ = find_element(time_re145[i], time_end)
        temp_time = time_re145[i][start_idx:end_idx]
        time.append(temp_time)
        x_len = xarr.shape[0]
        zeta_i = np.zeros((end_idx-start_idx, x_len))
        for j in range(start_idx, end_idx):
            zeta_i[j-start_idx,:] = second_moment_f(spectrum_re145[i][j,:], karr, xarr)
        zeta_2.append(zeta_i)

    return zeta_2, time

def mean_of_curves(x1,y1,x2,y2):
    y2 = y2[::-1]
    x2 = x2[::-1]
    y1 = y1[::-1]
    x1 = x1[::-1]
    min_y = max(np.min(y1), np.min(y2))
    x1 = x1[y1>min_y]
    y1 = y1[y1>min_y]
    x2 = x2[y2>min_y]
    y2 = y2[y2>min_y]
    yrange_1 = (np.max(y1)-np.min(y1))
    yrange_2 = (np.max(y2)-np.min(y2))
    print('yranges are: ', yrange_1, yrange_2)

    int_1 = sp.integrate.simpson(y=x1, x=y1)/yrange_1
    int_2 = sp.integrate.simpson(y=x2, x=y2)/yrange_2
    print('int_1: ', int_1)
    print('int_2: ', int_2)
    difference = int_1 - int_2
    print('----------------------------------')
    print('Difference is: ', difference)

    return difference

def gaussian_filter(x, y, std):
    x_uniform = np.linspace(x.min(), x.max(), 5000)
    y_uniform = np.interp(x_uniform, x, y)
    y_filtered = sp.ndimage.gaussian_filter1d(y_uniform, std)
    return x_uniform, y_filtered

def interpolated_savgol_filter(x, y, window_length, polyorder):
    x_uniform = np.linspace(x.min(), x.max(), 5000)
    y_uniform = np.interp(x_uniform, x, y)
    y_filtered = sp.signal.savgol_filter(y_uniform, window_length, polyorder)
    return x_uniform, y_filtered

def splines_filter(x, y, smooth_factor):
    spline = UnivariateSpline(x, y, k=2, s=smooth_factor)
    x_dense = np.linspace(x.min(), x.max(), 5000)
    slope_smooth = spline(x_dense)
    return x_dense, slope_smooth

def spectral_slope_old(karr, spectrum, ord):
    k_l = np.log(karr)
    e_l = np.log(spectrum)
    slope, _ = get_slope(k_l, e_l)
    # print("slope.type: ", type(slope))
    slope_mod = moving_avg(slope[:], ord)
    # k_l_new, slope_mod = gaussian_filter(k_l, slope, 70)
    # k_l_new, slope_mod = savgol_filter(k_l, slope, 40, 3)
    # k_l_new, slope_mod = splines_filter(k_l, slope, 2)
    return karr, slope_mod

def spectral_slope_new(karr, spectrum, s_factor):
    valid = spectrum > 0
    x = np.log(karr[valid])
    y = np.log(spectrum[valid])
    
    spline = UnivariateSpline(x, y, k=4, s=s_factor)
    x_part1 = np.linspace(x[0], x[100], 1000)
    x_dense = np.concatenate((x_part1, x[101:]))
    k_dense = np.exp(x_dense)
    log_e_dense = spline(x_dense)
    slope_dense = spline.derivative(n=1)(x_dense)
    # print("k_dense.shape: ", k_dense.shape)
    # print("log_e_dense.shape: ", log_e_dense.shape)
    # slope_dense, _ = get_slope(x_dense, log_e_dense)

    k_dense = k_dense[:]
    e_dense = np.exp(log_e_dense)
    return k_dense, e_dense, slope_dense


def spectral_slope_new_2(karr, spectrum, s_factor):
    
    valid = spectrum > 0
    x = np.log(karr[valid])
    y = np.log(spectrum[valid])
    f_interp = interp1d(x, y, kind='linear') 
    num_points = 5000
    x_dense = np.linspace(x.min(), x.max(), num_points)
    y_dense = f_interp(x_dense)

    # delta = x_dense[1] - x_dense[0]  # The uniform spacing
    # window_frac = 0.05
    # # window_length = int(num_points * window_frac)
    # # if window_length % 2 == 0:   # Ensure odd
    # #     window_length += 1

    log_E_smooth = gaussian_filter1d(y_dense, 70, order=0)
    slope_dense, _ = get_slope(x_dense, log_E_smooth)

    # window_length = 5
    # E_dense = savgol_filter(y_dense, window_length, polyorder=3, deriv=0)
    # slope_dense, _ = get_slope(x_dense, E_dense)

    return np.exp(x_dense), np.exp(log_E_smooth), slope_dense