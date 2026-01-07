#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 08:11:49 2025

@author: sofia
"""

#%%

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FuncFormatter
import numpy as np
from matplotlib import rcParams
from matplotlib import rc
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import QwaveMPS.src as qmps


#Parameters for plots style

def picStyle(fontsize):
    rc('text', usetex=True)
    rc('font',**{'family':'CMU Serif'}) #it is not changing the font
    rc('font',size=fontsize)
    rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    rcParams["legend.borderaxespad"] = 0.2
    rcParams["legend.fancybox"] = True
    rcParams["legend.frameon"] = True
    rcParams['figure.constrained_layout.use'] = False

def pic_style(fontsize):
    rc('font',size=fontsize)

def clean_ticks(x, pos):
    # Only show decimals if not an integer
    return f'{x:g}'

#%%


"""Choose the simulation parameters"""

"Choose the time step and end time"

delta_t = 0.02
tmax = 40#70
tlist=np.arange(0,tmax,delta_t)
d_t_l=2 #Time right channel bin dimension
d_t_r=2 #Time left channel bin dimension
d_t_total=np.array([d_t_l,d_t_r])

d_sys1=2 # tls bin dimension
d_sys_total=np.array([d_sys1]) #total system bin (in this case only 1 tls)

"Choose max bond dimension"

# bond = d_t_total^(number of excitations)
bond=18


""" Choose the initial state and coupling"""
gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1)
input_params = qmps.parameters.InputParams(
    delta_t=delta_t,
    tmax = tmax,
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=gamma_l,
    gamma_r = gamma_r,
    bond=bond,
)


i_s0=qmps.states.i_sg()

i_n0 = qmps.states.vacuum(delta_t, input_params)



cw_pump=2*np.pi
max_tmax = 500
pulse_pump = qmps.states.gaussian_envelope(max_tmax, qmps.parameters.InputParams(delta_t=delta_t, tmax=None, d_sys_total=None, d_t_total=None, bond=None, gamma_l=None, gamma_r=None), 0.5, 1.5)
pulse_pump = np.pi * qmps.states.normalize_pulse_envelope(delta_t, pulse_pump)

pump = cw_pump
#pump = pulse_pump

#%% CW Markovian

"""Choose the Hamiltonian"""
Hm=qmps.hamiltonian_1tls(input_params,pump)


"""Calculate time evolution of the system"""
bins = qmps.t_evol_mar(Hm,i_s0,i_n0,input_params)


"""Calculate population dynamics"""

pop = qmps.pop_dynamics(bins, input_params)

ss_correl = qmps.steady_state_correlations(bins, pop, input_params)
spect, w_list = qmps.spectrum_w(delta_t, ss_correl.g1_listr)

#%% CW No Markov

tau=1
phase=0#np.pi/2
#tmax = 10

bond=32

d_t=2 #time bin dimension of one channel
d_t_total=np.array([d_t]) #single channel for mirror case
input_params.tau = tau
input_params.bond = bond
input_params.phase = phase
input_params.d_t_total = d_t_total
input_params.tmax = tmax
i_n0 = qmps.states.vacuum(delta_t, input_params)


"""Choose the Hamiltonian"""
Hm=qmps.hamiltonian_1tls_feedback(input_params, pump)


""" Time evolution of the system"""

bins_nm = qmps.t_evol_nmar(Hm,i_s0,i_n0,input_params)


""" Calculate population dynamics"""

pops_nm=qmps.pop_dynamics_1tls_nmar(bins_nm, input_params)

ss_correls_nm=qmps.steady_state_correlations(bins_nm, pops_nm, input_params)

spect_nm,w_list_nm=qmps.spectrum_w(delta_t, ss_correls_nm.g1_list)

#%%

phase=np.pi

delta_t = 0.02
tmax2 = 200
tlist2=np.arange(0,tmax2,delta_t)

input_params.phase = phase
input_params.delta_t = delta_t
input_params.tmax = tmax2

"""Choose the Hamiltonian"""

Hm=qmps.hamiltonian_1tls_feedback(input_params,pump)


""" Time evolution of the system"""

bins2 = qmps.t_evol_nmar(Hm,i_s0,i_n0,input_params)


""" Calculate population dynamics"""

pops_nm2=qmps.pop_dynamics_1tls_nmar(bins2, input_params)

ss_correls_nm2=qmps.steady_state_correlations(bins2, pops_nm2, input_params)

spect_nm2,w_list_nm2=qmps.spectrum_w(delta_t,ss_correls_nm2.g1_list)


#%%

plt.figure()
plt.plot(tlist2,pops_nm2.pop)

#%%

fonts=24

picStyle(fonts)


fig, ax = plt.subplots(1,4,figsize=(18,4))#,sharey=True,sharex=True)
# plt.subplots_adjust(wspace=0.3,hspace=0.3)
ax[0].plot(tlist,np.real(pop.pop),linewidth = 4, color = 'k',linestyle='-',label=r'$n_{\rm TLS}^{\rm M}$') # TLS population
ax[0].plot(tlist,np.real(pops_nm.pop),linewidth = 3, color = 'grey',linestyle='-',label=r'$n_{\rm TLS}^{\rm NM}$') # TLS population
ax[0].legend(loc='upper right',frameon=False,handlelength=1.0)
ax[0].grid(True, linestyle='--', alpha=0.6)
ax[0].set_ylim([0.,1.05])
ax[0].set_xlim([0.,8])
formatter = FuncFormatter(clean_ticks)
ax[0].xaxis.set_major_formatter(formatter)
ax[0].yaxis.set_major_formatter(formatter)
# ax[0].set_xticklabels([])
ax[0].set_xlabel('$\gamma t$',fontsize=fonts)
ax[0].text(.5, 0.9, '(a)',fontsize=fonts)

# ax[1].axvspan(0.0, tau, color='gray', alpha=0.2)
ax[1].plot(ss_correl.t_cor,np.real(ss_correl.g1_listr),linewidth = 4, color = 'darkgreen',linestyle='-',label=r'$g^{(1)}_{\rm M}$') # TLS population
# ax[0,1].plot(tlist[:-1],np.real(pulsed_pump),linewidth = 3, color = 'silver',linestyle='--',label=r'$n_{\rm TLS}^{\rm an}$') # TLS population
ax[1].plot(ss_correls_nm.t_cor,np.real(ss_correls_nm.g1_list_nm),linewidth = 3,color = 'limegreen',linestyle='-',label=r'$g^{(1)}_{\rm NM}$') # Photons transmitted to the right channel
ax[1].legend(loc='upper right',frameon=False,handlelength=1.0)
ax[1].grid(True, linestyle='--', alpha=0.6)
ax[1].set_ylim([0.,1.05])
ax[1].set_xlim([0.,8])
ax[1].xaxis.set_major_formatter(formatter)
ax[1].yaxis.set_major_formatter(formatter)
ax[1].set_xlabel('$\gamma t$',fontsize=fonts)
ax[1].text(.5, 0.9, '(b)',fontsize=fonts)

ax[2].plot(ss_correl.t_cor,np.real(ss_correl.g2_listr),linewidth = 4, color = 'b',linestyle='-',label=r'$g^{(2)}_{\rm M}$') # TLS population
# ax[1,0].plot(tlist[:-1],np.real(pulsed_pump),linewidth = 3, color = 'silver',linestyle='--',label=r'$n_{\rm TLS}^{\rm an}$') # TLS population
ax[2].plot(ss_correls_nm.t_cor,np.real(ss_correls_nm.g2_list_nm),linewidth = 3,color = 'deepskyblue',linestyle='-',label=r'$g^{(2)}_{\rm NM}$') # Photons transmitted to the right channel
# ax[0,2].plot(tlist,np.real(ref_p),linewidth = 3,color = 'brown',linestyle='--',label=r'$N^{\rm out}_{L}$') # Photons transmitted to the right channel
ax[2].legend(loc='upper right',frameon=False,handlelength=1.0)#,bbox_to_anchor=(1, 1.07),labelspacing=0.2 )
ax[2].set_xlabel('$\gamma t$',fontsize=fonts)
ax[2].grid(True, linestyle='--', alpha=0.6)
ax[2].set_ylim([0.,2.05])
ax[2].set_xlim([0.,8])
ax[2].xaxis.set_major_formatter(formatter)
ax[2].yaxis.set_major_formatter(formatter)
ax[2].text(.5, 1.8, '(c)',fontsize=fonts)


# ax[0,3].axvspan(0.0, tau, color='gray', alpha=0.2)
ax[3].plot(w_list/cw_pump,np.real(spect)/max(spect_nm),linewidth = 4, color = 'purple',linestyle='-',label=r'$\mathcal{S}^{\rm M}$') # TLS population
# ax[1,1].plot(tlist,np.real(total_nm_p),linewidth = 3,color = 'orange',linestyle='-',label=r'$N^{\rm out}_{R/L}$') # Photons transmitted to the right channel
# ax[1,1].plot(tlist[:-1],np.real(pulsed_pump),linewidth = 3, color = 'silver',linestyle='--',label=r'$n_{\rm TLS}^{\rm an}$') # TLS population
ax[3].plot(w_list_nm/cw_pump,np.real(spect_nm)/max(spect_nm),linewidth = 3,color = 'magenta',linestyle='-',label=r'$\mathcal{S}^{\rm NM}$') # Photons transmitted to the right channel
# ax[0,3].plot(tlist,np.real(ph_loop_nm_p),linewidth = 4,color = 'b',linestyle=':',label=r'$N^{\rm in}$') # Photons transmitted to the right channel
ax[3].legend(loc='upper right',frameon=False,bbox_to_anchor=(1.05, .95),handlelength=1.0)
ax[3].set_xlabel('$(\omega - \omega_L)/g$',fontsize=fonts)
ax[3].grid(True, linestyle='--', alpha=0.6)
# ax[3].set_ylim([0.,1.05])
ax[3].set_xlim([-2.,2])
ax[3].xaxis.set_major_formatter(formatter)
ax[3].yaxis.set_major_formatter(formatter)
ax[3].text(-1.8, 0.9, '(d)',fontsize=fonts)


# plt.savefig('TLS_cw2pi_phase0_tau1.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()

#%%

fonts=24

picStyle(fonts)


fig, ax = plt.subplots(2,4,figsize=(18,8))#,sharey=True,sharex=True)
# plt.subplots_adjust(wspace=0.3,hspace=0.3)
ax[0,0].plot(tlist,np.real(pop.pop),linewidth = 4, color = 'k',linestyle='-',label=r'$n_{\rm TLS}^{\rm M}$') # TLS population
ax[0,0].plot(tlist,np.real(pops_nm.pop),linewidth = 3, color = 'grey',linestyle='-',label=r'$n_{\rm TLS}^{\rm NM}$') # TLS population
ax[0,0].legend(loc='upper right',frameon=False,handlelength=1.0)
ax[0,0].grid(True, linestyle='--', alpha=0.6)
ax[0,0].set_ylim([0.,1.05])
ax[0,0].set_xlim([0.,10])
formatter = FuncFormatter(clean_ticks)
ax[0,0].xaxis.set_major_formatter(formatter)
ax[0,0].yaxis.set_major_formatter(formatter)
# ax[0].set_xticklabels([])
# ax[0,0].set_xlabel('$\gamma t$',fontsize=fonts)
ax[0,0].text(.5, 0.9, '(a)',fontsize=fonts)
ax[0,0].text(7., 0.1, r'$\phi=0$',fontsize=fonts)

# ax[1].axvspan(0.0, tau, color='gray', alpha=0.2)
ax[0,1].plot(ss_correl.t_cor,np.real(ss_correl.g1_listr),linewidth = 4, color = 'darkgreen',linestyle='-',label=r'$g^{(1)}_{\rm M}$') # TLS population
# ax[0,1].plot(tlist[:-1],np.real(pulsed_pump),linewidth = 3, color = 'silver',linestyle='--',label=r'$n_{\rm TLS}^{\rm an}$') # TLS population
ax[0,1].plot(ss_correls_nm.t_cor,np.real(ss_correls_nm.g1_list_nm),linewidth = 3,color = 'limegreen',linestyle='-',label=r'$g^{(1)}_{\rm NM}$') # Photons transmitted to the right channel
ax[0,1].legend(loc='upper right',frameon=False,handlelength=1.0)
ax[0,1].grid(True, linestyle='--', alpha=0.6)
ax[0,1].set_ylim([0.,1.05])
ax[0,1].set_xlim([0.,10])
ax[0,1].xaxis.set_major_formatter(formatter)
ax[0,1].yaxis.set_major_formatter(formatter)
# ax[0,1].set_xlabel('$\gamma t$',fontsize=fonts)
ax[0,1].text(.5, 0.9, '(b)',fontsize=fonts)

ax[0,2].plot(ss_correl.t_cor,np.real(ss_correl.g2_listr),linewidth = 4, color = 'b',linestyle='-',label=r'$g^{(2)}_{\rm M}$') # TLS population
# ax[1,0].plot(tlist[:-1],np.real(pulsed_pump),linewidth = 3, color = 'silver',linestyle='--',label=r'$n_{\rm TLS}^{\rm an}$') # TLS population
ax[0,2].plot(ss_correls_nm.t_cor,np.real(ss_correls_nm.g2_list_nm),linewidth = 3,color = 'deepskyblue',linestyle='-',label=r'$g^{(2)}_{\rm NM}$') # Photons transmitted to the right channel
# ax[0,2].plot(tlist,np.real(ref_p),linewidth = 3,color = 'brown',linestyle='--',label=r'$N^{\rm out}_{L}$') # Photons transmitted to the right channel
ax[0,2].legend(loc='upper right',frameon=False,handlelength=1.0)#,bbox_to_anchor=(1, 1.07),labelspacing=0.2 )
# ax[0,2].set_xlabel('$\gamma t$',fontsize=fonts)
ax[0,2].grid(True, linestyle='--', alpha=0.6)
ax[0,2].set_ylim([0.,2.05])
ax[0,2].set_xlim([0.,10])
ax[0,2].xaxis.set_major_formatter(formatter)
ax[0,2].yaxis.set_major_formatter(formatter)
ax[0,2].text(.5, 1.8, '(c)',fontsize=fonts)


# ax[0,3].axvspan(0.0, tau, color='gray', alpha=0.2)
ax[0,3].plot(w_list/cw_pump,np.real(spect)/max(spect),linewidth = 4, color = 'purple',linestyle='-',label=r'$\mathcal{S}^{\rm M}$') # TLS population
# ax[1,1].plot(tlist,np.real(total_nm_p),linewidth = 3,color = 'orange',linestyle='-',label=r'$N^{\rm out}_{R/L}$') # Photons transmitted to the right channel
# ax[1,1].plot(tlist[:-1],np.real(pulsed_pump),linewidth = 3, color = 'silver',linestyle='--',label=r'$n_{\rm TLS}^{\rm an}$') # TLS population
ax[0,3].plot(w_list_nm/cw_pump,np.real(spect_nm)/max(np.real(spect_nm)),linewidth = 3,color = 'magenta',linestyle='-',label=r'$\mathcal{S}^{\rm NM}$') # Photons transmitted to the right channel
# ax[0,3].plot(tlist,np.real(ph_loop_nm_p),linewidth = 4,color = 'b',linestyle=':',label=r'$N^{\rm in}$') # Photons transmitted to the right channel
ax[0,3].legend(loc='upper right',frameon=False,bbox_to_anchor=(1.05, .95),handlelength=1.0)
# ax[0,3].set_xlabel('$(\omega - \omega_L)/g$',fontsize=fonts)
ax[0,3].grid(True, linestyle='--', alpha=0.6)
ax[0,3].set_ylim([0.,1.05])
ax[0,3].set_xlim([-2.,2])
ax[0,3].xaxis.set_major_formatter(formatter)
ax[0,3].yaxis.set_major_formatter(formatter)
ax[0,3].text(-1.8, 0.9, '(d)',fontsize=fonts)

ax[1,0].plot(tlist,np.real(pop.pop),linewidth = 4, color = 'k',linestyle='-',label=r'$n_{\rm TLS}^{\rm M}$') # TLS population
ax[1,0].plot(tlist2,np.real(pops_nm2.pop),linewidth = 3, color = 'grey',linestyle='-',label=r'$n_{\rm TLS}^{\rm NM}$') # TLS population
# ax[1,0].legend(loc='upper right',frameon=False,handlelength=1.0)
ax[1,0].grid(True, linestyle='--', alpha=0.6)
ax[1,0].set_ylim([0.,1.05])
ax[1,0].set_xlim([0.,10])
formatter = FuncFormatter(clean_ticks)
ax[1,0].xaxis.set_major_formatter(formatter)
ax[1,0].yaxis.set_major_formatter(formatter)
# ax[0].set_xticklabels([])
ax[1,0].set_xlabel('$\gamma t$',fontsize=fonts)
ax[1,0].text(.5, 0.9, '(e)',fontsize=fonts)
ax[1,0].text(7, 0.1, r'$\phi=\pi$',fontsize=fonts)

# ax[1].axvspan(0.0, tau, color='gray', alpha=0.2)
ax[1,1].plot(ss_correl.t_cor,np.real(ss_correl.g1_listr),linewidth = 4, color = 'darkgreen',linestyle='-',label=r'$g^{(1)}_{\rm M}$') # TLS population
# ax[0,1].plot(tlist[:-1],np.real(pulsed_pump),linewidth = 3, color = 'silver',linestyle='--',label=r'$n_{\rm TLS}^{\rm an}$') # TLS population
ax[1,1].plot(ss_correls_nm2.t_cor,np.real(ss_correls_nm2.g1_list),linewidth = 3,color = 'limegreen',linestyle='-',label=r'$g^{(1)}_{\rm NM}$') # Photons transmitted to the right channel
# ax[1,1].legend(loc='upper right',frameon=False,handlelength=1.0)
ax[1,1].grid(True, linestyle='--', alpha=0.6)
ax[1,1].set_ylim([-0.75,1.05])
ax[1,1].set_xlim([0.,10])
ax[1,1].xaxis.set_major_formatter(formatter)
ax[1,1].yaxis.set_major_formatter(formatter)
ax[1,1].set_xlabel('$\gamma t$',fontsize=fonts)
ax[1,1].text(.5, 0.8, '(f)',fontsize=fonts)

ax[1,2].plot(ss_correl.t_cor,np.real(ss_correl.g2_listr),linewidth = 4, color = 'b',linestyle='-',label=r'$g^{(2)}_{\rm M}$') # TLS population
# ax[1,0].plot(tlist[:-1],np.real(pulsed_pump),linewidth = 3, color = 'silver',linestyle='--',label=r'$n_{\rm TLS}^{\rm an}$') # TLS population
ax[1,2].plot(ss_correls_nm2.t_cor,np.real(ss_correls_nm2.g2_list),linewidth = 3,color = 'deepskyblue',linestyle='-',label=r'$g^{(2)}_{\rm NM}$') # Photons transmitted to the right channel
# ax[0,2].plot(tlist,np.real(ref_p),linewidth = 3,color = 'brown',linestyle='--',label=r'$N^{\rm out}_{L}$') # Photons transmitted to the right channel
# ax[1,2].legend(loc='upper right',frameon=False,handlelength=1.0)#,bbox_to_anchor=(1, 1.07),labelspacing=0.2 )
ax[1,2].set_xlabel('$\gamma t$',fontsize=fonts)
ax[1,2].grid(True, linestyle='--', alpha=0.6)
# ax[1,2].set_ylim([0.,2.05])
ax[1,2].set_xlim([0.,10])
ax[1,2].xaxis.set_major_formatter(formatter)
ax[1,2].yaxis.set_major_formatter(formatter)
ax[1,2].text(.8, 4.8, '(g)',fontsize=fonts)


# ax[0,3].axvspan(0.0, tau, color='gray', alpha=0.2)
ax[1,3].plot(w_list/cw_pump,np.real(spect)/max(spect),linewidth = 4, color = 'purple',linestyle='-',label=r'$\mathcal{S}^{\rm M}$') # TLS population
# ax[1,1].plot(tlist,np.real(total_nm_p),linewidth = 3,color = 'orange',linestyle='-',label=r'$N^{\rm out}_{R/L}$') # Photons transmitted to the right channel
# ax[1,1].plot(tlist[:-1],np.real(pulsed_pump),linewidth = 3, color = 'silver',linestyle='--',label=r'$n_{\rm TLS}^{\rm an}$') # TLS population
ax[1,3].plot(w_list_nm2/cw_pump,np.real(spect_nm2)/max(np.real(spect_nm2)),linewidth = 3,color = 'magenta',linestyle='-',label=r'$\mathcal{S}^{\rm NM}$') # Photons transmitted to the right channel
# ax[0,3].plot(tlist,np.real(ph_loop_nm_p),linewidth = 4,color = 'b',linestyle=':',label=r'$N^{\rm in}$') # Photons transmitted to the right channel
# ax[1,3].legend(loc='upper right',frameon=False,bbox_to_anchor=(1.05, .95),handlelength=1.0)
ax[1,3].set_xlabel('$(\omega - \omega_L)/\Omega$',fontsize=fonts)
ax[1,3].grid(True, linestyle='--', alpha=0.6)
# ax[1,3].set_ylim([0.,1.05])
ax[1,3].set_xlim([-2.,2])
ax[1,3].xaxis.set_major_formatter(formatter)
ax[1,3].yaxis.set_major_formatter(formatter)
ax[1,3].text(-1.8, 0.9, '(h)',fontsize=fonts)

# plt.savefig('TLS_cw2pi_tau1.pdf', format='pdf', dpi=600, bbox_inches='tight')
# plt.savefig('TLS_cw2pi_tau0p1.png', format='png', dpi=600, bbox_inches='tight')
plt.show()
# %%
