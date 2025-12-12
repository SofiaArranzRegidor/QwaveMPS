#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 15:08:26 2025

@author: sofia
"""

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FuncFormatter
import numpy as np

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import QwaveMPS.src as qmps

#Parameters for plots style

def pic_style(fontsize):
    rc('font',size=fontsize)

def clean_ticks(x, pos):
    # Only show decimals if not an integer
    return f'{x:g}'

#%%

""""Choose the simulation parameters"""

"Choose the time step and end time"

delta_t = 0.02
tmax = 25
tlist=np.arange(0,tmax+delta_t,delta_t)
d_t_l=2 #Time right channel bin dimension
d_t_r=2 #Time left channel bin dimension
d_t_total=np.array([d_t_l,d_t_r])

d_sys1=2 # tls bin dimension
d_sys_total=np.array([d_sys1]) #total system bin (in this case only 1 tls)

"Choose max bond dimension"

# bond = d_t_total^(number of excitations)
bond=18


""" Choose the initial state and coupling"""

i_s0=qmps.states.i_se()

i_n0 = qmps.states.vacuum(tmax, delta_t, d_t_total)


gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1)


"""Choose the Hamiltonian"""

#CW Drive
cw_pump=2*np.pi

Hm=qmps.hamiltonian_1tls(delta_t, gamma_l, gamma_r,d_sys_total,d_t_total,cw_pump)


"""Calculate time evolution of the system"""

sys_b,time_b,cor_b,schmidt = qmps.t_evol_mar(Hm,i_s0,i_n0,delta_t,tmax,bond,d_sys_total,d_t_total)


"""Calculate population dynamics"""

pop,tbins_r,tbins_l,int_n_r,int_n_l,total=qmps.pop_dynamics(sys_b,time_b,delta_t,d_sys_total,d_t_total)


"""Calculate steady state correlations"""
t_cor,g1_listl,g1_listr,g2_listl,g2_listr,c1_l,c1_r,c2_l,c2_r=qmps.steady_state_correlations(cor_b,pop,delta_t,d_t_total,bond)


"""Calculate the spectrum"""

spect,w_list=qmps.spectrum_w(delta_t,c1_r)

#%% Population dynamics

fonts=15
pic_style(fonts)


fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tlist,np.real(pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$') # TLS population
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95),labelspacing=0.2,fontsize=fonts)
plt.xlabel('Time, $\gamma t$',fontsize=fonts)
plt.ylabel('Populations',fontsize=fonts)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([0.,1.05])
plt.xlim([0.,10])
plt.tight_layout()
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
# plt.savefig('TLS_sym_decay.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()

#%% Steady state correlation dynamics

fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(t_cor,np.real(g2_listr),linewidth = 3, color = 'lime',linestyle='-',label=r'$g^{(2)}_R$') 
plt.plot(t_cor,np.real(g1_listr),linewidth = 3, color = 'darkgreen',linestyle='-',label=r'$g^{(1)}_R$') 
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95),labelspacing=0.2,fontsize=fonts)
plt.xlabel('Time, $\gamma t$',fontsize=fonts)
plt.ylabel('Correlations',fontsize=fonts)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([0.,2.05])
plt.xlim([0.,10])
plt.tight_layout()
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
# plt.savefig('TLS_sym_decay.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()

#%% Long time spectrum

fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(w_list/cw_pump,np.real(spect)/max(spect),linewidth = 4, color = 'purple',linestyle='-') # TLS population
# plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95),labelspacing=0.2,fontsize=fonts)
plt.xlabel('$(\omega - \omega_L)/g$',fontsize=fonts)
plt.ylabel('Spectrum',fontsize=fonts)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([0.,1.05])
plt.xlim([-3.,3])
plt.tight_layout()
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
# plt.savefig('TLS_sym_decay.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()

