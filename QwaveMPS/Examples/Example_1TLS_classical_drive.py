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
import time as t


#Parameters for plots style

def pic_style(fontsize):
    rc('font',size=fontsize)

def clean_ticks(x, pos):
    # Only show decimals if not an integer
    return f'{x:g}'

#%%


""""Choose the simulation parameters"""

#Choose the bins:
d_t_l=2 #Time right channel bin dimension
d_t_r=2 #Time left channel bin dimension
d_t_total=np.array([d_t_l,d_t_r])

d_sys1=2 # tls bin dimension
d_sys_total=np.array([d_sys1]) #total system bin (in this case only 1 tls)

#Choose the coupling:
gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1)

#Define input parameters
input_params = qmps.parameters.InputParams(
    delta_t=0.02,
    tmax = 35,
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=gamma_l,
    gamma_r = gamma_r,  
    bond=18
)

#Make a tlist for plots:
tmax=input_params.tmax
delta_t=input_params.delta_t
tlist=np.arange(0,tmax+(delta_t/2),delta_t)


""" Choose the initial state and coupling"""

i_s0=qmps.states.i_se()

i_n0 = qmps.states.vacuum(tmax,input_params)

start_time=t.time()

"""Choose the Hamiltonian"""

#CW Drive
cw_pump=2*np.pi

Hm=qmps.hamiltonian_1tls(input_params,cw_pump)


"""Calculate time evolution of the system"""

bins = qmps.t_evol_mar(Hm,i_s0,i_n0,input_params)


"""Calculate population dynamics"""

pop=qmps.pop_dynamics(bins,input_params)

print("--- %s seconds ---" %(t.time() - start_time))


"""Calculate steady state correlations"""

start_time=t.time()

ss_correl=qmps.steady_state_correlations(bins,pop,input_params)

print("ss correl --- %s seconds ---" %(t.time() - start_time))


"""Calculate the spectrum"""

start_time=t.time()

spect,w_list=qmps.spectrum_w(delta_t,ss_correl.c1_r)

print("spectrum --- %s seconds ---" %(t.time() - start_time))

#%% Population dynamics

fonts=15
pic_style(fonts)


fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tlist,np.real(pop.pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$') # TLS population
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
plt.plot(ss_correl.t_cor,np.real(ss_correl.g2_listr),linewidth = 3, color = 'lime',linestyle='-',label=r'$g^{(2)}_R$') 
plt.plot(ss_correl.t_cor,np.real(ss_correl.g1_listr),linewidth = 3, color = 'darkgreen',linestyle='-',label=r'$g^{(1)}_R$') 
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
plt.plot(w_list/cw_pump,np.real(spect)/max(np.real(spect)),linewidth = 4, color = 'purple',linestyle='-') # TLS population
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

