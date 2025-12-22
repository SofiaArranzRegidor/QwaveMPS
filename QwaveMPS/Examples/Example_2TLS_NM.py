#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 11:33:34 2025

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

d_sys1=2 # first tls bin dimension 
d_sys2=2 # second tls bin dimension
d_sys_total=np.array([d_sys1, d_sys2]) #total system bin dimension

#Choose the coupling:
gamma_l1,gamma_r1=qmps.coupling('symmetrical',gamma=1)
gamma_l2,gamma_r2=qmps.coupling('symmetrical',gamma=1)

#Define input parameters
input_params = qmps.parameters.InputParams(
    delta_t=0.05,
    tmax = 5,
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=gamma_l1,
    gamma_r = gamma_r1,
    gamma_l2 = gamma_l2,
    gamma_r2 = gamma_r2,
    bond=8,
    phase=np.pi,
    tau=0.5
)


#Make a tlist for plots:
tmax=input_params.tmax
delta_t=input_params.delta_t
tlist=np.arange(0,tmax+delta_t,delta_t)

""" Choose the initial state and coupling"""

i_s01=qmps.states.i_se()
i_s02= qmps.states.i_sg()
i_s0=np.kron(i_s01,i_s02)

#We can start with one excited and one ground, both excited, both ground, 
# or with an entangled state like the following one
# i_s0=1/np.sqrt(2)*(np.kron(i_s01,i_s02)+np.kron(i_s02,i_s01))

i_n0 = qmps.states.vacuum(tmax,input_params)

start_time=t.time()
"""Choose the Hamiltonian"""

hm=qmps.hamiltonian_2tls_nmar(input_params)


""" Time evolution of the system"""

bins = qmps.t_evol_nmar(hm,i_s0,i_n0,input_params)


""" Calculate population dynamics"""

pop=qmps.pop_dynamics_2tls(bins,input_params)

print("--- %s seconds ---" %(t.time() - start_time))
#%%

fonts=15
pic_style(fonts)


fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tlist,np.real(pop.pop1),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{\rm TLS1}$')
plt.plot(tlist,np.real(pop.pop2),linewidth = 3, color = 'skyblue',linestyle='--',label=r'$n_{\rm TLS2}$')
plt.plot(tlist,np.real(pop.int_n_r),linewidth = 3,color = 'orange',linestyle='-',label='T')
plt.plot(tlist,np.real(pop.int_n_l),linewidth = 3,color = 'b',linestyle=':',label='R')
plt.plot(tlist,np.real(pop.total),linewidth = 3,color = 'g',linestyle='-',label='Total')
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95),labelspacing=0.2)
plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,1.05])
plt.xlim([0.,tmax])
plt.tight_layout()
plt.show()