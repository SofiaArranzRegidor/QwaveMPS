#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 10:12:52 2025

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

"""One first need to choose the delay time tau:

In this case it is the roundtrip time of the feedback loop (back and forth from the mirror)

"""

""" Example with constructive feedback:

Choose a constructive feedback phase"""

#Choose the bins:
d_sys1=2 # tls bin dimension
d_sys_total=np.array([d_sys1]) #total system bin (in this case only 1 tls)

d_t=2 #time bin dimension of one channel
d_t_total=np.array([d_t]) #single channel for mirror case

#Copuling is symmetric by default
gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1)

#Define input parameters
input_params = qmps.parameters.InputParams(
    delta_t=0.03,
    tmax = 5,
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=gamma_l,
    gamma_r = gamma_r,  
    bond=4,
    tau=0.5,
    phase=np.pi
)


#Make a tlist for plots:
tmax=input_params.tmax
delta_t=input_params.delta_t
tlist=np.arange(0,tmax+delta_t,delta_t)


""" Choose the initial state"""

i_s0=qmps.states.i_se()
i_n0 = qmps.states.vacuum(tmax,input_params)


"""Choose the Hamiltonian"""

Hm=qmps.hamiltonian_1tls_feedback(input_params)


""" Time evolution of the system"""

bins = qmps.t_evol_nmar(Hm,i_s0,i_n0,input_params)


""" Calculate population dynamics"""

pop=qmps.pop_dynamics_1tls_nmar(bins,input_params)


#%%

fonts=15
pic_style(fonts)

fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tlist,np.real(pop.pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{\rm TLS}$')
plt.plot(tlist,np.real(pop.trans),linewidth = 3,color = 'orange',linestyle='-',label='T')
plt.plot(tlist,np.real(pop.loop),linewidth = 3,color = 'b',linestyle=':',label=r'$n_{\rm loop}$')
plt.plot(tlist,np.real(pop.total),linewidth = 3,color = 'g',linestyle='-',label='Total')
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95),labelspacing=0.2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim([0.,tmax])
plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.tight_layout()
plt.show()


#%%

""" Example with destructive feedback

Choose a destructive feedback phase"""

input_params.phase=0

"""Choose the Hamiltonian"""

hm=qmps.hamiltonian_1tls_feedback(input_params)


""" Time evolution of the system"""

bins = qmps.t_evol_nmar(hm,i_s0,i_n0,input_params)


""" Calculate population dynamics"""

pop_d = qmps.pop_dynamics_1tls_nmar(bins,input_params)


#%%

fonts=15
pic_style(fonts)

fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tlist,np.real(pop_d.pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{\rm TLS}$')
plt.plot(tlist,np.real(pop_d.trans),linewidth = 3,color = 'orange',linestyle='-',label='T')
plt.plot(tlist,np.real(pop_d.loop),linewidth = 3,color = 'b',linestyle=':',label=r'$n_{\rm loop}$')
plt.plot(tlist,np.real(pop_d.total),linewidth = 3,color = 'g',linestyle='-',label='Total')
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95),labelspacing=0.2)
plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim([0.,tmax])
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.tight_layout()
plt.show()
