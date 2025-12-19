#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This is a basic example of a single two-level system (TLS) decaying into the waveguide. 
All the examples are in units of the TLS total decay rate, gamma. Hence, in general, gamma=1.

It covers two cases:
    
    1. Symmetrical coupling into the waveguide
    2. Chiral coupling, where the TLS is only coupled to the right channel of the waveguide.

Structure:    
    
    1. Setup of the simulation parameters.
    
        - Time step (delta_t)
        - Maximum time (tmax)
        - Size of time bin (d_t_total). This is the field Hilbert subspace at each time step.
        (In this case we allow one photon per time step and per right (d_t_r) and left (d_t_l) channels.
         Hence, the subspace is d_t_total=2*2=4)
        - Size of the system bin (d_sys). This is the TLS Hilbert subspace 
        (for a single TLS, d_sys=2).
        - Maximum bond dimension (bond). bond=d_t_total^(number of excitations).    
        Starting with the TLS excited and field in vacuum, 1 excitation => bond=2
        
    2. Initial state and coupling configuration. 
    
        - Choice the system initial state (i_s0). Here, initially excited, 
            i_s0 = qmps.states.i_se()
        - Choice of the waveguide initial state (i_n0). Here, starting in vacuum,
            i_n0 = qmps.states.i_ng(d_t_total)
        - Choice of coupling. Here, it is first calculated with symmetrical coupling,
            gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1)            
          and the with chiral coupling,         
            gamma_l,gamma_r=qmps.coupling('chiral',gamma=1)
            
    3. Selection of the corresponding Hamiltonian.
    
    4. Time evolution calculation.
    
    5. Observables alculation (time dyanamics populations).
    
    6. Example plot containing,
    
        - Integrated photon flux transmitted to the right channel
        - Integrated photon flux reflected to the left channel
        - TLS population
        - Conservation check (for one excitation it should be 1)
    
Repeat for both cases (symmetrical and chiral).

"""


"""    

Requirements: 
    
ncon https://pypi.org/project/ncon/. To install it, write the following on your console: 
    
    pip install ncon 
    
"""

#%%

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


#%% Symmetrical Solution


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
    delta_t=0.05,
    tmax = 8,
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=gamma_l,
    gamma_r = gamma_r,  
    bond=4
)

#Make a tlist for plots:
tmax=input_params.tmax
delta_t=input_params.delta_t
tlist=np.arange(0,tmax+delta_t,delta_t)


""" Choose the initial state"""

i_s0=qmps.states.i_se() #TLS initially excited

i_n0 = qmps.states.vacuum(tmax,input_params) #waveguide in vacuum



"""Choose the Hamiltonian"""

Hm=qmps.hamiltonian_1tls(input_params)


"""Calculate time evolution of the system"""

bins = qmps.t_evol_mar(Hm,i_s0,i_n0,input_params)


"""Calculate population dynamics"""

pop = qmps.pop_dynamics(bins,input_params)



"""Plotting the results"""

fonts=15
pic_style(fonts)


fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tlist,np.real(pop.int_n_r),linewidth = 3,color = 'orange',linestyle='-',label=r'$N^{\rm out}_{R}$') # Photons transmitted to the right channel
plt.plot(tlist,np.real(pop.int_n_l),linewidth = 3,color = 'b',linestyle=':',label=r'$N^{\rm out}_{L}$') # Photons reflected to the left channel
plt.plot(tlist,np.real(pop.pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$') # TLS population
plt.plot(tlist,np.real(pop.total),linewidth = 3,color = 'g',linestyle='-',label='Total') # Conservation check (for one excitation it should be 1)
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95),labelspacing=0.2,fontsize=fonts)
plt.xlabel('Time, $\gamma t$',fontsize=fonts)
plt.ylabel('Populations',fontsize=fonts)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([0.,1.05])
plt.xlim([0.,tmax])
plt.tight_layout()
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
# plt.savefig('TLS_sym_decay.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()


#%% Right chiral Solution


""" Updated coupling"""

gamma_l,gamma_r=qmps.coupling('chiral_r',gamma=1)

input_params.gamma_l=gamma_l
input_params.gamma_r=gamma_r

"""Choose the Hamiltonian"""

hm=qmps.hamiltonian_1tls(input_params)

"""Calculate time evolution of the system"""

bins = qmps.t_evol_mar(hm,i_s0,i_n0,input_params)


"""Calculate population dynamics"""

pop_ch=qmps.pop_dynamics(bins,input_params)


"""Plotting the results"""

fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tlist,np.real(pop_ch.int_n_r),linewidth = 3,color = 'orange',linestyle='-',label='Transmission') # Photons transmitted to the right channel
plt.plot(tlist,np.real(pop_ch.int_n_l),linewidth = 3,color = 'b',linestyle=':',label='Reflection') # Photons reflected to the left channel
plt.plot(tlist,np.real(pop_ch.pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$') # TLS population
plt.plot(tlist,np.real(pop_ch.total),linewidth = 3,color = 'g',linestyle='-',label='Total') # Conservation check (for one excitation it should be 1)
plt.legend(loc='center right')
plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([0.,1.05])
plt.xlim([0.,tmax])
plt.tight_layout()
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
# plt.savefig('TLS_chiral_decay.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()

