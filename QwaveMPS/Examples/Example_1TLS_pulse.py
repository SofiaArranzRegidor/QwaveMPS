#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is an example of a single two-level system (TLS)
interacting with a Fock state pulse. 

All the examples are in units of the TLS total decay rate, gamma. Hence, in general, gamma=1.

It covers two cases:
    1. Example with a 1-photon tophat pulse
    2. Example with a 2-photon gaussian pulse

Computes time evolution, population dynamics, and first and second-order correlations (for case 1),
with example plots of the populations for both cases.


Requirements: 
    
ncon https://pypi.org/project/ncon/. To install it, write the following on your console: 
    
    pip install ncon 
        
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


#%% 1 photon Tophat Pulse


""""Choose the simulation parameters"""

#Choose the bins:
d_t_l=3 #Time right channel bin dimension
d_t_r=3 #Time left channel bin dimension
d_t_total=np.array([d_t_l,d_t_r])

d_sys1=2 # tls bin dimension
d_sys_total=np.array([d_sys1]) #total system bin (in this case only 1 tls)

#Choose the coupling:
gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1)

#Define input parameters
input_params = qmps.parameters.InputParams(
    delta_t=0.05,
    tmax = 5,
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


""" Choose the initial state and pulse parameters"""

# Pulse parameters por a 1-photon tophat pulse
pulse_time = 1 #length of the pulse in time
photon_num = 1 #number of photons

i_s0=qmps.states.i_sg()

#pulse envelope shape
pulse_env=qmps.states.tophat_envelope(pulse_time, input_params)

i_n0 = qmps.states.fock_pulse(pulse_env,pulse_time, input_params,photon_num, direction='R')


"""Choose the Hamiltonian"""

Hm=qmps.hamiltonian_1tls(input_params)

#To track computational time of populations
start_time=t.time()

"""Calculate time evolution of the system"""

bins = qmps.t_evol_mar(Hm,i_s0,i_n0,input_params)


"""Calculate population dynamics"""

pop=qmps.pop_dynamics(bins,input_params)

print("--- %s seconds ---" %(t.time() - start_time))


"""Calculate correlations"""

#To track computational time of g1
start_time=t.time()

g1_correl=qmps.first_order_correlation(bins, input_params)


print("G1 correl--- %s seconds ---" %(t.time() - start_time))

#To track computational time of g2
start_time=t.time()

g2_correl=qmps.second_order_correlation(bins, input_params)


print("G2 correl--- %s seconds ---" %(t.time() - start_time))

#%%
fonts=15
pic_style(fonts)


fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tlist,np.real(pop.int_n_r),linewidth = 3,color = 'orange',linestyle='-',label='Transmission') # Photons transmitted to the right channel
plt.plot(tlist,np.real(pop.int_n_l),linewidth = 3,color = 'b',linestyle=':',label='Reflection') # Photons reflected to the left channel
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
plt.show()



#%% 2 photon Gaussian pulse


""" Updated input field and simulation length"""


input_params.tmax=12
tmax=input_params.tmax
tlist=np.arange(0,tmax+delta_t,delta_t)

#We need a higher bond dimension for a 2-photon pulse
input_params.bond=8

# Pulse parameters por a 2-photon gaussian pulse
pulse_time = tmax
photon_num = 2
gaussian_center = 4
gaussian_width = 1


i_s0=qmps.states.i_sg()

pulse_envelope = qmps.states.gaussian_envelope(pulse_time, input_params, gaussian_width, gaussian_center)
i_n0 = qmps.states.fock_pulse(pulse_envelope,pulse_time, input_params, photon_num, direction='R')


start_time=t.time()
"""Calculate time evolution of the system"""

bins = qmps.t_evol_mar(Hm,i_s0,i_n0,input_params)


"""Calculate population dynamics"""

pop=qmps.pop_dynamics(bins,input_params)

print("2-photon pop--- %s seconds ---" %(t.time() - start_time))

#%%

fonts=15
pic_style(fonts)


fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tlist,np.real(pop.int_n_r),linewidth = 3,color = 'orange',linestyle='-',label='Transmission') # Photons transmitted to the right channel
plt.plot(tlist,np.real(pop.int_n_l),linewidth = 3,color = 'b',linestyle=':',label='Reflection') # Photons reflected to the left channel
plt.plot(tlist,np.real(pop.pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$') # TLS population
plt.plot(tlist,np.real(pop.total),linewidth = 3,color = 'g',linestyle='-',label='Total') # Conservation check (for one excitation it should be 1)
plt.legend(loc='center right')
plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([0.,2.05])
plt.xlim([0.,tmax])
plt.tight_layout()
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
# plt.savefig('TLS_chiral_decay.pdf', format='pdf', dpi=600, bbox_inches='tight')
plt.show()
