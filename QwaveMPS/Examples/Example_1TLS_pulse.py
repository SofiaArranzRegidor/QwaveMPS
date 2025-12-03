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
        - Size of time bin (d_t). This is the field Hilbert subspace at each time step.
        (In this case we allow one photon per time step and per right and left channels.
         Hence, the subspace is d_t=2*2=4)
        - Size of the system bin (d_sys). This is the TLS Hilbert subspace 
        (for a single TLS, d_sys=2).
        - Maximum bond dimension (bond). bond=d_t*(number of excitations).    
        Starting with the TLS excited and field in vacuum, 1 excitation => bond=2
        
    2. Initial state and coupling configuration. 
    
        - Choice the system initial state (i_s0). Here, initially excited, 
            i_s0 = QM.states.i_se()
        - Choice of the waveguide initial state (i_n0). Here, starting in vacuum,
            i_n0 = QM.states.i_ng(d_t)
        - Choice of coupling. Here, it is first calculated with symmetrical coupling,
            gamma_l,gamma_r=QM.coupling('symmetrical',gamma=1)            
          and the with chiral coupling,         
            gamma_l,gamma_r=QM.coupling('chiral',gamma=1)
            
    3. Selection of the corresponding Hamiltonian.
    
    4. Time evolution calculation.
    
    5. Observables alculation (time dyanamics populations).
    
    6. Example plot containing,
    
        - Photons transmitted to the right channel
        - Photons reflected to the left channel
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


#%% 1 photon Tophat Pulse


""""Choose the simulation parameters"""

"Choose the time step and end time"

delta_t = 0.1
tmax = 5
tlist=np.arange(0,tmax+delta_t,delta_t)
d_t_l=3 #Time right channel bin dimension
d_t_r=3 #Time left channel bin dimension
d_t_total=np.array([d_t_l,d_t_r])

d_sys1=2 # tls bin dimension
d_sys_total=np.array([d_sys1]) #total system bin (in this case only 1 tls)

"Choose max bond dimension"

# bond = d_t_total*(number of excitations)
bond=4


""" Choose the initial state and coupling"""
# Pulse parameters
pulse_time = 1
photon_num = 1


i_s0=qmps.states.i_sg()


pulse_env=qmps.states.tophat_envelope(pulse_time, delta_t)

i_n0 = qmps.states.fock_pulse(pulse_env,pulse_time, delta_t, d_t_total, bond, photon_num_r=photon_num)

gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1)


"""Choose the Hamiltonian"""

Hm=qmps.hamiltonian_1tls(delta_t, gamma_l, gamma_r,d_sys_total,d_t_total)


"""Calculate time evolution of the system"""

sys_b,time_b,cor_b,schmidt = qmps.t_evol_mar(Hm,i_s0,i_n0,delta_t,tmax,bond,d_sys_total,d_t_total)


"""Calculate population dynamics"""

pop,tbins_r,tbins_l,trans,ref,total=qmps.pop_dynamics(sys_b,time_b,delta_t,d_sys_total,d_t_total)


g1_rr_matrix,g1_ll_matrix,g1_rl_matrix,g1_lr_matrix=qmps.first_order_correlation(cor_b, delta_t,d_t_total,bond)

#%%
fonts=15
pic_style(fonts)


fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tlist,np.real(trans),linewidth = 3,color = 'orange',linestyle='-',label='Transmission') # Photons transmitted to the right channel
plt.plot(tlist,np.real(ref),linewidth = 3,color = 'b',linestyle=':',label='Reflection') # Photons reflected to the left channel
plt.plot(tlist,np.real(pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$') # TLS population
plt.plot(tlist,np.real(total),linewidth = 3,color = 'g',linestyle='-',label='Total') # Conservation check (for one excitation it should be 1)
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



#%% 2 photon Gaussian pulse


""" Updated input field and simulation length"""
tmax = 12
tlist=np.arange(0,tmax+delta_t,delta_t)

pulse_time = tmax
photon_num = 2
gaussian_mean = 4
gaussian_variance = 1


bond=8

i_s0=qmps.states.i_sg()

pulse_envelope = qmps.states.gaussian_envelope(pulse_time, delta_t, gaussian_variance, gaussian_mean)
i_n0 = qmps.states.fock_pulse(pulse_envelope,pulse_time, delta_t, d_t_total, bond, photon_num_r=photon_num)



"""Calculate time evolution of the system"""

sys_b,time_b,cor_b,schmidt = qmps.t_evol_mar(Hm,i_s0,i_n0,delta_t,tmax,bond,d_sys_total,d_t_total)


"""Calculate population dynamics"""

pop,tbins_r,tbins_l,trans,ref,total=qmps.pop_dynamics(sys_b,time_b,delta_t,d_sys_total,d_t_total)



fonts=15
pic_style(fonts)


fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tlist,np.real(trans),linewidth = 3,color = 'orange',linestyle='-',label='Transmission') # Photons transmitted to the right channel
plt.plot(tlist,np.real(ref),linewidth = 3,color = 'b',linestyle=':',label='Reflection') # Photons reflected to the left channel
plt.plot(tlist,np.real(pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$') # TLS population
plt.plot(tlist,np.real(total),linewidth = 3,color = 'g',linestyle='-',label='Total') # Conservation check (for one excitation it should be 1)
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
