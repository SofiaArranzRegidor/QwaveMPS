#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:18:14 2026

@author: sofia
"""

#%% 
# Imports
#--------

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FuncFormatter
import numpy as np

import QwaveMPS as qmps
import QwaveMPS.operators as qops
import time as t


def pic_style(fontsize):
    rc('font',size=fontsize)

def clean_ticks(x, pos):
    # Only show decimals if not an integer
    return f'{x:g}'


#%% 
# Pulse
#----------------------------------
#
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#Choose the simulation parameters
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

""""Choose the simulation parameters"""
#Choose the bin dimensions
# Here setting to 2 to accommodate a 1 photon space:
d_t_l=4 #Time right channel bin dimension
d_t_r=4 #Time left channel bin dimension
d_t_total=np.array([d_t_l,d_t_r])

d_sys1=2 # tls bin dimension
d_sys_total=np.array([d_sys1]) #total system bin (in this case only 1 tls)

#Choose the coupling:
gamma_l,gamma_r=qmps.states.coupling('symmetrical',gamma=1)

#Define input parameters
input_params = qmps.parameters.InputParams(
    delta_t=0.05,
    tmax = 12,
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=gamma_l,
    gamma_r = gamma_r,  
    bond_max=16
)

#Make a tlist for plots:
tmax=input_params.tmax
delta_t=input_params.delta_t
tlist=np.arange(0,tmax+delta_t,delta_t)

#%%
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#Choose the initial state and Hamiltonian
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#In this case, we need to also specify the pulse parameters
#that will go in the photonic part of the initial state

""" Choose the initial state and tophat pulse parameters"""
sys_initial_state=qmps.states.tls_ground()

# Pulse parameters por a 1-photon tophat pulse
pulse_time = 2#length of the pulse in time units of gamma
photon_num = 1 #number of photons
alpha = 1#np.sqrt(2)

# Pulse parameters for a 2-photon gaussian pulse
pulse_time = tmax
gaussian_center = 4
gaussian_width = 0.5

pulse_env = qmps.states.gaussian_envelope(tmax, input_params, gaussian_width, gaussian_center)

wg_initial_state = qmps.states.coherent_pulse([None, pulse_env],tmax, input_params, [0,alpha])

"""Choose the Hamiltonian"""
Hm=qmps.hamiltonians.hamiltonian_1tls(input_params)

#To track computational time of populations
start_time=t.time()

#%%
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#Calculate the time evolution
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#Time evolution calculation in the Markovian regime

"""Calculate time evolution of the system"""

bins = qmps.simulation.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)

#%%
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#Calculate the population dynamics
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
"""Calculate population dynamics"""
# Photonic operators
left_flux_op = qops.b_dag_l(input_params) @ qops.b_l(input_params)
right_flux_op = qops.b_dag_r(input_params) @ qops.b_r(input_params)
photon_flux_ops = [left_flux_op, right_flux_op]

tls_pop = qops.single_time_expectation(bins.system_states, qops.tls_pop())
photon_fluxes = qops.single_time_expectation(bins.output_field_states, photon_flux_ops)
flux_in = qops.single_time_expectation(bins.input_field_states, photon_flux_ops)

# Calculate total quanta that has entered the system, tls population + net flux out
total_quanta = tls_pop + np.cumsum(photon_fluxes[0] + photon_fluxes[1]) * delta_t
print("--- %s seconds ---" %(t.time() - start_time))

#%%
#^^^^^^^^^^^^^^^^
#Plot the results
#^^^^^^^^^^^^^^^^
#

plt.plot(tlist,np.real(photon_fluxes[1]),linewidth = 3,color = 'violet',linestyle='-',label=r'$n_{R}$') # Photon flux transmitted to the right channel
plt.plot(tlist,np.real(photon_fluxes[0]),linewidth = 3,color = 'green',linestyle=':',label=r'$n_{L}$') # Photon flux reflected to the left channel
plt.plot(tlist,np.real(tls_pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$') # TLS population
plt.plot(tlist,np.real(flux_in[1]),linewidth = 3, color = 'grey',linestyle='--',label=r'$n_{R}^{\rm in}$') # Photon flux in from right
plt.plot(tlist,np.real(total_quanta),linewidth = 3,color = 'g',linestyle='-',label='Total') # Conservation check (for one excitation it should be 1)
plt.legend()
plt.xlabel(r'Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
# plt.ylim([0.,1.05])
plt.xlim([0.,tmax])
plt.show()



