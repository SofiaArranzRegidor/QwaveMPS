#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is an example of a single two-level system (TLS) interacting with 
a two photon Fock state pulse. 

All the examples are in units of the TLS total decay rate, gamma. Hence, in general, gamma=1.

It covers the case of a 2-photon tophat pulse

This example efficiently computes correlation functions using a single function call.
It then computes the time dependent spectrum, the spectral intensity, and checks that
the long time limit of the spectrum matches the time integral of the spectral intensity.

This features the following example plots:
    1. Time dependent spectral intensity
    2. Time dependent spectrum
    3. Long-time spectrum / time integrated spectral intensity




Computes the time dependent spectrum and spectral intensity in transmission


Requirements: 
    
ncon https://pypi.org/project/ncon/. To install it, write the following on your console: 
    
    pip install ncon 

References:
    Phys. Rev. Research 7, 023295, Arranz-Regidor et. al. (2025)

"""

#%% Imports
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FuncFormatter
import numpy as np

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import QwaveMPS.src as qmps
import time as t

#Parameters for plots style

def pic_style(fontsize):
    rc('font',size=fontsize)

def clean_ticks(x, pos):
    # Only show decimals if not an integer
    return f'{x:g}'
formatter = FuncFormatter(clean_ticks)


#%% 1 photon Tophat Pulse


""""Choose the simulation parameters"""

#Choose the bin dimensions
# Dimension of 3 for 2 photon input/observables
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
    tmax = 8,
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=gamma_l,
    gamma_r = gamma_r,  
    bond_max=8
)

#Make a tlist for plots:
tmax=input_params.tmax
delta_t=input_params.delta_t
tlist=np.arange(0,tmax+delta_t,delta_t)


""" Choose the initial state and pulse parameters"""

# Pulse parameters por a 1-photon tophat pulse
pulse_time = 2 #length of the pulse in time units of gamma
photon_num = 2 #number of photons

sys_initial_state=qmps.states.tls_ground()

#pulse envelope shape
pulse_env=qmps.states.tophat_envelope(pulse_time, input_params)

# Create the pulse envelope
wg_initial_state = qmps.states.fock_pulse(pulse_env,pulse_time, photon_num, input_params, direction='R')

# Multiple pulses may be appended in the usual list appending way
#wg_initial_state += qmps.states.fock_pulse(pulse_env,pulse_time, input_params,photon_num, direction='L')

"""Choose the Hamiltonian"""

Hm=qmps.hamiltonian_1tls(input_params)

#To track computational time of simulation and correlations
start_time=t.time()

"""Calculate time evolution of the system"""

bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)


"""Calculate G1 in transmission and reflection, though use 4 op to calculate G2 at same time"""
# This is much faster than two separate correlation function calls
# Requires padding the last two operators of the correlation function with identities so that we have
# <A(t)B(t+t')II> = <A(t)B(t+t')>

a_ops = []; b_ops = []; c_ops = []; d_ops = []
b_dag_r = qmps.b_dag_r(input_params) ; b_r = qmps.b_r(input_params)
b_dag_l = qmps.b_dag_l(input_params); b_l = qmps.b_l(input_params)
dim = b_r.shape[0]

# Add Right moving (transmission) first
a_ops.append(b_dag_r)
b_ops.append(b_r)
c_ops.append(np.eye(dim))
d_ops.append(np.eye(dim))


# Add left moving correlation operators
a_ops.append(b_dag_l)
b_ops.append(b_l)
c_ops.append(np.eye(dim))
d_ops.append(np.eye(dim))


"""Also calculate the G2"""
# Add Right moving (transmission) first
a_ops.append(b_dag_r)
b_ops.append(b_dag_r)
c_ops.append(b_r)
d_ops.append(b_r)


# Add left moving correlation operators
a_ops.append(b_dag_l)
b_ops.append(b_dag_l)
c_ops.append(b_l)
d_ops.append(b_l)

"""Can also consider a kind of two time squeezing operators in transmission"""
X = b_dag_r + b_r
a_ops.append(X)
b_ops.append(X)
c_ops.append(np.eye(dim))
d_ops.append(np.eye(dim))


correlations, correlation_tlist = qmps.correlation_4op_2t(bins.correlation_bins, a_ops, b_ops, c_ops, d_ops, input_params)

print("--- %s seconds ---" %(t.time() - start_time))
#%%
#To track computational time of spectra and spectral intensity
start_time=t.time()

# Index of the correlations of measured spectrum/intensity
index = 0

# Padding used for better spectral resolution
padding_factor = 5 
padding = correlations[index].shape[0]*padding_factor

spectral_intensity, w_list_intensity = qmps.spectral_intensity(correlations[index], input_params, padding=padding)
print("Spectral intensity--- %s seconds ---" %(t.time() - start_time))

start_time=t.time()
time_dependent_spectrum, w_list_spectrum = qmps.time_dependent_spectrum(correlations[index], input_params, padding=padding)
print("Time dependent spectra--- %s seconds ---" %(t.time() - start_time))



"""Graph Examples"""
fonts=15
pic_style(fonts)

"""Example: Spectral Intensity"""

X,Y = np.meshgrid(w_list_intensity,correlation_tlist)
z = spectral_intensity
absMax = np.abs(z).max()

cmap = 'seismic'
fig, ax = plt.subplots(figsize=(4.5, 4))
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap=cmap, vmin=-absMax, vmax=absMax,rasterized=True)
cbar = fig.colorbar(cf,ax=ax, pad=0)

ax.set_ylabel(r'Time, $\gamma t$')
ax.set_xlabel(r'$(\omega-\omega_p)/\gamma$')
cbar.set_label(r'$I(\omega,t)\ $[A.u.]',labelpad=0)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
cbar.ax.yaxis.set_major_formatter(formatter)

cbar.ax.tick_params(width=0.75)
plt.xlim([-20,20])
plt.ylim([0,4])
plt.show()


""" Example: Spectrum """
X,Y = np.meshgrid(w_list_spectrum, correlation_tlist)
z = time_dependent_spectrum
absMax = np.abs(z).max()
fig, ax = plt.subplots(figsize=(4.5, 4))
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap=cmap, vmin=-absMax, vmax=absMax,rasterized=True)
cbar = fig.colorbar(cf,ax=ax, pad=0)

ax.set_ylabel(r'Time, $\gamma t$')
ax.set_xlabel(r'$(\omega-\omega_p)/\gamma$')
cbar.set_label(r'$S(\omega,t)\ [\gamma^{-1}]$',labelpad=0)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
cbar.ax.yaxis.set_major_formatter(formatter)

cbar.ax.tick_params(width=0.75)
plt.xlim([-20,20])
plt.ylim([0,4])
plt.show()

"""Example: Long Time Spectra / Time Integrated Intensity"""

# Integrate the intensity over all times
time_integrated_intensity = np.sum(spectral_intensity, axis=0)*delta_t

fig, ax = plt.subplots(figsize=(6, 4))
plt.plot(w_list_intensity, time_integrated_intensity,linewidth = 3,color = 'orange',linestyle='-',label=r'$I(\omega)$')
plt.plot(w_list_spectrum, time_dependent_spectrum[-1,:],linewidth = 3,color = 'blue',linestyle=':',label=r'$S(\omega)$')
plt.legend(loc='center', bbox_to_anchor=(1,0.5))
plt.xlabel(r'$(\omega-\omega_p)/\gamma$')
plt.ylabel(r'Spectrum [$\gamma^{-1}$]')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim([-10,10])
plt.tight_layout()
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.show()

# %%
