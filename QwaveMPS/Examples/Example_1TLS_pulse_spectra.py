#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is an example of a single two-level system (TLS)
interacting with a Fock state pulse. 

All the examples are in units of the TLS total decay rate, gamma. Hence, in general, gamma=1.

It covers two case of a 1-photon tophat pulse

Computes the time dependent spectrum and spectral intensity in transmission


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
wg_initial_state = qmps.states.fock_pulse(pulse_env,pulse_time, input_params,photon_num, direction='R')

# Multiple pulses may be appended in the usual list appending way
#wg_initial_state += qmps.states.fock_pulse(pulse_env,pulse_time, input_params,photon_num, direction='L')

"""Choose the Hamiltonian"""

Hm=qmps.hamiltonian_1tls(input_params)

#To track computational time of simulation and correlations
start_time=t.time()

"""Calculate time evolution of the system"""

bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)


"""Calculate G1 in transmission and reflection, though use 4 op to calculate G2 at same time"""
a_ops = []; b_ops = []; c_ops = []; d_ops = []
b_dag_r = qmps.b_dag_r(input_params) ; a_r = qmps.a_r(input_params)
b_dag_l = qmps.b_dag_l(input_params); a_l = qmps.a_l(input_params)
dim = a_r.shape[0]

# Add Right moving (transmission) first
a_ops.append(b_dag_r)
b_ops.append(a_r)
c_ops.append(np.eye(dim))
d_ops.append(np.eye(dim))


# Add left moving correlation operators
a_ops.append(b_dag_l)
b_ops.append(a_l)
c_ops.append(np.eye(dim))
d_ops.append(np.eye(dim))


"""Also calculate the G2"""
# Add Right moving (transmission) first
a_ops.append(b_dag_r)
b_ops.append(b_dag_r)
c_ops.append(a_r)
d_ops.append(a_r)


# Add left moving correlation operators
a_ops.append(b_dag_l)
b_ops.append(b_dag_l)
c_ops.append(a_l)
d_ops.append(a_l)

"""Consider also a kind of two time squeezing operators in transmission"""
X = b_dag_r + a_r
a_ops.append(X)
b_ops.append(X)
c_ops.append(np.eye(dim))
d_ops.append(np.eye(dim))


correlations, correlation_tlist = qmps.correlation_4op_2t(bins.correlation_bins, a_ops, b_ops, c_ops, d_ops, input_params)

print("--- %s seconds ---" %(t.time() - start_time))
#%%

#To track computational time of spectra and spectral intensity
index = 0
padding_factor = 5
padding = correlations[index].shape[0]*padding_factor
start_time=t.time()

spectral_intensity, w_list_intensity = qmps.spectral_intensity(correlations[index], input_params, padding=padding)
print("Spectral intensity--- %s seconds ---" %(t.time() - start_time))

start_time=t.time()
time_dependent_spectrum, w_list_spectrum = qmps.time_dependent_spectrum(correlations[index], input_params, padding=padding)
print("Time dependent spectra--- %s seconds ---" %(t.time() - start_time))



"""Graph an example"""
import cmasher as cmr
fonts=15
pic_style(fonts)

"""Example: Spectral Intensity"""

X,Y = np.meshgrid(w_list_intensity,correlation_tlist)
z = spectral_intensity
absMax = np.abs(z).max()

cmap = cmr.get_sub_cmap('seismic', 0, 1)
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

# Plot the long time spectra / Time Integrated intensity

# Integrate the intensity over all times
time_integrated_intensity = np.sum(spectral_intensity, axis=0)*delta_t

fig, ax = plt.subplots(figsize=(6, 4))
plt.plot(w_list_intensity, time_integrated_intensity,linewidth = 3,color = 'orange',linestyle='-',label=r'$I(\omega)$')
plt.plot(w_list_spectrum, time_dependent_spectrum[-1,:],linewidth = 3,color = 'blue',linestyle=':',label=r'$S(\omega)$')
plt.legend(loc='center', bbox_to_anchor=(1,0.5))
plt.xlabel(r'$(\omega-\omega_p)/\gamma$')
plt.ylabel('Spectrum')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim([-10,10])
plt.tight_layout()
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.show()
