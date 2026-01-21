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


#%% 1 photon Tophat Pulse


""""Choose the simulation parameters"""

#Choose the bin dimensions
# Here setting to 3 to accommodate a 2 photon space:
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
    bond_max=4
)

#Make a tlist for plots:
tmax=input_params.tmax
delta_t=input_params.delta_t
tlist=np.arange(0,tmax+delta_t,delta_t)


""" Choose the initial state and pulse parameters"""

# Pulse parameters por a 1-photon tophat pulse
pulse_time = 1 #length of the pulse in time units of gamma
photon_num = 1 #number of photons

sys_initial_state=qmps.states.tls_ground()

#pulse envelope shape
pulse_env=qmps.states.tophat_envelope(pulse_time, input_params)

# Create the pulse envelope
wg_initial_state = qmps.states.fock_pulse(pulse_env,pulse_time, input_params,photon_num, direction='R')

# Multiple pulses may be appended in the usual list appending way
#wg_initial_state += qmps.states.fock_pulse(pulse_env,pulse_time, input_params,photon_num, direction='L')

"""Choose the Hamiltonian"""

Hm=qmps.hamiltonian_1tls(input_params)

#To track computational time of populations
start_time=t.time()

"""Calculate time evolution of the system"""

bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)


"""Calculate population dynamics"""
# Photonic operators
left_flux_op = qmps.a_dag_l(delta_t, d_t_total) @ qmps.a_l(delta_t, d_t_total)
right_flux_op = qmps.a_dag_r(delta_t, d_t_total) @ qmps.a_r(delta_t, d_t_total)
photon_flux_ops = [left_flux_op, right_flux_op]

tls_pop = qmps.single_time_expectation(bins.system_states, qmps.tls_pop())
photon_fluxes = qmps.single_time_expectation(bins.output_field_states, photon_flux_ops)
flux_in = qmps.single_time_expectation(bins.input_field_states, photon_flux_ops)

# Calculate total quanta that has entered the system, tls population + net flux out
total_quanta = tls_pop + np.cumsum(photon_fluxes[0] + photon_fluxes[1]) * delta_t
print("--- %s seconds ---" %(t.time() - start_time))

#%%
fonts=15
pic_style(fonts)


fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tlist,np.real(photon_fluxes[1]),linewidth = 3,color = 'orange',linestyle='-',label='Transmission') # Photons transmitted to the right channel
plt.plot(tlist,np.real(photon_fluxes[0]),linewidth = 3,color = 'b',linestyle=':',label='Reflection') # Photons reflected to the left channel
plt.plot(tlist,np.real(tls_pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$') # TLS population
plt.plot(tlist,np.real(flux_in[0]),linewidth = 3, color = 'skyblue',linestyle='--',label=r'$n_{\rm in,L}$') # Photon flux in from right
plt.plot(tlist,np.real(flux_in[1]),linewidth = 3, color = 'magenta',linestyle='--',label=r'$n_{\rm in,R}$') # Photon flux in from left
plt.plot(tlist,np.real(total_quanta),linewidth = 3,color = 'g',linestyle='-',label='Total') # Conservation check (for one excitation it should be 1)
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

#%%
"""Calculate correlations 
(both could be calculated in same call for faster performance with use of identity operators)"""

#To track computational time of g1
start_time=t.time()

# Construct list of ops with the structure <A(t)B(t+tau)>
# Much faster to calculate using a list and a single correlation_2op_2t() function call then
# Three separate calls
a_op_list = []; b_op_list = []
a_dag_l = qmps.a_dag_l(delta_t, d_t_total); a_l = qmps.a_l(delta_t, d_t_total)
a_dag_r = qmps.a_dag_r(delta_t, d_t_total); a_r = qmps.a_r(delta_t, d_t_total)

# Add op <a_R^\dag(t) a_R(t+tau)>
a_op_list.append(a_dag_r)
b_op_list.append(a_r)

# Add op <a_L^\dag(t) a_L(t+tau)>
a_op_list.append(a_dag_l)
b_op_list.append(a_l)

# Add op <a_L^\dag(t) a_R(t+tau)>
a_op_list.append(a_dag_l)
b_op_list.append(a_r)


g1_correlations, correlation_tlist = qmps.correlation_2op_2t(bins.correlation_bins, a_op_list, b_op_list, input_params)

print("G1 correl--- %s seconds ---" %(t.time() - start_time))

#%%% Graph an example in the t,tau plane
import cmasher as cmr

"""Example graphing G1_{RR}"""
X,Y = np.meshgrid(correlation_tlist,correlation_tlist)
z = np.real(g1_correlations[0])
absMax = np.abs(z).max()

cmap = cmr.get_sub_cmap('seismic', 0, 1)
fig, ax = plt.subplots(figsize=(4.5, 4))
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap=cmap, vmin=-absMax, vmax=absMax,rasterized=True)
cbar = fig.colorbar(cf,ax=ax, pad=0)

ax.set_ylabel(r'Time, $\gamma t$')
ax.set_xlabel(r'Time, $\gamma\tau$')

cbar.set_label(r'$G^{(1)}_{RR}(t,\tau)\ [\gamma^{-2}]$',labelpad=0)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
cbar.ax.yaxis.set_major_formatter(formatter)

cbar.ax.tick_params(width=0.75)
plt.show()


""" Example graphing G1_{LL} """
z = np.real(g1_correlations[1])
absMax = np.abs(z).max()
fig, ax = plt.subplots(figsize=(4.5, 4))
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap=cmap, vmin=-absMax, vmax=absMax,rasterized=True)
cbar = fig.colorbar(cf,ax=ax, pad=0)

ax.set_ylabel(r'Time, $\gamma t$')
ax.set_xlabel(r'Time, $\gamma\tau$')

cbar.set_label(r'$G^{(1)}_{RR}(t,\tau)\ [\gamma^{-1}]$',labelpad=0)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
cbar.ax.yaxis.set_major_formatter(formatter)

cbar.ax.tick_params(width=0.75)
plt.show()

#%% 2 photon Gaussian pulse


""" Updated input field and simulation length"""


input_params.tmax=10
tmax=input_params.tmax
tlist=np.arange(0,tmax+delta_t,delta_t)

#We need a higher bond dimension for a 2-photon pulse
input_params.bond_max=8

# Pulse parameters por a 2-photon gaussian pulse
pulse_time = tmax
photon_num = 2
gaussian_center = 4
gaussian_width = 1


sys_initial_state=qmps.states.tls_ground()

pulse_envelope = qmps.states.gaussian_envelope(pulse_time, input_params, gaussian_width, gaussian_center)
wg_initial_state = qmps.states.fock_pulse(pulse_envelope,pulse_time, input_params, photon_num, direction='R')


start_time=t.time()
"""Calculate time evolution of the system"""

bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)


"""Calculate population dynamics"""

tls_pop = qmps.single_time_expectation(bins.system_states, qmps.tls_pop())
photon_fluxes = qmps.single_time_expectation(bins.output_field_states, photon_flux_ops)
flux_in = qmps.single_time_expectation(bins.input_field_states, photon_flux_ops)

total_quanta = tls_pop + np.cumsum(photon_fluxes[0] + photon_fluxes[1]) * delta_t

print("2-photon pop--- %s seconds ---" %(t.time() - start_time))

#%%% Plot single time dynamics

fonts=15
pic_style(fonts)


fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tlist,np.real(photon_fluxes[1]),linewidth = 3,color = 'orange',linestyle='-',label='Transmission') # Photons transmitted to the right channel
plt.plot(tlist,np.real(photon_fluxes[0]),linewidth = 3,color = 'b',linestyle=':',label='Reflection') # Photons reflected to the left channel
plt.plot(tlist,np.real(tls_pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$') # TLS population
plt.plot(tlist,np.real(flux_in[1]),linewidth = 3, color = 'skyblue',linestyle='--',label=r'$n_{\rm in,R}$') # Photon flux in from right
plt.plot(tlist,np.real(total_quanta),linewidth = 3,color = 'g',linestyle='-',label='Total') # Conservation check (for one excitation it should be 1)
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

#%%% G2 Calculation

#To track computational time of g2
start_time=t.time()

# For speed calculating several at once, but could also calculate all at once
a_op_list = []; b_op_list = []; c_op_list = []; d_op_list = []
# Add op <a_R^\dag(t) a_R^\dag(t+tau) a_R^(t+tau) a_R(t)>
a_op_list.append(a_dag_r)
b_op_list.append(a_dag_r)
c_op_list.append(a_r)
d_op_list.append(a_r)

# Add op <a_L^\dag(t) a_L^\dag(t+tau) a_L^(t+tau) a_L(t)>
a_op_list.append(a_dag_l)
b_op_list.append(a_dag_l)
c_op_list.append(a_l)
d_op_list.append(a_l)


# Add op <a_R^\dag(t) a_L^\dag(t+tau) a_L^(t+tau) a_R(t)>
a_op_list.append(a_dag_r)
b_op_list.append(a_dag_l)
c_op_list.append(a_l)
d_op_list.append(a_r)

# Add op <a_L^\dag(t) a_R^\dag(t+tau) a_R^(t+tau) a_L(t)>
a_op_list.append(a_dag_l)
b_op_list.append(a_dag_r)
c_op_list.append(a_r)
d_op_list.append(a_l)



g2_correlations, correlation_tlist = qmps.correlation_4op_2t(bins.correlation_bins, a_op_list, b_op_list, c_op_list, d_op_list, input_params)


print("G2 correl--- %s seconds ---" %(t.time() - start_time))

#%%% Plot an example
import cmasher as cmr
import matplotlib.ticker as ticker

"""Example graphing G2_{RR}"""
# G2_{RR} is symmetric over t1,t2, so symmeterize for plotting w.r.t. t1,t2
def symmeterize_data(data):
    transformedData = np.zeros(data.shape)
    t_size, tau_size = data.shape # shape is square    
    # Create broadcasted index arrays
    i,j = np.triu_indices(t_size)
    # Compute destination indices: (t, t + tau)
    transformedData[i,j] = data[i, j-i]
    # Fill in the other side of the data
    transformedData = transformedData + transformedData.T - np.diag(np.diag(transformedData))
    return transformedData

# Transform G2_{LR} and G2_{RL} to get t1,t2 coordinates (complete for tau<0)
def transform_LR_RL_data(dataRL, dataLR):
    transformedData = np.zeros(dataLR.shape, dtype=complex)
    t_size, tau_size = dataLR.shape # Shape is square
    
    # Add contributions from both t>= tau and t<= tau (diagonal is equal)
    i, j = np.triu_indices(t_size)
    transformedData[i, j] = dataLR[i, j - i]
    transformedData[j, i] = dataRL[i, j - i]

    return transformedData


X,Y = np.meshgrid(correlation_tlist,correlation_tlist)
z = np.real(symmeterize_data(g2_correlations[0]))
absMax = np.abs(z).max()

cmap = cmr.get_sub_cmap('seismic', 0.5, 1)
fig, ax = plt.subplots(figsize=(4.5, 4))
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap=cmap, vmin=0, vmax=absMax,rasterized=True)
cbar = fig.colorbar(cf,ax=ax, pad=0)

ax.set_ylabel(r'Time, $\gamma t$')
ax.set_xlabel(r'Time, $\gamma(t+\tau)$')

cbar.set_label(r'$G^{(2)}_{RR}(t,\tau)\ [\gamma^{-2}]$',labelpad=0)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
cbar.ax.yaxis.set_major_formatter(formatter)

cbar.ax.tick_params(width=0.75)
plt.show()


z = np.real(symmeterize_data(g2_correlations[1]))
absMax = np.abs(z).max()

cmap = cmr.get_sub_cmap('seismic', 0.5, 1)
fig, ax = plt.subplots(figsize=(4.5, 4))
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap=cmap, vmin=0, vmax=absMax,rasterized=True)
cbar = fig.colorbar(cf,ax=ax, pad=0)

ax.set_ylabel(r'Time, $\gamma t$')
ax.set_xlabel(r'Time, $\gamma(t+\tau)$')

cbar.set_label(r'$G^{(2)}_{LL}(t,\tau)\ [\gamma^{-2}]$',labelpad=0)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
cbar.ax.yaxis.set_major_formatter(formatter)

cbar.ax.tick_params(width=0.75)
plt.show()

z = np.real(transform_LR_RL_data(g2_correlations[2],g2_correlations[3]))
absMax = np.abs(z).max()

cmap = cmr.get_sub_cmap('seismic', 0.5, 1)
fig, ax = plt.subplots(figsize=(4.5, 4))
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap=cmap, vmin=0, vmax=absMax,rasterized=True)
cbar = fig.colorbar(cf,ax=ax, pad=0)

ax.set_ylabel(r'Time, $\gamma t$')
ax.set_xlabel(r'Time, $\gamma(t+\tau)$')

cbar.set_label(r'$G^{(2)}_{LR}(t,\tau)\ [\gamma^{-2}]$',labelpad=0)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
cbar.ax.yaxis.set_major_formatter(formatter)

cbar.ax.tick_params(width=0.75)
plt.show()
# %%
