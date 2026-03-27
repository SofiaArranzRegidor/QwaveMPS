#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an example of a single two-level system (TLS)
driven by a classical Rabi field pi pulse from above the waveguide. 

All the examples are in units of the TLS total decay rate, gamma. Hence, in general, gamma=1.

Computes time evolution, population dynamics, steady-state correlations,
and the emission spectrum, with the following example plots:
        1. TLS population dynamics
        2. First and second-order steady-state correlations
        3. Comparison to first and second-order full correlations at two time points.
        4. Long-time emission spectrum

Requirements: 
    
ncon https://pypi.org/project/ncon/. To install it, write the following on your console: 
    
    pip install ncon 
        
"""
#%% Imports and plot functions
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FuncFormatter
import numpy as np

import QwaveMPS as qmps
import time as t


#Parameters for plots style

def pic_style(fontsize):
    rc('font',size=fontsize)

def clean_ticks(x, pos):
    # Only show decimals if not an integer
    return f'{x:g}'

#%% Parameters, Simulations, and single time expectations values

""""Choose the simulation parameters"""

#Choose the bins:
# Dimension chosen to be 2 to as TLS only results in emission in single quanta subspace per unit time
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
    tmax = 10,
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=gamma_l,
    gamma_r = gamma_r,  
    bond_max=4
)

#Make a tlist for plots:
tmax=input_params.tmax
delta_t=input_params.delta_t
tlist=np.arange(0,tmax+(delta_t/2),delta_t)


""" Choose the initial state and coupling"""

sys_initial_state=qmps.states.tls_ground()
wg_initial_state = None


"""Choose the Hamiltonian"""

#Pi pulse from above
pulse_time = tmax
gaussian_center = 1.5
gaussian_width = 0.5
pulsed_pump = np.pi * qmps.states.gaussian_envelope(pulse_time, input_params, gaussian_width, gaussian_center)

# Hamiltonian is 1TLS pumped (from above) by a pi pulse
Hm=qmps.hamiltonians.hamiltonian_1tls(input_params,pulsed_pump)


#To track computational time
start_time=t.time()


"""Calculate time evolution of the system"""
bins = qmps.simulation.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)

"""Define relevant photonic operators"""
tls_pop_op = qmps.tls_pop()
flux_pop_l = qmps.b_dag_l(input_params) @ qmps.b_l(input_params)
flux_pop_r = qmps.b_dag_r(input_params) @ qmps.b_r(input_params)
photon_pop_ops = [flux_pop_l, flux_pop_r]


"""Calculate population dynamics"""
tls_pop = qmps.single_time_expectation(bins.system_states,qmps.tls_pop())
photon_fluxes = qmps.single_time_expectation(bins.output_field_states, photon_pop_ops)

# Integrated fluxes
net_flux_l = np.cumsum(photon_fluxes[0]) * delta_t
net_flux_r = np.cumsum(photon_fluxes[1]) * delta_t

print("--- %s seconds ---" %(t.time() - start_time))

#%%% Population dynamics Graphing

fonts=15
pic_style(fonts)

fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tlist,np.real(tls_pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$') # TLS population
plt.plot(tlist,np.real(net_flux_r),linewidth = 3,color = 'orange',linestyle='-',label=r'$N^{\rm out}_{R}$') # Photon flux transmitted to the right channel
plt.plot(tlist,np.real(net_flux_l),linewidth = 3,color = 'b',linestyle=':',label=r'$N^{\rm out}_{L}$') # Photon flux transmitted to the left channel
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95),labelspacing=0.2,fontsize=fonts)
plt.xlabel(r'Time, $\gamma t$',fontsize=fonts)
plt.ylabel('Populations',fontsize=fonts)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([0.,1.05])
plt.xlim([0.,10])
plt.tight_layout()
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.show()

#%% Two time correlation
"""Calculate two time correlation"""

start_time=t.time()

# Choose operators of which to correlations
a_op_list = []; b_op_list = []; c_op_list = []; d_op_list = []
b_dag_r = qmps.b_dag_r(input_params); b_r = qmps.b_r(input_params)
dim = b_r.shape[0]

# Add op <b_R^\dag(t) b_R(t+t')>
a_op_list.append(b_dag_r)
b_op_list.append(b_r)
c_op_list.append(np.eye(dim))
d_op_list.append(np.eye(dim))


# Add op <b_R^\dag(t) b_R^\dag(t+tau) b_R(t+tau) b_R(t)>
a_op_list.append(b_dag_r)
b_op_list.append(b_dag_r)
c_op_list.append(b_r)
d_op_list.append(b_r)

# Calculate the correlation
correlations, correlation_tlist = qmps.correlation_4op_2t(bins.correlation_bins,
                                    a_op_list, b_op_list, c_op_list, d_op_list, input_params)

print("Correlation time --- %s seconds ---" %(t.time() - start_time))

#%%% Plot the correlation results
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

"""Example graphing G1_{RR}"""
X,Y = np.meshgrid(correlation_tlist,correlation_tlist)

# Use a function to transform from t,t' coordinates to t1, t2 so that t2=t+t'
z = np.real(correlations[0])
abs_max = np.abs(z).max()

# Just take top half of the seismic cmap, only have positive values
base_cmap = colormaps.get_cmap('seismic')
cmap = LinearSegmentedColormap.from_list('seismic_half',base_cmap(np.linspace(0.5, 1.0, 256)))

fig, ax = plt.subplots(figsize=(4.5, 4))
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap=cmap, vmin=0, vmax=abs_max,rasterized=True)
cbar = fig.colorbar(cf,ax=ax, pad=0)

ax.set_ylabel(r'Time, $\gamma t$')
ax.set_xlabel(r'Time, $\gamma t^\prime$')
ax.set_xlim([0,6])
ax.set_ylim([0,6])

cbar.set_label(r'$G^{(1)}_{RR}(t,t^\prime)\ [\gamma]$',labelpad=0)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
cbar.ax.yaxis.set_major_formatter(formatter)

cbar.ax.tick_params(width=0.75)
plt.show()

"""Example graphing G1_{RR}"""
X,Y = np.meshgrid(correlation_tlist,correlation_tlist)

# Use a function to transform from t,t' coordinates to t1, t2 so that t2=t+t'
z = np.real(qmps.transform_t_tau_to_t1_t2(correlations[1]))
abs_max = np.abs(z).max()

fig, ax = plt.subplots(figsize=(4.5, 4))
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap=cmap, vmin=0, vmax=abs_max,rasterized=True)
cbar = fig.colorbar(cf,ax=ax, pad=0)

ax.set_ylabel(r'Time, $\gamma t$')
ax.set_xlabel(r'Time, $\gamma(t+t^\prime)$')
ax.set_xlim([0,6])
ax.set_ylim([0,6])

cbar.set_label(r'$G^{(2)}_{RR}(t,t^\prime)\ [\gamma^{2}]$',labelpad=0)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
cbar.ax.yaxis.set_major_formatter(formatter)

cbar.ax.tick_params(width=0.75)
plt.show()
