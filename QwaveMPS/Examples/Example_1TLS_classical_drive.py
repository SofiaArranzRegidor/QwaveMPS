#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an example of a single two-level system (TLS)
driven by a classical continuous-wave (CW) field from above the waveguide. 

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

#%% Parameters, Simulations, and single time expectations values

""""Choose the simulation parameters"""

#Choose the bins:
# Dimension chosen to be 3 to check second order photonic observables
d_t_l=3 #Time right channel bin dimension
d_t_r=3 #Time left channel bin dimension
d_t_total=np.array([d_t_l,d_t_r])

d_sys1=2 # tls bin dimension
d_sys_total=np.array([d_sys1]) #total system bin (in this case only 1 tls)

#Choose the coupling:
gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1)

#Define input parameters
input_params = qmps.parameters.InputParams(
    delta_t=0.02,
    tmax = 35, # Long max time to reach steady state
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=gamma_l,
    gamma_r = gamma_r,  
    bond_max=18
)

#Make a tlist for plots:
tmax=input_params.tmax
delta_t=input_params.delta_t
tlist=np.arange(0,tmax+(delta_t/2),delta_t)


""" Choose the initial state and coupling"""

sys_initial_state=qmps.states.tls_excited()
wg_initial_state = None# qmps.states.vacuum(tmax,input_params)


"""Choose the Hamiltonian"""

#CW Drive
cw_pump=2*np.pi

# Hamiltonian is 1TLS pumped (from above) by CW
Hm=qmps.hamiltonians.hamiltonian_1tls(input_params,cw_pump)


#To track computational time
start_time=t.time()


"""Calculate time evolution of the system"""
bins = qmps.simulation.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)

"""Define relevant photonic operators"""
tls_pop_op = qmps.tls_pop()
flux_pop_l = qmps.b_dag_l(input_params) @ qmps.b_l(input_params)
flux_pop_r = qmps.b_dag_r(input_params) @ qmps.b_r(input_params)
g2_same_time_op = qmps.b_dag_r(input_params) @ qmps.b_dag_r(input_params) @ qmps.b_r(input_params) @ qmps.b_r(input_params)
photon_pop_ops = [flux_pop_l, flux_pop_r]


"""Calculate population dynamics"""
tls_pop = qmps.single_time_expectation(bins.system_states,qmps.tls_pop())
photon_fluxes = qmps.single_time_expectation(bins.output_field_states, photon_pop_ops)
g2_same_time = qmps.single_time_expectation(bins.output_field_states, g2_same_time_op)


print("--- %s seconds ---" %(t.time() - start_time))

#%%% Population dynamics Graphing

fonts=15
pic_style(fonts)

fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tlist,np.real(tls_pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$') # TLS population
plt.plot(tlist,np.real(photon_fluxes[1]),linewidth = 3,color = 'orange',linestyle='-',label=r'$n^{\rm out}_{R}$') # Photon flux transmitted to the right channel
plt.plot(tlist,np.real(photon_fluxes[0]),linewidth = 3,color = 'b',linestyle=':',label=r'$n^{\rm out}_{L}$') # Photon flux transmitted to the left channel
plt.plot(tlist,np.real(g2_same_time),linewidth = 3,color = 'cyan',linestyle='--',label=r'$G^{(2)}_{R}$') # G2, same time, right
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

#%% Steady state correlation dynamics
"""Calculate steady state correlations"""

start_time=t.time()

# Choose operators of which to take steady state correlations
a_op_list = []; b_op_list = []
b_dag_l = qmps.b_dag_l(input_params); b_l = qmps.b_l(input_params)
b_dag_r = qmps.b_dag_r(input_params); b_r = qmps.b_r(input_params)

# Add op <b_R^\dag(t) b_R(t+t')>
a_op_list.append(b_dag_r)
b_op_list.append(b_r)

# Add op <b_L^\dag(t) b_L(t+t')>
a_op_list.append(b_dag_l)
b_op_list.append(b_l)

# Add op <b_L^\dag(t) b_R(t+t')>
a_op_list.append(b_dag_l)
b_op_list.append(b_r)

# Calculate the steady state correlation (returns the list of tau dependent correlation lists,
# list of tau values for the correlation, and the initial t values of steady state)
correlations_ss_2op,tau_lists_ss_2op,ts_steady_2op = qmps.correlation_ss_2op(bins.correlation_bins,bins.output_field_states,a_op_list, b_op_list, input_params)


# Test out the case of a single 4op steady state correlation
# Calculate for the op <b_R^\dag(t) b_R^\dag(t+tau) b_R(t+tau) b_R(t)>
correlation_ss_4op, tau_list_ss_4op, t_steady_4op = qmps.correlation_ss_4op(bins.correlation_bins, bins.output_field_states,
                                                                      b_dag_r, b_dag_r, b_r, b_r, input_params)

print("Convergence of single time expectation: ", ts_steady_2op[0], t_steady_4op)
print("SS correlations as two function calls --- %s seconds ---" %(t.time() - start_time))

# Note that optimal coding would use identity matrices so that we would only have to make a single
# function call to calculate the correlations (pad a c_op_list and d_op_list with identities)

# This also ensures that if some operators converge later than others, the later time is used
start_time=t.time()

c_op_list = [np.eye(input_params.d_t)]*3
d_op_list = [np.eye(input_params.d_t)]*3

# Add in the last correlation, <b_R^\dag(t) b_R^\dag(t+tau) b_R(t+tau) b_R(t)>
a_op_list.append(b_dag_r)
b_op_list.append(b_dag_r)
c_op_list.append(b_r)
d_op_list.append(b_r)

correlations_ss, tau_lists_ss, ts_steady = qmps.correlation_ss_4op(bins.correlation_bins, bins.output_field_states,
                                                        a_op_list, b_op_list, c_op_list, d_op_list, input_params)

print("Convergence of single time expectations: ",ts_steady[0], ts_steady[-1])
print("SS correlation as single function call --- %s seconds ---" %(t.time() - start_time))


fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tau_lists_ss,np.real(correlations_ss[0]),linewidth = 3, color = 'darkgreen',linestyle='-',label=r'$G^{(1)}_R$') 
plt.plot(tau_lists_ss,np.real(correlations_ss[-1]),linewidth = 3, color = 'lime',linestyle='-',label=r'$G^{(2)}_R$') 
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95),labelspacing=0.2,fontsize=fonts)
plt.xlabel(r'Time, $\gamma t^\prime$',fontsize=fonts)
plt.ylabel('Correlations',fontsize=fonts)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([0.,None])
plt.xlim([0.,10])
plt.tight_layout()
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.show()

#%% Full single time Correlation functions about the steady state time
"""Test the whole single time correlation function calculation for the previous operators"""
start_time=t.time()

# Determine the full single time correlation at the steady state time (including negative tau)
correlations_1t, tau_list_1t = qmps.correlation_4op_1t(bins.correlation_bins,
                                    a_op_list, b_op_list, c_op_list, d_op_list, ts_steady.max(), input_params)


print("Full single time correlations --- %s seconds ---" %(t.time() - start_time))

#%%% Graphing of these single time dynamics
fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tau_lists_ss,np.real(correlations_ss[0]),linewidth = 3, color = 'darkgreen',linestyle='-',label=r'$g^{(1)}_{R,SS}$') 
plt.plot(tau_lists_ss,np.real(correlations_ss[-1]),linewidth = 3, color = 'lime',linestyle='-',label=r'$g^{(2)}_{R,SS}$') 
plt.plot(tau_list_1t,np.real(correlations_1t[0]),linewidth = 3, color = 'skyblue',linestyle=':',label=r'$g^{(1)}_R$') 
plt.plot(tau_list_1t,np.real(correlations_1t[-1]),linewidth = 3, color = 'orange',linestyle=':',label=r'$g^{(2)}_R$') 

plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95),labelspacing=0.2,fontsize=fonts)
plt.xlabel(r'Time, $\gamma t^\prime$',fontsize=fonts)
plt.ylabel('Correlations',fontsize=fonts)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([0.,None])
plt.xlim([-5,10])
plt.tight_layout()
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.show()


#%% Long time spectrum
"""Calculate the spectrum"""

start_time=t.time()

# Calculate the steady state spectrum of G1_R using the previously calculated steady state result
spect,w_list=qmps.spectrum_w(input_params.delta_t,correlations_ss[0])

print("spectrum --- %s seconds ---" %(t.time() - start_time))

# Plot the spectrum
fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(w_list/cw_pump,np.real(spect)/max(np.real(spect)),linewidth = 4, color = 'purple',linestyle='-') # TLS population
plt.xlabel(r'$(\omega - \omega_L)/\Omega$',fontsize=fonts)
plt.ylabel('Spectrum',fontsize=fonts)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([0.,1.05])
plt.xlim([-3.,3])
plt.tight_layout()
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.show()

