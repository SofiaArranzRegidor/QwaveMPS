#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 11:33:34 2025

@author: sofia
"""

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


#%%

"""Choose the time step and end time"""
N=8
d_sys1=2 # first tls bin dimension 
d_sys_total=np.array([d_sys1]*N) #total system bin dimension

d_t_l=3 #Time right channel bin dimension
d_t_r=3 #Time left channel bin dimension
d_t_total=np.array([d_t_r])
d_t = np.prod(d_t_total)
input_params = qmps.parameters.InputParams(
    delta_t=0.05, # Time step of the simulation
    tmax = 40,#30, # Maximum simulation time
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=1,
    gamma_r = 1,  
    bond_max=32 # Maximum bond dimension, simulation parameter that adjusts truncation of entanglement information
)
tlist=np.arange(0,input_params.tmax+input_params.delta_t, input_params.delta_t)


"""Choose the delay time"""

tau=1
#taus = [tau] * (N-1)
taus = [1,0.5,1,0.5,1,0.5,1]

""" Choose the initial state and coupling"""
i_s = np.zeros([1,d_sys1,1],dtype=complex) #system bin
i_n0 = np.zeros([1,d_t,1],dtype=complex) #initial time bin
#i_s[:,dSys-1,:] = 1. # TLS in |0> state

i_s0 = np.zeros([1,np.prod(d_sys_total),1],dtype=complex) #system bin

# Start with first 2 in chain excited
#i_s0[:,int(2**(len(d_sys_total)-1) + 2**(len(d_sys_total)-2)),:] = 1; #i_s0[:,d_sys1-1,:] = 10e-9 # TLS in |0> state
# Just First one excited
#i_s0[:,int(2**(len(d_sys_total)-1)),:] = 1; #i_s0[:,d_sys1-1,:] = 10e-9 # TLS in |0> state
# All excited
i_s0[:,int(2**(len(d_sys_total))-1),:] = 1; #i_s0[:,d_sys1-1,:] = 10e-9 # TLS in |0> state



#We can start with one excited and one ground, both excited, both ground, 
# or with an entangled state like the following one
# i_s0=1/np.sqrt(2)*(np.kron(i_s01,i_s02)+np.kron(i_s02,i_s01))
i_n0 = qmps.fock_pulse(qmps.tophat_envelope(2, input_params), 2, 1, input_params, 'R')

"""Choose the Hamiltonian"""

#hm = qmps.hamN2LSChiral(0, delta_t, d_t, N)
hams = []
for i in range(len(d_sys_total)):
    hm = qmps.hamiltonian_1tls_chiral(input_params)
    hams.append(hm)


""" Time evolution of the system"""
bins = qmps.t_evol_nmar_chiral(hams,i_s0,i_n0,taus,input_params)

#%%
time_b = bins.output_field_states[0] # Bins entering the feedback channel between the TLS's

out_bins = bins.output_field_states[-1]

""" Calculate population dynamics"""

#pop1,pop2,tbins_r,tbins_l,trans,ref,total=qmps.pop_dynamics_2tls(sys_b,time_b,delta_t,d_sys_total,d_t_total,tau_b,tau)

sys_pop_op = qmps.sigmaplus() @ qmps.sigmaminus()
flux_op = qmps.b_pop(input_params)

sys_pops = []
for i in range(len(d_sys_total)):
    sys_pops.append(qmps.single_time_expectation(bins.system_states[i], sys_pop_op))
out_flux = qmps.single_time_expectation(out_bins, flux_op)

#%%

fonts=15
pic_style(fonts)


#fig, ax = plt.subplots(figsize=(4.5, 4))
fig, ax = plt.subplots(figsize=(8, 6))

for i in range(N):
    plt.plot(tlist,np.real(sys_pops[i]),linewidth = 3,linestyle='-',label=r'$n_{\rm TLS}^{('+str(i)+r')}$')

plt.plot(tlist,np.real(out_flux),linewidth = 3, color = 'skyblue',linestyle='--',label=r'$n_{\rm out}$')
#plt.plot(tlist,np.real(trans),linewidth = 3,color = 'orange',linestyle='-',label='T')
#plt.plot(tlist,np.real(ref),linewidth = 3,color = 'b',linestyle=':',label='R')
#plt.plot(tlist,total,linewidth = 3,color = 'g',linestyle='-',label='Total')
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95),labelspacing=0.2)
plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.grid(True, linestyle='--', alpha=0.6)
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.ylim([0.,1.05])
plt.xlim([0.,5*N])
plt.tight_layout()


plt.show()

#%%% G2 Calculation

#To track computational time of G2
start_time=t.time()

# Have to create operators again for this larger space
b_dag = qmps.b_dag(input_params); b = qmps.b(input_params)
ops_same_time = []; ops_two_time = []

# Add op <b_R^\dag(t) b_R^\dag(t+t') b_R^(t+t') b_R(t)> 
ops_same_time.append(b_dag @ b_dag @ b @ b)
ops_two_time.append(np.kron(b_dag@b, b_dag@b))

# Could also consider G1 correlation function (in both orders)
# For example: <b_R^\dag(t)b_R(t+t')> 
ops_same_time.append(b_dag @ b); ops_two_time.append(np.kron(b_dag, b))
ops_same_time.append(b_dag @ b); ops_two_time.append(np.kron(b, b_dag))

correlations, correlation_tlist = qmps.correlations_2t(bins.correlation_bins, ops_same_time, ops_two_time, input_params, True)


print("G2 correl--- %s seconds ---" %(t.time() - start_time))

#%%% Plot an example
import matplotlib.ticker as ticker
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

"""Example graphing G2_{RR}"""
X,Y = np.meshgrid(correlation_tlist,correlation_tlist)
xmax = N*3

# Use a function to transform from t,t' coordinates to t1, t2 so that t2=t+t'
z = np.real(qmps.transform_t_tau_to_t1_t2(correlations[0]))
absMax = np.abs(z).max()

# Just take top half of the seismic cmap, only have positive values
base_cmap = colormaps.get_cmap('seismic')
cmap = LinearSegmentedColormap.from_list('seismic_half',base_cmap(np.linspace(0.5, 1.0, 256)))

fig, ax = plt.subplots(figsize=(4.5, 4))
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap=cmap, vmin=0, vmax=absMax,rasterized=True)
cbar = fig.colorbar(cf,ax=ax, pad=0)

ax.set_ylabel(r'Time, $\gamma t$')
ax.set_xlabel(r'Time, $\gamma(t+t^\prime)$')
ax.set_xlim([0,xmax])
ax.set_ylim([0,xmax])


cbar.set_label(r'$G^{(2)}_{TT}(t,t^\prime)\ [\gamma^{2}]$',labelpad=0)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
cbar.ax.yaxis.set_major_formatter(formatter)

cbar.ax.tick_params(width=0.75)
plt.show()


"""Example graphing G1"""
z = np.real(qmps.transform_t_tau_to_t1_t2(correlations[1], correlations[2]))
absMax = np.abs(z).max()

fig, ax = plt.subplots(figsize=(4.5, 4))
cmap='seismic'
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap=cmap, vmin=-absMax, vmax=absMax,rasterized=True)
cbar = fig.colorbar(cf,ax=ax, pad=0)

ax.set_ylabel(r'Time, $\gamma t$')
ax.set_xlabel(r'Time, $\gamma(t+t^\prime)$')
ax.set_xlim([0,xmax])
ax.set_ylim([0,xmax])


cbar.set_label(r'$G^{(1)}_{TT}(t,t^\prime)\ [\gamma^{2}]$',labelpad=0)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
cbar.ax.yaxis.set_major_formatter(formatter)

cbar.ax.tick_params(width=0.75)
plt.show()


# %% Plot the spectrum
padding_factor = 5 
index = 1
padding = correlations[index].shape[0]*padding_factor

spectral_intensity, w_list_intensity = qmps.spectral_intensity(correlations[index], input_params, padding=padding)
time_dependent_spectrum, w_list_spectrum = qmps.time_dependent_spectrum(correlations[index], input_params, padding=padding)

#%%%
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
plt.xlim([-5,5])
plt.ylim([0,xmax])
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
plt.xlim([-5,5])
plt.ylim([0,xmax])
plt.show()

"""Example: Long Time Spectra / Time Integrated Intensity"""

# Integrate the intensity over all times
fig, ax = plt.subplots(figsize=(6, 4))
plt.plot(w_list_spectrum, time_dependent_spectrum[-1,:],linewidth = 3,color = 'blue',linestyle=':',label=r'$S(\omega)$')
plt.legend(loc='center', bbox_to_anchor=(1,0.5))
plt.xlabel(r'$(\omega-\omega_p)/\gamma$')
plt.ylabel(r'Spectrum [$\gamma^{-1}$]')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim([-5, 5])
plt.tight_layout()
formatter = FuncFormatter(clean_ticks)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
plt.show()
