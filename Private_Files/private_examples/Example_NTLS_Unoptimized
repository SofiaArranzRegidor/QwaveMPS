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

#Parameters for plots style

def pic_style(fontsize):
    rc('font',size=fontsize)

def clean_ticks(x, pos):
    # Only show decimals if not an integer
    return f'{x:g}'


#%%

"""Choose the time step and end time"""

N=4

d_sys1=2 # first tls bin dimension 
d_sys_total=np.array([d_sys1]*N) #total system bin dimension

d_t_l=3 #Time right channel bin dimension
d_t_r=3 #Time left channel bin dimension
d_t_total=np.array([d_t_r])
d_t = np.prod(d_t_total)

"""Choose the delay time"""


input_params = qmps.parameters.InputParams(
    delta_t=0.05, # Time step of the simulation
    tmax = 20, # Maximum simulation time
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=1,
    gamma_r = 1,  
    bond_max=32 # Maximum bond dimension, simulation parameter that adjusts truncation of entanglement information
)
tlist=np.arange(0,input_params.tmax+input_params.delta_t, input_params.delta_t)

tau=1
taus = [tau] * (N-1)

""" Choose the initial state and coupling"""

i_n0 = np.zeros([1,d_t,1],dtype=complex) #initial time bin
#i_s[:,dSys-1,:] = 1. # TLS in |0> state

i_s0 = np.zeros([1,np.prod(d_sys_total),1],dtype=complex) #system bin

# Start with first 2 in chain excited
#i_s0[:,int(2**(len(d_sys_total)-1) + 2**(len(d_sys_total)-2)),:] = 1; #i_s0[:,d_sys1-1,:] = 10e-9 # TLS in |0> state
# Just First one excited
#i_s0[:,int(2**(len(d_sys_total)-1)),:] = 1; #i_s0[:,d_sys1-1,:] = 10e-9 # TLS in |0> state
# All excited
i_s0[:,int(2**(len(d_sys_total))-1),:] = 1; #i_s0[:,d_sys1-1,:] = 10e-9 # TLS in |0> state

i_n0 = None

"""Choose the Hamiltonian"""
hm = qmps.hamN2LSChiral(0, input_params.delta_t, d_t, N)

""" Time evolution of the system"""

#sys_b,time_b,tau_b = qmps.t_evol_nmar_old(hm,i_s0,i_n0,tau,delta_t,tmax,bond,d_sys_total,d_t_total)
bins = qmps.t_evol_nmar(hm,i_s0,i_n0,taus,input_params)

time_b = bins.output_field_states[0] # Bins entering the feedback channel between the TLS's

out_bins = bins.output_field_states[-1]

""" Calculate population dynamics"""

#pop1,pop2,tbins_r,tbins_l,trans,ref,total=qmps.pop_dynamics_2tls(sys_b,time_b,delta_t,d_sys_total,d_t_total,tau_b,tau)

sys_pop_ops = []
for i in range(N):
    op = qmps.sigmaplus() @ qmps.sigmaminus()
    op = np.kron(np.eye(2**(i)), np.kron(op, np.eye(2**(N-i-1))))
    sys_pop_ops.append(op)

flux_op = qmps.b_pop(input_params)

sys_pops = qmps.single_time_expectation(bins.system_states, sys_pop_ops)
out_flux = qmps.single_time_expectation(out_bins, [flux_op])[0]

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
plt.xlim([0.,tmax])
plt.tight_layout()


plt.show()