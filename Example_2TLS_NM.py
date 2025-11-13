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

delta_t = 0.05
tmax = 5
tlist=np.arange(0,tmax+delta_t,delta_t)

d_sys1=2 # first tls bin dimension 
d_sys2=2 # second tls bin dimension
d_sys_total=np.array([d_sys1, d_sys2]) #total system bin dimension

d_t_l=2 #Time right channel bin dimension
d_t_r=2 #Time left channel bin dimension
d_t_total=np.array([d_t_l,d_t_r])

"""Choose the delay time"""

tau=0.5

""" Choose the initial state and coupling"""

i_s01=qmps.states.i_se()
i_s02= qmps.states.i_sg()
i_s0=np.kron(i_s01,i_s02)
#We can start with one excited and one ground, both excited, both ground, 
# or with an entangled state like the following one
# i_s0=1/np.sqrt(2)*(np.kron(i_s01,i_s02)+np.kron(i_s02,i_s01))



i_n01=qmps.states.i_ng(d_t_l)
i_n02=qmps.states.i_ng(d_t_r)
i_n0=np.kron(i_n01,i_n02)

#Copuling is symmetric by default
gamma_l1,gamma_r1=qmps.coupling('symmetrical',gamma=1)
gamma_l2,gamma_r2=qmps.coupling('symmetrical',gamma=1)

phase=np.pi

"""Choose the Hamiltonian"""

hm=qmps.hamiltonian_2tls_nmar(delta_t,gamma_l1,gamma_r1,gamma_l2,gamma_r2,phase,d_sys_total,d_t_total)


""" Choose max bond dimension"""

bond=8


""" Time evolution of the system"""

sys_b,time_b,tau_b = qmps.t_evol_nmar(hm,i_s0,i_n0,tau,delta_t,tmax,bond,d_sys_total,d_t_total)


""" Calculate population dynamics"""

pop1,pop2,tbins_r,tbins_l,trans,ref,total=qmps.pop_dynamics_2tls(sys_b,time_b,delta_t,d_sys_total,d_t_total,tau_b,tau)


#%%

fonts=15
pic_style(fonts)


fig, ax = plt.subplots(figsize=(4.5, 4))
plt.plot(tlist,np.real(pop1),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{\rm TLS1}$')
plt.plot(tlist,np.real(pop2),linewidth = 3, color = 'skyblue',linestyle='--',label=r'$n_{\rm TLS2}$')
plt.plot(tlist,np.real(trans),linewidth = 3,color = 'orange',linestyle='-',label='T')
plt.plot(tlist,np.real(ref),linewidth = 3,color = 'b',linestyle=':',label='R')
plt.plot(tlist,total,linewidth = 3,color = 'g',linestyle='-',label='Total')
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