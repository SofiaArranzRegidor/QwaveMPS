#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This is a basic example of a single TLS decaying into the waveguide. 
It contains the solution for a TLS decaying symmetrically, 
and a chiral case where the TLS is only coupled to the right channel of the waveguide.

Requirements: ncon https://pypi.org/project/ncon/. 
To install it, write the following on your console: 

    pip install ncon 

"""

import matplotlib.pyplot as plt
import numpy as np
import QwaveMPS as QM

#%%

"""Symmetrical Solution"""

#%%


""""Choose the simulation parameters"""

"Choose the time step and end time"

delta_t = 0.05
tmax = 8
tlist=np.arange(0,tmax+delta_t,delta_t)
d_t=4
d_sys=2

"Choose max bond dimension"

bond=8


""" Choose the initial state and coupling"""

i_s0=QM.states.i_se()
i_n0=QM.states.i_ng(d_t)


gamma_l,gamma_r=QM.coupling('symmetrical',gamma=1)


"""Choose the Hamiltonian"""

Hm=QM.hamiltonian_1tls(delta_t, gamma_l, gamma_r)


"""Calculate time evolution of the system"""

sys_b,time_b = QM.t_evol_mar(Hm,i_s0,i_n0,delta_t,tmax,bond,d_sys,d_t)


"""Calculate population dynamics"""

pop,tbinsR,tbinsL,trans,ref,total=QM.pop_dynamics(sys_b,time_b,delta_t)


#%%

plt.figure(figsize=(4.5,4))
plt.plot(tlist,np.real(trans),linewidth = 3,color = 'orange',linestyle='-',label='Transmission')
plt.plot(tlist,np.real(ref),linewidth = 3,color = 'b',linestyle=':',label='Reflection')
plt.plot(tlist,np.real(pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$')
plt.plot(tlist,np.real(total),linewidth = 3,color = 'g',linestyle='-',label='Total')
# plt.plot(tlist,np.real(tbinsR)/delta_t,linewidth = 2,color = 'r',linestyle='--',label=r'$n_R/dt$')
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95),labelspacing=0.2)
plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.ylim([0.,1.05])
plt.xlim([0.,tmax])
plt.tight_layout()
plt.show()


#%%

"""Right chiral Solution"""

#%%


""" Updated coupling"""

gamma_l,gamma_r=QM.coupling('chiral_r',gamma=1)



"""Choose the Hamiltonian"""

Hm=QM.hamiltonian_1tls(delta_t, gamma_l, gamma_r)


"""Calculate time evolution of the system"""

sys_b,time_b = QM.t_evol_mar(Hm,i_s0,i_n0,delta_t,tmax,bond,d_sys,d_t)


"""Calculate population dynamics"""

pop,tbinsR,tbinsL,trans,ref,total=QM.pop_dynamics(sys_b,time_b,delta_t)


#%%

plt.figure(figsize=(4.5,4))
plt.plot(tlist,np.real(trans),linewidth = 3,color = 'orange',linestyle='-',label='Transmission')
plt.plot(tlist,np.real(ref),linewidth = 3,color = 'b',linestyle=':',label='Reflection')
plt.plot(tlist,np.real(pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$')
plt.plot(tlist,np.real(total),linewidth = 3,color = 'g',linestyle='-',label='Total')
# plt.plot(tlist,np.real(tbinsR)/delta_t,linewidth = 2,color = 'r',linestyle='--',label=r'$n_R/dt$')
plt.legend(loc='center right')
plt.xlabel('Time, $\gamma t$')
plt.ylabel('Populations')
plt.ylim([0.,1.05])
plt.xlim([0.,tmax])
plt.tight_layout()
plt.show()

