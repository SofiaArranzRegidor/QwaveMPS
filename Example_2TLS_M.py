#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 15:51:31 2025

@author: sofia
"""

import matplotlib.pyplot as plt
import numpy as np
import QwaveMPS as QM

#%%

"""Choose the simulation parameters"""

"Choose the time step and end time"

delta_t = 0.05
tmax = 8
tlist=np.arange(0,tmax+delta_t,delta_t)
d_t=4
d_sys=4

"Choose max bond dimension"

bond=8


""" Choose the initial state and coupling"""

i_s01=QM.states.i_se()
i_s02= QM.states.i_sg()

# i_s0=1/np.sqrt(2)*(np.kron(i_s01,i_s02)+np.kron(i_s02,i_s01))


i_s0=np.kron(i_s01,i_s02)

i_n0=QM.states.i_ng(d_t)



gamma_l1,gamma_r1=QM.coupling('symmetrical',gamma=1)
gamma_l2,gamma_r2=QM.coupling('symmetrical',gamma=1)


phase=np.pi

"""Choose the Hamiltonian"""

Hm=QM.hamiltonian_2TLS_m(delta_t, gamma_l1, gamma_r1, gamma_l2, gamma_r2,phase,d_t,d_sys)


"""Calculate time evolution of the system"""

sys_b,time_b = QM.t_evol_m(Hm,i_s0,i_n0,delta_t,tmax,bond,d_sys,d_t)


"""Calculate population dynamics"""

pop1,pop2,tbinsR,tbinsL,trans,ref,total=QM.pop_dynamics_2TLS(sys_b,time_b,delta_t)


#%%

plt.figure(figsize=(4.5,4))
plt.plot(tlist,np.real(pop1),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{\rm TLS1}$')
plt.plot(tlist,np.real(pop2),linewidth = 3, color = 'skyblue',linestyle='--',label=r'$n_{\rm TLS2}$')
# plt.plot(tlist,np.real(tbinsR)/delta_t,linewidth = 3,color = 'r',linestyle='-',label=r'$n_R/dt$')
plt.plot(tlist,np.real(trans),linewidth = 3,color = 'orange',linestyle='-',label='T')
plt.plot(tlist,np.real(ref),linewidth = 3,color = 'b',linestyle=':',label='R')
plt.plot(tlist,total,linewidth = 3,color = 'g',linestyle='-',label='Total')
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95),labelspacing=0.2)
plt.xlabel('Time, t$\gamma$')
plt.ylim([0.,1.05])
plt.xlim([0.,tmax])
plt.tight_layout()
plt.show()