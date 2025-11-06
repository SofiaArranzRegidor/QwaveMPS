#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 10:12:52 2025

@author: sofia
"""

import matplotlib.pyplot as plt
import numpy as np
import QwaveMPS as QM

#%%

"""Choose the time step and end time"""

delta_t = 0.1
tmax = 5
tlist=np.arange(0,tmax+delta_t,delta_t)
d_sys=2
d_t=2

"""Choose the delay time"""

tau=0.5

""" Choose the initial state and coupling"""

i_s0=QM.states.i_se()
i_n0=QM.states.i_ng(d_t)

#Copuling is symmetric by default
gamma_l,gamma_r=QM.coupling('symmetrical',gamma=1)


phase=np.pi

"""Choose the Hamiltonian"""

Hm=QM.hamiltonian_1TLS_feedback(delta_t,gamma_l,gamma_r,phase,d_t,d_sys)


""" Choose max bond dimension"""

bond=16


""" Time evolution of the system"""

sys_b,time_b,tau_b = QM.t_evol_nm(Hm,i_s0,i_n0,tau,delta_t,tmax,bond,d_t,d_sys)


""" Calculate population dynamics"""

pop,tbins,trans,ph_loop,total=QM.pop_dynamics_1TLS_nm(sys_b,time_b,tau_b,tau,delta_t)


#%%

plt.figure(figsize=(4.5,4))
plt.plot(tlist,np.real(pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{\rm TLS}$')
plt.plot(tlist,np.real(tbins)/delta_t,linewidth = 3,color = 'r',linestyle='-',label=r'$n_R/dt$')
plt.plot(tlist,np.real(trans),linewidth = 3,color = 'orange',linestyle='-',label='T')
plt.plot(tlist,np.real(ph_loop),linewidth = 3,color = 'b',linestyle=':',label='Ph. in loop')
plt.plot(tlist,total,linewidth = 3,color = 'g',linestyle='-',label='Total')
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95),labelspacing=0.2)
# plt.xlabel('Time, t$\gamma$')
# plt.ylim([0.9,1.05])
plt.xlim([0.,tmax])
# plt.xlim([0.,0.5])
plt.tight_layout()
plt.show()