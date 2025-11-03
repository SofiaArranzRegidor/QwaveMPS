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

Deltat = 0.05
tmax = 8
tlist=np.arange(0,tmax+Deltat,Deltat)
d_t=4
d_sys=4

"Choose max bond dimension"

bond=8


""" Choose the initial state and coupling"""

i_s01=QM.states.i_se()
i_s02= QM.states.i_sg()

# i_s0=1/np.sqrt(2)*(np.kron(i_s01,i_s02)+np.kron(i_s02,i_s01))


i_s0=np.kron(i_s01,i_s02)

i_n0=QM.states.input_state_generator(d_t)



gammaL1,gammaR1=QM.coupling()
gammaL2,gammaR2=QM.coupling()


phase=np.pi

"""Choose the Hamiltonian"""

Hm=QM.Hamiltonian_2TLS_M(Deltat, gammaL1, gammaR1, gammaL2, gammaR2,phase,d_t,d_sys)


"""Calculate time evolution of the system"""

sys_b,time_b = QM.t_evol_M(Hm,i_s0,i_n0,Deltat,tmax,bond,d_sys,d_t)


"""Calculate population dynamics"""

pop1,pop2,tbinsR,tbinsL,trans,ref,total=QM.pop_dynamics_2TLS(sys_b,time_b,Deltat)


#%%

plt.figure(figsize=(4.5,4))
plt.plot(tlist,np.real(pop1),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{\rm TLS1}$')
plt.plot(tlist,np.real(pop2),linewidth = 3, color = 'skyblue',linestyle='--',label=r'$n_{\rm TLS2}$')
# plt.plot(tlist,np.real(tbinsR)/Deltat,linewidth = 3,color = 'r',linestyle='-',label=r'$n_R/dt$')
plt.plot(tlist,np.real(trans),linewidth = 3,color = 'orange',linestyle='-',label='T')
plt.plot(tlist,np.real(ref),linewidth = 3,color = 'b',linestyle=':',label='R')
plt.plot(tlist,total,linewidth = 3,color = 'g',linestyle='-',label='Total')
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95),labelspacing=0.2)
plt.xlabel('Time, t$\gamma$')
plt.ylim([0.,1.05])
plt.xlim([0.,tmax])
plt.tight_layout()
plt.show()