#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a basic example to check that the code is working.

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


"Choose max bond dimension"

bond=8


""" Choose the initial state and coupling"""

i_s0=QM.initial_state.i_se()
i_n0=QM.initial_state.i_ng()

gammaL,gammaR=QM.coupling()


"""Choose the Hamiltonian"""

Hm=QM.Hamiltonian_1TLS(Deltat, gammaL, gammaR)


"""Calculate time evolution of the system"""

sys_b,time_b = QM.t_evol_M(Hm,i_s0,i_n0,Deltat,tmax,bond)


"""Calculate population dynamics"""

pop,tbinsR,tbinsL,trans,ref,total=QM.pop_dynamics(sys_b,time_b,Deltat)


#%%

plt.figure(figsize=(4.5,4))
plt.plot(tlist,np.real(trans),linewidth = 3,color = 'orange',linestyle='-',label='Transmission')
plt.plot(tlist,np.real(ref),linewidth = 3,color = 'b',linestyle=':',label='Reflection')
plt.plot(tlist,np.real(total),linewidth = 3,color = 'g',linestyle='-',label='Total')
plt.plot(tlist,np.real(pop),linewidth = 3, color = 'k',linestyle='-',label='TLS pop')
# plt.plot(tlist,np.real(tbinsR)/Deltat,linewidth = 2,color = 'r',linestyle='--',label=r'$n_R/dt$')
plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95),labelspacing=0.2)
plt.xlabel('Time, $\gamma t$')
plt.ylabel('Population')
plt.ylim([0.,1.05])
plt.xlim([0.,tmax])
plt.tight_layout()
plt.show()