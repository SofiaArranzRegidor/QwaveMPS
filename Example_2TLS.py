#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 11:33:34 2025

@author: sofia
"""

import matplotlib.pyplot as plt
import numpy as np
import QwaveMPS as QM

#%%

"""Choose the time step and end time"""

Deltat = 0.02
tmax = 5
tlist=np.arange(0,tmax+Deltat,Deltat)

"""Choose the delay time"""

tau=0.5

""" Choose the initial state and coupling"""

i_s01=QM.initial_state.i_se()
i_s02= QM.initial_state.i_sg()

#We can start with one excited and one ground, both excited, both ground, 
# or with an entangled state like the following one
i_s0=1/np.sqrt(2)*(np.kron(i_s01,i_s02)+np.kron(i_s02,i_s01))
i_n0=QM.initial_state.i_ng()

#Copuling is symmetric by default
gammaL1,gammaR1=QM.coupling()
gammaL2,gammaR2=QM.coupling()

phase=np.pi

"""Choose the Hamiltonian"""

Hm=QM.Hamiltonian_2TLS(Deltat,gammaL1,gammaR1,gammaL2,gammaR2,phase,d_sys=4)


""" Choose max bond dimension"""

bond=8


""" Time evolution of the system"""

sys_b,time_b,tau_b = QM.t_evol_NM(Hm,i_s0,i_n0,tau,Deltat,tmax,bond,d_sys=4)


""" Calculate population dynamics"""

pop1,pop2,tbinsR,tbinsL,trans,ref,total=QM.pop_dynamics_2TLS(sys_b,time_b,tau_b,tau,Deltat)


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