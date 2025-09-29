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
tmax = 8
tlist=np.arange(0,tmax+Deltat,Deltat)

"""Choose the delay time"""

tau=2

""" Choose the initial state and coupling"""

i_s01=QM.initial_state.i_se()
i_s02= QM.initial_state.i_sg()
i_s0=np.kron(i_s01,i_s02)
i_n0=QM.initial_state.i_ng()

gammaL1,gammaR1=QM.coupling()
gammaL2,gammaR2=QM.coupling()

phase=0

"""Choose the Hamiltonian"""

Hm=QM.Hamiltonian_2TLS(Deltat,gammaL1,gammaR1,gammaL2,gammaR2,phase,d_sys=4)


""" Choose max bond dimension"""

bond=8


""" Time evolution of the system"""

sys_b,time_b,tau_b = QM.t_evol_NM(Hm,i_s0,i_n0,tau,Deltat,tmax,bond,d_sys=4)


""" Calculate population dynamics"""

pop1,pop2,tbinsR,tbinsL,trans,ref,total=QM.pop_dynamics_2TLS(sys_b,time_b,tau_b,tau,Deltat)


#%%

plt.figure()
# plt.plot(tlist,np.real(pop_an),linewidth = 2,color = 'magenta',linestyle='-',label='analytical')
plt.plot(tlist,np.real(pop1),linewidth = 2, color = 'k',linestyle='-',label='TLS1 pop')
plt.plot(tlist,np.real(pop2),linewidth = 2, color = 'skyblue',linestyle='-',label='TLS2 pop')
plt.plot(tlist,np.real(trans),linewidth = 2,color = 'y',linestyle='-',label='T')
plt.plot(tlist,np.real(ref),linewidth = 2,color = 'magenta',linestyle='--',label='R')
plt.plot(tlist,total,linewidth = 2,color = 'b',linestyle='--',label='Total')
plt.plot(tlist,np.real(tbinsR)/Deltat,linewidth = 2,color = 'r',linestyle='--',label='n_R')
# plt.plot(tlist,gaussian_f(tlist),linewidth = 2,color = 'magenta',linestyle='-',label='pulse')
plt.legend()
plt.xlabel('Time, t$\gamma$')
# plt.ylabel('TLS Population')
plt.ylim([0.,1.05])
# plt.xlim([0.,10.])
plt.grid()
plt.tight_layout()
# plt.savefig('pulse_5.pdf')
plt.show()