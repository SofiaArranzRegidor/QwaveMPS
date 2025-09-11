#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a basic example to check that the code is working.

"""

import matplotlib.pyplot as plt
import numpy as np
import QwaveMPS as QM

#%%

"""Choose the time step and end time"""

Deltat = 0.05
tmax = 10
tlist=np.arange(0,tmax+Deltat,Deltat)

""" Choose the initial state and coupling"""

i_s0=QM.initial_state.i_se()
i_n0=QM.initial_state.i_ng()

gammaL,gammaR=QM.coupling()


"""Choose the Hamiltonian"""

Hm=QM.Hamiltonian_1TLS(Deltat, gammaL, gammaR)


""" Choose max bond dimension"""

bond=8


""" Time evolution of the system"""

sys_b,time_b = QM.t_evol(Hm,i_s0,i_n0,Deltat,tmax,bond)


""" Calculate population dynamics"""

pop,tbinsR,tbinsL,trans,ref,total=QM.pop_dynamics(sys_b,time_b,Deltat)


#%%

plt.figure()
# plt.plot(tlist,np.real(pop_an),linewidth = 2,color = 'magenta',linestyle='-',label='analytical')
plt.plot(tlist,np.real(pop),linewidth = 2, color = 'k',linestyle='-',label='TLS pop')
plt.plot(tlist,np.real(trans),linewidth = 2,color = 'y',linestyle='-',label='T')
plt.plot(tlist,np.real(ref),linewidth = 2,color = 'magenta',linestyle='--',label='R')
plt.plot(tlist,total,linewidth = 2,color = 'b',linestyle='--',label='Total')
plt.plot(tlist,np.real(tbinsR)/Deltat,linewidth = 2,color = 'r',linestyle='--',label='n_R')
# plt.plot(tlist,gaussian_f(tlist),linewidth = 2,color = 'magenta',linestyle='-',label='pulse')
plt.legend()
plt.xlabel('Time, t$\gamma$')
# plt.ylabel('TLS Population')
# plt.ylim([0.,1.05])
# plt.xlim([0.,10.])
plt.grid()
plt.tight_layout()
# plt.savefig('pulse_5.pdf')
plt.show()