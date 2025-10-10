#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the hamiltonians for different cases

"""

import numpy as np
from .operators import basic_operators

op=basic_operators()


def Hamiltonian_1TLS(Deltat,gammaL,gammaR,d_t=2,d_sys=2,Omega=0,Delta=0):
    '''
    Hamilltonian for 1 TLS in the waveguide
    Deltat is the timestep
    gammaL and gammaR are the left and right decay rates 
    d_sys is the TLS bin dimension (2 by default)
    d_t is the time bin dimension (2 by default)
    Omega is a possible classical pump (turned off by default)
    Delta is the detuning between the pump and TLS  (turned off by default)
    '''
    Msys=Omega*Deltat*(np.kron(op.sigmaplus(),np.eye(d_t*d_t)) + np.kron(op.sigmaminus(),np.eye(d_t*d_t))) +Deltat*Delta*np.kron(op.e(),np.eye(d_t*d_t)) 
    t1= np.sqrt(gammaL)*(np.kron(op.sigmaplus(),op.DeltaBL(Deltat)) + np.kron(op.sigmaminus(),op.DeltaBdagL(Deltat))) 
    t2= np.sqrt(gammaR)*(np.kron(op.sigmaplus(),op.DeltaBR(Deltat)) + np.kron(op.sigmaminus(),op.DeltaBdagR(Deltat))) 
    M=1j*(Msys+t1+t2)#.reshape(d_sys,d_t*d_t,d_sys,d_t*d_t)
    return M

    
def Hamiltonian_1TLS_feedback(Deltat,gammaL,gammaR,phase,d_t,d_sys,Omega=0,Delta=0):
    '''
    Hamilltonian for 1 TLS in a semi-infinite waveguide with a side mirror
    Deltat is the timestep
    gammaL and gammaR are the left and right decay rates 
    phase is the feedback phase between the TLS and the mirror
    d_sys is the TLS bin dimension (2 by default)
    d_t is the time bin dimension (2 by default)
    Omega is a possible classical pump (turned off by default)
    Delta is the detuning between the pump and TLS  (turned off by default)
    '''
    Msys=Omega*Deltat*(np.kron(np.kron(np.eye(d_t),op.sigmaplus()),np.eye(d_t)) +np.kron(np.kron(np.eye(d_t),op.sigmaminus()),np.eye(d_t)))
    t1=np.sqrt(gammaL)*np.kron(np.kron(op.DeltaB(Deltat)*np.exp(-1j*phase),op.sigmaplus()),np.eye(d_t))
    t2=np.sqrt(gammaR)*np.kron(np.kron(np.eye(d_t),op.sigmaplus()),op.DeltaB(Deltat))
    t3=np.sqrt(gammaL)*np.kron(np.kron(op.DeltaBdag(Deltat)*np.exp(1j*phase),op.sigmaminus()),np.eye(d_t))
    t4=np.sqrt(gammaR)*np.kron(np.kron(np.eye(d_t),op.sigmaminus()),op.DeltaBdag(Deltat))   
    M =  1j*(Msys + t1 + t2 + t3 + t4)
    return M


def Hamiltonian_2TLS(Deltat,gammaL1,gammaR1,gammaL2,gammaR2,phase,d_sys,d_t,Omega1=0,Delta1=0,Omega2=0,Delta2=0):
    '''
    Hamilltonian for 2 TLSs in an infinite waveguide 
    Deltat is the timestep
    gammaL1 and gammaR1 are the left and right decay rates of the firt TLS
    gammaL2 and gammaR2 are the left and right decay rates of the second TLS
    phase is the feedback phase between the TLSs
    d_sys is the TLS bin dimension (2 by default)
    d_t is the time bin dimension (2 by default)
    Omega is a possible classical pump (turned off by default)
    Delta is the detuning between the pump and TLS  (turned off by default)
    '''
    d_sys1=int(d_sys/2)
    d_sys2=int(d_sys/2)
    sigmaplus1=np.kron(op.sigmaplus(),np.eye(d_sys2))
    sigmaminus1=np.kron(op.sigmaminus(),np.eye(d_sys2))
    sigmaplus2=np.kron(np.eye(d_sys1),op.sigmaplus())
    sigmaminus2=np.kron(np.eye(d_sys1),op.sigmaminus())
    e1=np.kron(op.e(),np.eye(d_sys2))    
    e2=np.kron(np.eye(d_sys1),op.e())   
    #TLS1 system term
    Msys1=1j*Deltat*Omega1*(np.kron(np.kron(np.eye(d_t),sigmaplus1),np.eye(d_t)) + np.kron(np.kron(np.eye(d_t),sigmaminus1),np.eye(d_t)))
    +1j*Deltat*Delta1*np.kron(np.kron(np.eye(d_t),e1),np.eye(d_t)) 
 
    #TLS2 system term  
    Msys2=1j*Deltat*Omega2*(np.kron(np.kron(np.eye(d_t),sigmaplus2),np.eye(d_t)) + np.kron(np.kron(np.eye(d_t),sigmaminus2),np.eye(d_t)))
    +1j*Deltat*Delta2* np.kron(np.kron(np.eye(d_t),e2),np.eye(d_t)) 
 
    #interaction terms
    t11 = np.sqrt(gammaL2)*np.kron(np.kron(np.eye(d_t),sigmaminus2),op.DeltaBdagL(Deltat))
    t11hc = -np.sqrt(gammaL2)*np.kron(np.kron(np.eye(d_t),sigmaplus2),op.DeltaBL(Deltat))
    t21 = np.sqrt(gammaR2)*np.kron(np.kron(op.DeltaBdagR(Deltat)*np.exp(1j*phase),sigmaminus2),np.eye(d_t))
    t21hc = -np.sqrt(gammaR2)*np.kron(np.kron(op.DeltaBR(Deltat)*np.exp(-1j*phase),sigmaplus2),np.eye(d_t))
    t12 = np.sqrt(gammaL1)*np.kron(np.kron(op.DeltaBdagL(Deltat)*np.exp(1j*phase),sigmaminus1),np.eye(d_t))
    t12hc = -np.sqrt(gammaL1)*np.kron(np.kron(op.DeltaBL(Deltat)*np.exp(-1j*phase),sigmaplus1),np.eye(d_t))
    t22 = np.sqrt(gammaR1)*np.kron(np.kron(np.eye(d_t),sigmaminus1),op.DeltaBdagR(Deltat))
    t22hc = -np.sqrt(gammaR1)*np.kron(np.kron(np.eye(d_t),sigmaplus1),op.DeltaBR(Deltat))
 
    M = (Msys1 + Msys2 + t11 + t11hc + t21 + t21hc + t12 + t12hc + t22 + t22hc)
    return M

