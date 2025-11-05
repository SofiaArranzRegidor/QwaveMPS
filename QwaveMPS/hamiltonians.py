#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the hamiltonians for different cases

"""

import numpy as np
from .operators import *

# op=basic_operators()


def hamiltonian_1TLS(delta_t:float, gammaL:float, gammaR:float, d_t:int=2, d_sys:int=2, Omega:float=0, Delta:float=0) -> np.ndarray:
    """
    Hamilltonian for 1 TLS in the waveguide
    
    Parameters
    ----------
    delta_t : float
        Time step for system evolution.

    gammaL : float
        Left decay rate.

    gammaR : float
        Right decay rate.

    d_sys : int, default: 2
        TLS bin dimension

    d_t : int, default: 2, default: 2
        Time bin dimension

    Omega : float, default: 0
        Classical pump

    Delta : float, default: 0
        Detuning between the pump and TLS frequencies

    Returns
    -------
    Hamiltonian : ndarray
        Hamiltonian coupling a single TLS pumped by a classical field to a waveguide.

    Examples
    -------- 
    """
    
    Msys=Omega*delta_t*(np.kron(sigmaplus(),np.eye(d_t*d_t)) + np.kron(sigmaminus(),np.eye(d_t*d_t))) +delta_t*Delta*np.kron(e(),np.eye(d_t*d_t)) 
    t1= np.sqrt(gammaL)*(np.kron(sigmaplus(),delta_bL(delta_t)) + np.kron(sigmaminus(),delta_b_dagL(delta_t))) 
    t2= np.sqrt(gammaR)*(np.kron(sigmaplus(),delta_bR(delta_t)) + np.kron(sigmaminus(),delta_b_dagR(delta_t))) 
    M=Msys+t1+t2
    return M

    
def hamiltonian_1TLS_feedback(delta_t:float, gammaL:float, gammaR:float, phase:float, d_t:int, d_sys:int, Omega:float=0, Delta:float=0) -> np.ndarray:
    """
    Hamilltonian for 1 TLS in a semi-infinite waveguide with a side mirror
    
    Parameters
    ----------
    delta_t : float
        Time step for system evolution.

    gammaL : float
        Left decay rate.

    gammaR : float
        Right decay rate.

    phase : float
        Feedback phase between the TLS and the mirror.

    d_sys : int, default: 2
        TLS bin dimension

    d_t : int, default: 2, default: 2
        Time bin dimension

    Omega : float, default: 0
        Classical pump

    Delta : float, default: 0
        Detuning between the pump and TLS frequencies

    Returns
    -------
    Hamiltonian : ndarray
        Hamiltonian coupling a single TLS pumped by a classical field to the semi-infinite waveguide.

    Examples
    -------- 
    """
    Msys=Omega*delta_t*(np.kron(np.kron(np.eye(d_t),sigmaplus()),np.eye(d_t)) +np.kron(np.kron(np.eye(d_t),sigmaminus()),np.eye(d_t)))
    t1=np.sqrt(gammaL)*np.kron(np.kron(delta_b(delta_t)*np.exp(-1j*phase),sigmaplus()),np.eye(d_t))
    t2=np.sqrt(gammaR)*np.kron(np.kron(np.eye(d_t),sigmaplus()),delta_b(delta_t))
    t3=np.sqrt(gammaL)*np.kron(np.kron(delta_b_dag(delta_t)*np.exp(1j*phase),sigmaminus()),np.eye(d_t))
    t4=np.sqrt(gammaR)*np.kron(np.kron(np.eye(d_t),sigmaminus()),delta_b_dag(delta_t))   
    M = Msys + t1 + t2 + t3 + t4
    return M


def hamiltonian_2TLS_NM(delta_t:float, gammaL1:float, gammaR1:float, gammaL2:float, gammaR2:float, phase:float, d_sys:int, d_t:int, Omega1:float=0, Delta1:float=0, Omega2:float=0, Delta2:float=0) -> np.ndarray:
    """
    Hamilltonian for 2 TLSs in an infinite waveguide.
    
    Parameters
    ----------
    delta_t : float
        Time step for system evolution.

    gammaL1 : float
        Left decay rate of the first TLS.

    gammaR1 : float
        Right decay rate of the first TLS.

    gammaL2 : float
        Left decay rate of the second TLS.

    gammaR2 : float
        Right decay rate of the second TLS.

    phase : float
        Feedback phase between the TLSs.

    d_sys : int, default: 2
        TLS bin dimension

    d_t : int, default: 2, default: 2
        Time bin dimension

    Omega1 : float, default: 0
        Classical pump for the first TLS.

    Delta1 : float, default: 0
        Detuning between the pump and TLS frequencies for the first TLS.
    
    Omega2 : float, default: 0
        Classical pump for the second TLS.

    Delta2 : float, default: 0
        Detuning between the pump and TLS frequencies for the second TLS.

    Returns
    -------
    Hamiltonian : ndarray
        Hamiltonian coupling a two TLSs pumped by a classical fields to an infinite waveguide.

    Examples
    -------- 
    """
    d_sys1=int(d_sys/2)
    d_sys2=int(d_sys/2)
    sigmaplus1=np.kron(sigmaplus(),np.eye(d_sys2))
    sigmaminus1=np.kron(sigmaminus(),np.eye(d_sys2))
    sigmaplus2=np.kron(np.eye(d_sys1),sigmaplus())
    sigmaminus2=np.kron(np.eye(d_sys1),sigmaminus())
    e1=np.kron(e(),np.eye(d_sys2))    
    e2=np.kron(np.eye(d_sys1),e())   
    #TLS1 system term
    Msys1=delta_t*Omega1*(np.kron(np.kron(np.eye(d_t),sigmaplus1),np.eye(d_t)) + np.kron(np.kron(np.eye(d_t),sigmaminus1),np.eye(d_t)))
    +delta_t*Delta1*np.kron(np.kron(np.eye(d_t),e1),np.eye(d_t)) 
 
    #TLS2 system term  
    Msys2=delta_t*Omega2*(np.kron(np.kron(np.eye(d_t),sigmaplus2),np.eye(d_t)) + np.kron(np.kron(np.eye(d_t),sigmaminus2),np.eye(d_t)))
    +delta_t*Delta2* np.kron(np.kron(np.eye(d_t),e2),np.eye(d_t)) 
 
    #interaction terms
    t11 = np.sqrt(gammaL2)*np.kron(np.kron(np.eye(d_t),sigmaminus2),delta_b_dagL(delta_t))
    t11hc = +np.sqrt(gammaL2)*np.kron(np.kron(np.eye(d_t),sigmaplus2),delta_bL(delta_t))
    t21 = np.sqrt(gammaR2)*np.kron(np.kron(delta_b_dagR(delta_t)*np.exp(1j*phase),sigmaminus2),np.eye(d_t))
    t21hc = +np.sqrt(gammaR2)*np.kron(np.kron(delta_bR(delta_t)*np.exp(-1j*phase),sigmaplus2),np.eye(d_t))
    t12 = np.sqrt(gammaL1)*np.kron(np.kron(delta_b_dagL(delta_t)*np.exp(1j*phase),sigmaminus1),np.eye(d_t))
    t12hc = +np.sqrt(gammaL1)*np.kron(np.kron(delta_bL(delta_t)*np.exp(-1j*phase),sigmaplus1),np.eye(d_t))
    t22 = np.sqrt(gammaR1)*np.kron(np.kron(np.eye(d_t),sigmaminus1),delta_b_dagR(delta_t))
    t22hc = +np.sqrt(gammaR1)*np.kron(np.kron(np.eye(d_t),sigmaplus1),delta_bR(delta_t))
 
    M = (Msys1 + Msys2 + t11 + t11hc + t21 + t21hc + t12 + t12hc + t22 + t22hc)
    return M

def hamiltonian_2TLS_M(delta_t:float, gammaL1:float, gammaR1:float, gammaL2:float, gammaR2:float, phase:float, d_sys:int, d_t:int, Omega1:float=0, Delta1:float=0, Omega2:float=0, Delta2:float=0) -> np.ndarray:
    """
    Hamilltonian for 2 TLSs in an infinite waveguide.
    
    Parameters
    ----------
    delta_t : float
        Time step for system evolution.

    gammaL1 : float
        Left decay rate of the first TLS.

    gammaR1 : float
        Right decay rate of the first TLS.

    gammaL2 : float
        Left decay rate of the second TLS.

    gammaR2 : float
        Right decay rate of the second TLS.

    phase : float
        Feedback phase between the TLSs.

    d_sys : int, default: 2
        TLS bin dimension

    d_t : int, default: 2, default: 2
        Time bin dimension

    Omega1 : float, default: 0
        Classical pump for the first TLS.

    Delta1 : float, default: 0
        Detuning between the pump and TLS frequencies for the first TLS.
    
    Omega2 : float, default: 0
        Classical pump for the second TLS.

    Delta2 : float, default: 0
        Detuning between the pump and TLS frequencies for the second TLS.

    Returns
    -------
    Hamiltonian : ndarray
        Hamiltonian coupling a two TLSs pumped by a classical fields to an infinite waveguide.

    Examples
    -------- 
    """
    d_sys1=int(d_sys/2)
    d_sys2=int(d_sys/2)
    sigmaplus1=np.kron(sigmaplus(),np.eye(d_sys2))
    sigmaminus1=np.kron(sigmaminus(),np.eye(d_sys2))
    sigmaplus2=np.kron(np.eye(d_sys1),sigmaplus())
    sigmaminus2=np.kron(np.eye(d_sys1),sigmaminus())
    e1=np.kron(e(),np.eye(d_sys2))    
    e2=np.kron(np.eye(d_sys1),e())   
    #TLS1 system term
    Msys1=delta_t*Omega1*(np.kron(sigmaplus1,np.eye(d_t)) + np.kron(sigmaminus1,np.eye(d_t)))
    +delta_t*Delta1*np.kron(e1,np.eye(d_t)) 
 
    #TLS2 system term  
    Msys2=delta_t*Omega2*(np.kron(sigmaplus2,np.eye(d_t)) + np.kron(sigmaminus2,np.eye(d_t)))
    +delta_t*Delta2* np.kron(e2,np.eye(d_t)) 
 
    #interaction terms
    t1R = np.sqrt(gammaR1)*(np.kron(sigmaminus1,delta_b_dagR(delta_t)) 
    + np.kron(sigmaplus1,delta_bR(delta_t)))
    t1L = np.sqrt(gammaL1)*(np.kron(sigmaminus1,delta_b_dagL(delta_t)*np.exp(1j*phase)) 
    + np.kron(sigmaplus1,delta_bL(delta_t)*np.exp(-1j*phase)))
    t2R = np.sqrt(gammaR2)*(np.kron(sigmaminus2,delta_b_dagR(delta_t)*np.exp(1j*phase)) 
    + np.kron(sigmaplus2,delta_bR(delta_t)*np.exp(-1j*phase)))                                                                                          
    t2L = np.sqrt(gammaL2)*(np.kron(sigmaminus2,delta_b_dagL(delta_t)) 
    + np.kron(sigmaplus2,delta_bL(delta_t)))
 
    M = (Msys1 + Msys2 + t1R + t1L + t2R + t2L )
    return M