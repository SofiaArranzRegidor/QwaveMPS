#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the hamiltonians for different cases

"""

import numpy as np
from .operators import *

# op=basic_operators()


def hamiltonian_1tls(delta_t:float, gamma_l:float, gamma_r:float, d_t:int=2, d_sys:int=2, omega:float=0, delta:float=0) -> np.ndarray:
    """
    Hamilltonian for 1 TLS in the waveguide
    
    Parameters
    ----------
    delta_t : float
        Time step for system evolution.

    gamma_l : float
        Left decay rate.

    gamma_r : float
        Right decay rate.

    d_sys : int, default: 2
        TLS bin dimension

    d_t : int, default: 2, default: 2
        Time bin dimension

    omega : float, default: 0
        Classical pump

    delta : float, default: 0
        Detuning between the pump and TLS frequencies

    Returns
    -------
    Hamiltonian : ndarray
        Hamiltonian coupling a single TLS pumped by a classical field to a waveguide.

    Examples
    -------- 
    """
    
    hm_sys=omega*delta_t*(np.kron(sigmaplus(),np.eye(d_t*d_t)) + np.kron(sigmaminus(),np.eye(d_t*d_t))) +delta_t*delta*np.kron(e(),np.eye(d_t*d_t)) 
    t1= np.sqrt(gamma_l)*(np.kron(sigmaplus(),delta_b_l(delta_t)) + np.kron(sigmaminus(),delta_b_dag_l(delta_t))) 
    t2= np.sqrt(gamma_r)*(np.kron(sigmaplus(),delta_b_r(delta_t)) + np.kron(sigmaminus(),delta_b_dag_r(delta_t))) 
    hm_total=hm_sys+t1+t2
    return hm_total

    
def hamiltonian_1tls_feedback(delta_t:float, gamma_l:float, gamma_r:float, phase:float, d_t:int, d_sys:int, omega:float=0, delta:float=0) -> np.ndarray:
    """
    Hamilltonian for 1 TLS in a semi-infinite waveguide with a side mirror
    
    Parameters
    ----------
    delta_t : float
        Time step for system evolution.

    gamma_l : float
        Left decay rate.

    gamma_r : float
        Right decay rate.

    phase : float
        Feedback phase between the TLS and the mirror.

    d_sys : int, default: 2
        TLS bin dimension

    d_t : int, default: 2, default: 2
        Time bin dimension

    omega : float, default: 0
        Classical pump

    delta : float, default: 0
        Detuning between the pump and TLS frequencies

    Returns
    -------
    Hamiltonian : ndarray
        Hamiltonian coupling a single TLS pumped by a classical field to the semi-infinite waveguide.

    Examples
    -------- 
    """
    hm_sys=omega*delta_t*(np.kron(np.kron(np.eye(d_t),sigmaplus()),np.eye(d_t)) +np.kron(np.kron(np.eye(d_t),sigmaminus()),np.eye(d_t)))
    t1=np.sqrt(gamma_l)*np.kron(np.kron(delta_b(delta_t)*np.exp(-1j*phase),sigmaplus()),np.eye(d_t))
    t2=np.sqrt(gamma_r)*np.kron(np.kron(np.eye(d_t),sigmaplus()),delta_b(delta_t))
    t3=np.sqrt(gamma_l)*np.kron(np.kron(delta_b_dag(delta_t)*np.exp(1j*phase),sigmaminus()),np.eye(d_t))
    t4=np.sqrt(gamma_r)*np.kron(np.kron(np.eye(d_t),sigmaminus()),delta_b_dag(delta_t))   
    hm_total = hm_sys + t1 + t2 + t3 + t4
    return hm_total


def hamiltonian_2tls_nmar(delta_t:float, gamma_l1:float, gamma_r1:float, gamma_l2:float, gamma_r2:float, phase:float, d_sys:int, d_t:int, omega1:float=0, delta1:float=0, omega2:float=0, delta2:float=0) -> np.ndarray:
    """
    Hamilltonian for 2 TLSs in an infinite waveguide.
    
    Parameters
    ----------
    delta_t : float
        Time step for system evolution.

    gamma_l1 : float
        Left decay rate of the first TLS.

    gamma_r1 : float
        Right decay rate of the first TLS.

    gamma_l2 : float
        Left decay rate of the second TLS.

    gamma_r2 : float
        Right decay rate of the second TLS.

    phase : float
        Feedback phase between the TLSs.

    d_sys : int, default: 2
        TLS bin dimension

    d_t : int, default: 2, default: 2
        Time bin dimension

    omega1 : float, default: 0
        Classical pump for the first TLS.

    delta1 : float, default: 0
        Detuning between the pump and TLS frequencies for the first TLS.
    
    omega2 : float, default: 0
        Classical pump for the second TLS.

    delta2 : float, default: 0
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
    hm_sys1=delta_t*omega1*(np.kron(np.kron(np.eye(d_t),sigmaplus1),np.eye(d_t)) + np.kron(np.kron(np.eye(d_t),sigmaminus1),np.eye(d_t)))
    +delta_t*delta1*np.kron(np.kron(np.eye(d_t),e1),np.eye(d_t)) 
 
    #TLS2 system term  
    hm_sys2=delta_t*omega2*(np.kron(np.kron(np.eye(d_t),sigmaplus2),np.eye(d_t)) + np.kron(np.kron(np.eye(d_t),sigmaminus2),np.eye(d_t)))
    +delta_t*delta2* np.kron(np.kron(np.eye(d_t),e2),np.eye(d_t)) 
 
    #interaction terms
    t11 = np.sqrt(gamma_l2)*np.kron(np.kron(np.eye(d_t),sigmaminus2),delta_b_dag_l(delta_t))
    t11hc = +np.sqrt(gamma_l2)*np.kron(np.kron(np.eye(d_t),sigmaplus2),delta_b_l(delta_t))
    t21 = np.sqrt(gamma_r2)*np.kron(np.kron(delta_b_dag_r(delta_t)*np.exp(1j*phase),sigmaminus2),np.eye(d_t))
    t21hc = +np.sqrt(gamma_r2)*np.kron(np.kron(delta_b_r(delta_t)*np.exp(-1j*phase),sigmaplus2),np.eye(d_t))
    t12 = np.sqrt(gamma_l1)*np.kron(np.kron(delta_b_dag_l(delta_t)*np.exp(1j*phase),sigmaminus1),np.eye(d_t))
    t12hc = +np.sqrt(gamma_l1)*np.kron(np.kron(delta_b_l(delta_t)*np.exp(-1j*phase),sigmaplus1),np.eye(d_t))
    t22 = np.sqrt(gamma_r1)*np.kron(np.kron(np.eye(d_t),sigmaminus1),delta_b_dag_r(delta_t))
    t22hc = +np.sqrt(gamma_r1)*np.kron(np.kron(np.eye(d_t),sigmaplus1),delta_b_r(delta_t))
 
    hm_total = (hm_sys1 + hm_sys2 + t11 + t11hc + t21 + t21hc + t12 + t12hc + t22 + t22hc)
    return hm_total

def hamiltonian_2tls_mar(delta_t:float, gamma_l1:float, gamma_r1:float, gamma_l2:float, gamma_r2:float, phase:float, d_sys:int, d_t:int, omega1:float=0, delta1:float=0, omega2:float=0, delta2:float=0) -> np.ndarray:
    """
    Hamilltonian for 2 TLSs in an infinite waveguide.
    
    Parameters
    ----------
    delta_t : float
        Time step for system evolution.

    gamma_l1 : float
        Left decay rate of the first TLS.

    gamma_r1 : float
        Right decay rate of the first TLS.

    gamma_l2 : float
        Left decay rate of the second TLS.

    gamma_r2 : float
        Right decay rate of the second TLS.

    phase : float
        Feedback phase between the TLSs.

    d_sys : int, default: 2
        TLS bin dimension

    d_t : int, default: 2, default: 2
        Time bin dimension

    omega1 : float, default: 0
        Classical pump for the first TLS.

    delta1 : float, default: 0
        Detuning between the pump and TLS frequencies for the first TLS.
    
    omega2 : float, default: 0
        Classical pump for the second TLS.

    delta2 : float, default: 0
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
    hm_sys1=delta_t*omega1*(np.kron(sigmaplus1,np.eye(d_t)) + np.kron(sigmaminus1,np.eye(d_t)))
    +delta_t*delta1*np.kron(e1,np.eye(d_t)) 
 
    #TLS2 system term  
    hm_sys2=delta_t*omega2*(np.kron(sigmaplus2,np.eye(d_t)) + np.kron(sigmaminus2,np.eye(d_t)))
    +delta_t*delta2* np.kron(e2,np.eye(d_t)) 
 
    #interaction terms
    t1R = np.sqrt(gamma_r1)*(np.kron(sigmaminus1,delta_b_dag_r(delta_t)) 
    + np.kron(sigmaplus1,delta_b_r(delta_t)))
    t1L = np.sqrt(gamma_l1)*(np.kron(sigmaminus1,delta_b_dag_l(delta_t)*np.exp(1j*phase)) 
    + np.kron(sigmaplus1,delta_b_l(delta_t)*np.exp(-1j*phase)))
    t2R = np.sqrt(gamma_r2)*(np.kron(sigmaminus2,delta_b_dag_r(delta_t)*np.exp(1j*phase)) 
    + np.kron(sigmaplus2,delta_b_r(delta_t)*np.exp(-1j*phase)))                                                                                          
    t2L = np.sqrt(gamma_l2)*(np.kron(sigmaminus2,delta_b_dag_l(delta_t)) 
    + np.kron(sigmaplus2,delta_b_l(delta_t)))
 
    hm_total = (hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L )
    return hm_total