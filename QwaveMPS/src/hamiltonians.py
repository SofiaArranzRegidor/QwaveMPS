#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the hamiltonians for different cases

"""

import numpy as np
from QwaveMPS.src.operators import *
from QwaveMPS.src.parameters import InputParams


def hamiltonian_1tls(params:InputParams, omega:float|np.ndarray=0, delta:float=0) -> np.ndarray|list:
    """
    Hamiltonian for 1 TLS in the waveguide
    
    Parameters
    ----------
    delta_t : float
        Time step for system evolution.

    gamma_l : float
        Left decay rate.

    gamma_r : float
        Right decay rate.

    d_sys_total : ndarray
        List of sizes of system Hilbert spaces.

    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

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
    delta_t = params.delta_t
    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total
    gamma_l = params.gamma_l
    gamma_r = params.gamma_r
    
    d_t_l=d_t_total[0]
    d_t_r=d_t_total[1]
    d_sys=np.prod(d_sys_total)
    t1= np.sqrt(gamma_l)*(np.kron(sigmaplus(d_sys),delta_b_l(delta_t,d_t_total)) + np.kron(sigmaminus(d_sys),delta_b_dag_l(delta_t,d_t_total))) 
    t2= np.sqrt(gamma_r)*(np.kron(sigmaplus(d_sys),delta_b_r(delta_t,d_t_total)) + np.kron(sigmaminus(d_sys),delta_b_dag_r(delta_t,d_t_total))) 
    if isinstance(omega, np.ndarray):
        omegas = tuple(omega)
        def hm_total(t_k):
            hm_sys=omegas[t_k]/2*delta_t*(np.kron(sigmaplus(d_sys),np.eye(d_t_l*d_t_r)) + np.kron(sigmaminus(d_sys),np.eye(d_t_l*d_t_r))) +delta_t*delta*np.kron(e(d_sys),np.eye(d_t_l*d_t_r)) 
            hm = hm_sys+t1+t2
            return hm  
    else:
        hm_sys=omega/2*delta_t*(np.kron(sigmaplus(d_sys),np.eye(d_t_l*d_t_r)) + np.kron(sigmaminus(d_sys),np.eye(d_t_l*d_t_r))) +delta_t*delta*np.kron(e(d_sys),np.eye(d_t_l*d_t_r)) 
        hm_total=hm_sys+t1+t2
    return hm_total

    
def hamiltonian_1tls_feedback(params:InputParams,omega:float|np.ndarray=0, delta:float=0) -> np.ndarray|list:
    """
    Hamiltonian for 1 TLS in a semi-infinite waveguide with a side mirror
    
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

    d_sys_total : ndarray
        List of sizes of system Hilbert spaces.

    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

    omega : float, default: 0
        Classical pump

    delta : float, default: 0
        Detuning between the pump and TLS frequencies

    Returns
    -------
    Hamiltonian: ndarray
        Hamiltonian coupling a single TLS pumped by a classical field to the semi-infinite waveguide.

    Examples
    -------- 
    """
    delta_t = params.delta_t
    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total
    gamma_l = params.gamma_l
    gamma_r = params.gamma_r
    phase = params.phase
    
    d_t=np.prod(d_t_total)
    d_sys=np.prod(d_sys_total)
    t1=np.sqrt(gamma_l)*np.kron(np.kron(delta_b(delta_t)*np.exp(-1j*phase),sigmaplus(d_sys)),np.eye(d_t))
    t2=np.sqrt(gamma_r)*np.kron(np.kron(np.eye(d_t),sigmaplus(d_sys)),delta_b(delta_t))
    t3=np.sqrt(gamma_l)*np.kron(np.kron(delta_b_dag(delta_t)*np.exp(1j*phase),sigmaminus()),np.eye(d_t))
    t4=np.sqrt(gamma_r)*np.kron(np.kron(np.eye(d_t),sigmaminus(d_sys)),delta_b_dag(delta_t))   
    if isinstance(omega, np.ndarray):
        omegas = tuple(omega)
        def hm_total(t_k):  
            hm_sys=omegas[t_k]/2*delta_t*(np.kron(np.kron(np.eye(d_t),sigmaplus(d_sys)),np.eye(d_t)) +np.kron(np.kron(np.eye(d_t),sigmaminus(d_sys)),np.eye(d_t)))
            hm = hm_sys + t1 + t2 + t3 + t4
            return hm
    else:        
        hm_sys=omega/2*delta_t*(np.kron(np.kron(np.eye(d_t),sigmaplus(d_sys)),np.eye(d_t)) +np.kron(np.kron(np.eye(d_t),sigmaminus(d_sys)),np.eye(d_t)))
        hm_total = hm_sys + t1 + t2 + t3 + t4
    return hm_total


def hamiltonian_2tls_nmar(params:InputParams,omega1:float=0, delta1:float=0, omega2:float=0, delta2:float=0) -> np.ndarray:
    """
    Hamiltonian for 2 TLSs in an infinite waveguide.
    
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

    d_sys_total : ndarray
        List of sizes of system Hilbert spaces.

    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

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
    delta_t,gamma_l1, gamma_r1, gamma_l2, gamma_r2,phase,d_sys_total, d_t_total \
    =params.delta_t,params.gamma_l, params.gamma_r,params.gamma_l2,params.gamma_r2,params.phase,params.d_sys_total,params.d_t_total
    
    
    d_sys1=d_sys_total[0]
    d_sys2=d_sys_total[1]
    d_t=np.prod(d_t_total)
    
    sigmaplus1=np.kron(sigmaplus(d_sys1),np.eye(d_sys2))
    sigmaminus1=np.kron(sigmaminus(d_sys1),np.eye(d_sys2))
    sigmaplus2=np.kron(np.eye(d_sys1),sigmaplus(d_sys2))
    sigmaminus2=np.kron(np.eye(d_sys1),sigmaminus(d_sys2))
    e1=np.kron(e(),np.eye(d_sys2))    
    e2=np.kron(np.eye(d_sys1),e(d_sys2))   
    
    #interaction terms
    t11 = np.sqrt(gamma_l2)*np.kron(np.kron(np.eye(d_t),sigmaminus2),delta_b_dag_l(delta_t,d_t_total))
    t11hc = +np.sqrt(gamma_l2)*np.kron(np.kron(np.eye(d_t),sigmaplus2),delta_b_l(delta_t,d_t_total))
    t21 = np.sqrt(gamma_r2)*np.kron(np.kron(delta_b_dag_r(delta_t,d_t_total)*np.exp(1j*phase),sigmaminus2),np.eye(d_t))
    t21hc = +np.sqrt(gamma_r2)*np.kron(np.kron(delta_b_r(delta_t,d_t_total)*np.exp(-1j*phase),sigmaplus2),np.eye(d_t))
    t12 = np.sqrt(gamma_l1)*np.kron(np.kron(delta_b_dag_l(delta_t,d_t_total)*np.exp(1j*phase),sigmaminus1),np.eye(d_t))
    t12hc = +np.sqrt(gamma_l1)*np.kron(np.kron(delta_b_l(delta_t,d_t_total)*np.exp(-1j*phase),sigmaplus1),np.eye(d_t))
    t22 = np.sqrt(gamma_r1)*np.kron(np.kron(np.eye(d_t),sigmaminus1),delta_b_dag_r(delta_t,d_t_total))
    t22hc = +np.sqrt(gamma_r1)*np.kron(np.kron(np.eye(d_t),sigmaplus1),delta_b_r(delta_t,d_t_total))
     
    if isinstance(omega1, np.ndarray) and isinstance(omega2, np.ndarray):
        omega1s = tuple(omega1)
        omega2s = tuple(omega2)
        def hm_total(t_k):
            hm_sys1 = delta_t*omega1s[t_k]/2*(np.kron(np.kron(np.eye(d_t),sigmaplus1),np.eye(d_t)) + np.kron(np.kron(np.eye(d_t),sigmaminus1),np.eye(d_t)))
            +delta_t*delta1*np.kron(np.kron(np.eye(d_t),e1),np.eye(d_t)) 
            hm_sys2 = delta_t*omega2s[t_k]/2*(np.kron(np.kron(np.eye(d_t),sigmaplus2),np.eye(d_t)) + np.kron(np.kron(np.eye(d_t),sigmaminus2),np.eye(d_t)))
            +delta_t*delta2* np.kron(np.kron(np.eye(d_t),e2),np.eye(d_t)) 
            
            return hm_sys1 + hm_sys2 + t11 + t11hc + t21 + t21hc + t12 + t12hc + t22 + t22hc
   
    elif isinstance(omega1, np.ndarray):
        omega1s = tuple(omega1)
        hm_sys2=delta_t*omega2/2*(np.kron(np.kron(np.eye(d_t),sigmaplus2),np.eye(d_t)) + np.kron(np.kron(np.eye(d_t),sigmaminus2),np.eye(d_t)))
        +delta_t*delta2* np.kron(np.kron(np.eye(d_t),e2),np.eye(d_t)) 
        
        def hm_total(t_k):
            hm_sys1=delta_t*omega1s[t_k]/2*(np.kron(np.kron(np.eye(d_t),sigmaplus1),np.eye(d_t)) + np.kron(np.kron(np.eye(d_t),sigmaminus1),np.eye(d_t)))
            +delta_t*delta1*np.kron(np.kron(np.eye(d_t),e1),np.eye(d_t)) 
            
            return hm_sys1 + hm_sys2 + t11 + t11hc + t21 + t21hc + t12 + t12hc + t22 + t22hc

    elif isinstance(omega2, np.ndarray):
        omega2s = tuple(omega2)
        hm_sys1=delta_t*omega1/2*(np.kron(np.kron(np.eye(d_t),sigmaplus1),np.eye(d_t)) + np.kron(np.kron(np.eye(d_t),sigmaminus1),np.eye(d_t)))
        +delta_t*delta1*np.kron(np.kron(np.eye(d_t),e1),np.eye(d_t)) 

        def hm_total(t_k):
            hm_sys2=delta_t*omega2s[t_k]/2*(np.kron(np.kron(np.eye(d_t),sigmaplus2),np.eye(d_t)) + np.kron(np.kron(np.eye(d_t),sigmaminus2),np.eye(d_t)))
            +delta_t*delta2* np.kron(np.kron(np.eye(d_t),e2),np.eye(d_t)) 
             
            return hm_sys1 + hm_sys2 + t11 + t11hc + t21 + t21hc + t12 + t12hc + t22 + t22hc
        
    else:
        hm_sys1=delta_t*omega1/2*(np.kron(np.kron(np.eye(d_t),sigmaplus1),np.eye(d_t)) + np.kron(np.kron(np.eye(d_t),sigmaminus1),np.eye(d_t)))
        +delta_t*delta1*np.kron(np.kron(np.eye(d_t),e1),np.eye(d_t)) 
        
        hm_sys2=delta_t*omega2/2*(np.kron(np.kron(np.eye(d_t),sigmaplus2),np.eye(d_t)) + np.kron(np.kron(np.eye(d_t),sigmaminus2),np.eye(d_t)))
        +delta_t*delta2* np.kron(np.kron(np.eye(d_t),e2),np.eye(d_t)) 
        
        hm_total=hm_sys1 + hm_sys2 + t11 + t11hc + t21 + t21hc + t12 + t12hc + t22 + t22hc
    return hm_total


def hamiltonian_2tls_mar(params:InputParams, omega1:float=0, delta1:float=0, omega2:float=0, delta2:float=0) -> np.ndarray:
    """
    Hamiltonian for 2 TLSs in an infinite waveguide.
    
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

    d_sys_total : ndarray
        List of sizes of system Hilbert spaces.

    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

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
    delta_t,gamma_l1, gamma_r1, gamma_l2, gamma_r2,phase,d_sys_total, d_t_total \
    =params.delta_t,params.gamma_l, params.gamma_r,params.gamma_l2,params.gamma_r2,params.phase,params.d_sys_total,params.d_t_total
    
    
    d_sys1=d_sys_total[0]
    d_sys2=d_sys_total[1]
    d_t=np.prod(d_t_total)
    
    sigmaplus1=np.kron(sigmaplus(d_sys1),np.eye(d_sys2))
    sigmaminus1=np.kron(sigmaminus(d_sys1),np.eye(d_sys2))
    sigmaplus2=np.kron(np.eye(d_sys1),sigmaplus(d_sys2))
    sigmaminus2=np.kron(np.eye(d_sys1),sigmaminus(d_sys2))
    e1=np.kron(e(d_sys1),np.eye(d_sys2))    
    e2=np.kron(np.eye(d_sys1),e(d_sys2))   
    
 
    #interaction terms
    t1R = np.sqrt(gamma_r1)*(np.kron(sigmaminus1,delta_b_dag_r(delta_t,d_t_total)) 
    + np.kron(sigmaplus1,delta_b_r(delta_t,d_t_total)))
    t1L = np.sqrt(gamma_l1)*(np.kron(sigmaminus1,delta_b_dag_l(delta_t,d_t_total)*np.exp(1j*phase)) 
    + np.kron(sigmaplus1,delta_b_l(delta_t,d_t_total)*np.exp(-1j*phase)))
    t2R = np.sqrt(gamma_r2)*(np.kron(sigmaminus2,delta_b_dag_r(delta_t,d_t_total)*np.exp(1j*phase)) 
    + np.kron(sigmaplus2,delta_b_r(delta_t,d_t_total)*np.exp(-1j*phase)))                                                                                          
    t2L = np.sqrt(gamma_l2)*(np.kron(sigmaminus2,delta_b_dag_l(delta_t,d_t_total)) 
    + np.kron(sigmaplus2,delta_b_l(delta_t,d_t_total)))
    
    if isinstance(omega1, np.ndarray) and isinstance(omega2, np.ndarray):
        hm_total=[]
        for om1,om2 in zip(omega1,omega2):
            hm_sys1=delta_t*om1/2*(np.kron(sigmaplus1,np.eye(d_t)) + np.kron(sigmaminus1,np.eye(d_t)))
            +delta_t*delta1*np.kron(e1,np.eye(d_t)) 
            
            hm_sys2=delta_t*om2/2*(np.kron(sigmaplus2,np.eye(d_t)) + np.kron(sigmaminus2,np.eye(d_t)))
            +delta_t*delta2* np.kron(e2,np.eye(d_t)) 
           
            hm_total.append(hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L)
   
    elif isinstance(omega1, np.ndarray):
        hm_total=[]
        hm_sys2=delta_t*omega2/2*(np.kron(sigmaplus2,np.eye(d_t)) + np.kron(sigmaminus2,np.eye(d_t)))
        +delta_t*delta2* np.kron(e2,np.eye(d_t))  
        for om1 in omega1:
            hm_sys1=delta_t*om1/2*(np.kron(sigmaplus1,np.eye(d_t)) + np.kron(sigmaminus1,np.eye(d_t)))
            +delta_t*delta1*np.kron(e1,np.eye(d_t)) 
            
            hm_total.append(hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L)

    elif isinstance(omega2, np.ndarray):
        hm_total=[]
        hm_sys1=delta_t*omega1/2*(np.kron(sigmaplus1,np.eye(d_t)) + np.kron(sigmaminus1,np.eye(d_t)))
        +delta_t*delta1*np.kron(e1,np.eye(d_t)) 
        for om2 in omega2:
            hm_sys2=delta_t*om2/2*(np.kron(sigmaplus2,np.eye(d_t)) + np.kron(sigmaminus2,np.eye(d_t)))
            +delta_t*delta2* np.kron(e2,np.eye(d_t)) 
             
            hm_total.append(hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L)
        
    else:
        hm_sys1=delta_t*omega1/2*(np.kron(sigmaplus1,np.eye(d_t)) + np.kron(sigmaminus1,np.eye(d_t)))
        +delta_t*delta1*np.kron(e1,np.eye(d_t)) 
     
        hm_sys2=delta_t*omega2/2*(np.kron(sigmaplus2,np.eye(d_t)) + np.kron(sigmaminus2,np.eye(d_t)))
        +delta_t*delta2* np.kron(e2,np.eye(d_t)) 
 
        hm_total = (hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L )
    return hm_total
