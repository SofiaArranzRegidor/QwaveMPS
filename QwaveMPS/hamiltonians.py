#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains Hamiltonian constructors for different cases:
    - Single two-level system coupled to an infinite waveguide
    - Single two-level system with a mirror (feedback / semi-infinite waveguide)
    - Two two-level systems in the waveguide: Markovian regime (mar)
    - Two two-level systems in the waveguide: non-Markovian regime (nmar)

"""

import numpy as np
from QwaveMPS.operators import *
from QwaveMPS.parameters import InputParams
from typing import Callable, TypeAlias

# Type alias: Hamiltonian can be either a single ndarray or a callable indexed by time point for time dependent cases
Hamiltonian: TypeAlias = np.ndarray | Callable[[int], np.ndarray]

def hamiltonian_1tls(params:InputParams, omega:float|np.ndarray=0, delta:float=0) -> Hamiltonian:
    """
    Hamiltonian for 1 two-level system coupled to an infinite waveguide.
    The returned Hamiltonian includes:
    - A classical pump term (omega) acting on the TLS (sigma^+ + sigma^-)
    - A detuning term delta * |e><e| for the TLS
    - Interaction terms between the TLS and left/right photonic modes.
    
    Parameters
    ----------
    params : InputParams
        Class containing the input parameters.

    omega : float/np.ndarray, default: 0
       Classical pump amplitude. 
       If a float is provided (CW pump) a single Hamiltonian ndarray is returned. 
       If a 1D np.ndarray is given (pulsed light), the
       function returns a callable hm_total(t_k) that yields the Hamiltonian
       at discrete time index t_k using omega[t_k].

    delta : float, optional
        Detuning between the pump and two-level system transition frequency.

    Returns
    -------
    Hamiltonian : np.ndarray | Callable[[int], np.ndarray]
        Hamiltonian as a numpy.ndarray (time-independent drive) or a callable that
        accepts a time index and returns the Hamiltonian (time-dependent drive).
    """
    delta_t = params.delta_t
    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total
    gamma_l = params.gamma_l
    gamma_r = params.gamma_r
    
    d_t_l=d_t_total[0]
    d_t_r=d_t_total[1]
    d_sys=np.prod(d_sys_total)
    t1= np.sqrt(gamma_l)*(np.kron(sigmaplus(),delta_b_l(delta_t,d_t_total)) + np.kron(sigmaminus(),delta_b_dag_l(delta_t,d_t_total))) 
    t2= np.sqrt(gamma_r)*(np.kron(sigmaplus(),delta_b_r(delta_t,d_t_total)) + np.kron(sigmaminus(),delta_b_dag_r(delta_t,d_t_total))) 
    if isinstance(omega, np.ndarray):
        omegas = tuple(omega)
        def hm_total(t_k):
            hm_sys=omegas[t_k]/2*delta_t*(np.kron(sigmaplus(),np.eye(d_t_l*d_t_r)) + np.kron(sigmaminus(),np.eye(d_t_l*d_t_r))) +delta_t*delta*np.kron(e(d_sys),np.eye(d_t_l*d_t_r)) 
            hm = hm_sys+t1+t2
            return hm  
    else:
        hm_sys=omega/2*delta_t*(np.kron(sigmaplus(),np.eye(d_t_l*d_t_r)) + np.kron(sigmaminus(),np.eye(d_t_l*d_t_r))) +delta_t*delta*np.kron(e(d_sys),np.eye(d_t_l*d_t_r)) 
        hm_total=hm_sys+t1+t2
    return hm_total

    
def hamiltonian_1tls_feedback(params:InputParams,omega:float|np.ndarray=0, delta:float=0) -> Hamiltonian:
    """
    Hamiltonian for 1 two-level system in a semi-infinite waveguide with a side mirror (with feedback).   
    
    The returned Hamiltonian includes:
    - A classical pump term (omega) acting on the TLS (sigma^+ + sigma^-)
    - A detuning term delta * |e><e| for the TLS
    - Interaction terms between the TLS and a single photonic mode 
    (on the present and feedback bins).
    
    Parameters
    ----------
    params : InputParams
        Class containing the input parameters.
        (It must include the phase)
        
    omega : float or np.ndarray, default: 0
       Classical pump amplitude. 
       If a float is provided (CW pump) a single Hamiltonian ndarray is returned. 
       If a 1D np.ndarray is given (pulsed light), the
       function returns a callable hm_total(t_k) that yields the Hamiltonian
       at discrete time index t_k using omega[t_k].

    delta : float, default: 0
        Detuning between the pump and TLS transition frequency.

    Returns
    -------
    Hamiltonian : np.ndarray | Callable[[int], np.ndarray]
        Hamiltonian as a numpy.ndarray (time-independent drive) or a callable that
        accepts a time index and returns the Hamiltonian (time-dependent drive).
    """
    delta_t = params.delta_t
    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total
    gamma_l = params.gamma_l
    gamma_r = params.gamma_r
    phase = params.phase
    
    d_t=np.prod(d_t_total)
    d_sys=np.prod(d_sys_total)
    t1=np.sqrt(gamma_l)*np.kron(np.kron(delta_b(delta_t,d_t)*np.exp(-1j*phase),sigmaplus()),np.eye(d_t))
    t2=np.sqrt(gamma_r)*np.kron(np.kron(np.eye(d_t),sigmaplus()),delta_b(delta_t,d_t))
    t3=np.sqrt(gamma_l)*np.kron(np.kron(delta_b_dag(delta_t,d_t)*np.exp(1j*phase),sigmaminus()),np.eye(d_t))
    t4=np.sqrt(gamma_r)*np.kron(np.kron(np.eye(d_t),sigmaminus()),delta_b_dag(delta_t,d_t))   
    if isinstance(omega, np.ndarray):
        omegas = tuple(omega)
        def hm_total(t_k):  
            hm_sys=omegas[t_k]/2*delta_t*(np.kron(np.kron(np.eye(d_t),sigmaplus()),np.eye(d_t)) +np.kron(np.kron(np.eye(d_t),sigmaminus()),np.eye(d_t)))
            hm = hm_sys + t1 + t2 + t3 + t4
            return hm
    else:        
        hm_sys=omega/2*delta_t*(np.kron(np.kron(np.eye(d_t),sigmaplus()),np.eye(d_t)) +np.kron(np.kron(np.eye(d_t),sigmaminus()),np.eye(d_t)))
        hm_total = hm_sys + t1 + t2 + t3 + t4
    return hm_total

def hamiltonian_2tls_mar(params:InputParams, omega1:float|np.ndarray=0, delta1:float=0, omega2:float|np.ndarray=0, delta2:float=0) -> Hamiltonian:
    """
    Hamiltonian for 2 two-level systems in an infinite waveguide in the Markovian regime.
    
    The returned Hamiltonian includes:
    - Classical pump terms (omega1/omega2) acting on TLS1/TLS2
    - A detuning term delta1/delta2 * |e><e| for the TLS1/TLS2
    - Interaction terms between the TLSs and left/right photonic modes.
    
    Parameters
    ----------
    params : InputParams
        Class containing the input parameters.

    omega1 : float/np.ndarray, default: 0
        Drive for two-level system 1 
        (can be a float for CW pumps or a time-dependent array for pulsed light).

    delta1 : float, default: 0
        Detuning for two-level system 1.

    omega2 : float/np.ndarray, default: 0
        Drive for two-level system 2
        (can be a float for CW pumps or a time-dependent array for pulsed light).

    delta2 : float, default: 0
        Detuning for two-level system 1.

    Returns
    -------
    Hamiltonian : np.ndarray | Callable[[int], np.ndarray]
        Hamiltonian as a numpy.ndarray (time-independent drive) or a callable that
        accepts a time index and returns the Hamiltonian (time-dependent drive).
    """
    delta_t,gamma_l1, gamma_r1, gamma_l2, gamma_r2,phase,d_sys_total, d_t_total \
    =params.delta_t,params.gamma_l, params.gamma_r,params.gamma_l2,params.gamma_r2,params.phase,params.d_sys_total,params.d_t_total
    
    
    d_sys1=d_sys_total[0]
    d_sys2=d_sys_total[1]
    d_t=np.prod(d_t_total)
    
    sigmaplus1=np.kron(sigmaplus(),np.eye(d_sys2))
    sigmaminus1=np.kron(sigmaminus(),np.eye(d_sys2))
    sigmaplus2=np.kron(np.eye(d_sys1),sigmaplus())
    sigmaminus2=np.kron(np.eye(d_sys1),sigmaminus())
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
        omega1s = tuple(omega1)
        omega2s = tuple(omega2)
        hm_total=[]
        def hm_total(t_k):
            hm_sys1=delta_t*omega1s[t_k]/2*(np.kron(sigmaplus1,np.eye(d_t)) + np.kron(sigmaminus1,np.eye(d_t)))
            +delta_t*delta1*np.kron(e1,np.eye(d_t)) 
            
            hm_sys2=delta_t*omega2s[t_k]/2*(np.kron(sigmaplus2,np.eye(d_t)) + np.kron(sigmaminus2,np.eye(d_t)))
            +delta_t*delta2* np.kron(e2,np.eye(d_t)) 
           
            return hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L
   
    elif isinstance(omega1, np.ndarray):
        omega1s = tuple(omega1)
        hm_sys2=delta_t*omega2/2*(np.kron(sigmaplus2,np.eye(d_t)) + np.kron(sigmaminus2,np.eye(d_t)))
        +delta_t*delta2* np.kron(e2,np.eye(d_t))  

        def hm_total(t_k):
            hm_sys1=delta_t*omega1s[t_k]/2*(np.kron(sigmaplus1,np.eye(d_t)) + np.kron(sigmaminus1,np.eye(d_t)))
            +delta_t*delta1*np.kron(e1,np.eye(d_t)) 
            
            return hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L

    elif isinstance(omega2, np.ndarray):
        omega2s = tuple(omega2)
        hm_sys1=delta_t*omega1/2*(np.kron(sigmaplus1,np.eye(d_t)) + np.kron(sigmaminus1,np.eye(d_t)))
        +delta_t*delta1*np.kron(e1,np.eye(d_t)) 
        
        def hm_total(t_k):
            hm_sys2=delta_t*omega2s[t_k]/2*(np.kron(sigmaplus2,np.eye(d_t)) + np.kron(sigmaminus2,np.eye(d_t)))
            +delta_t*delta2* np.kron(e2,np.eye(d_t)) 
             
            return hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L
        
    else:
        hm_sys1=delta_t*omega1/2*(np.kron(sigmaplus1,np.eye(d_t)) + np.kron(sigmaminus1,np.eye(d_t)))
        +delta_t*delta1*np.kron(e1,np.eye(d_t)) 
     
        hm_sys2=delta_t*omega2/2*(np.kron(sigmaplus2,np.eye(d_t)) + np.kron(sigmaminus2,np.eye(d_t)))
        +delta_t*delta2* np.kron(e2,np.eye(d_t)) 
 
        hm_total = (hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L )
    return hm_total

def hamiltonian_2tls_nmar(params:InputParams,omega1:float|np.ndarray=0, delta1:float=0, omega2:float|np.ndarray=0, delta2:float=0) -> Hamiltonian:
    """
    Hamiltonian for 2 two-level systems in an infinite waveguide in the non-Markovian regime (feedback).
    
    The returned Hamiltonian includes:
    - Classical pump terms (omega1/omega2) acting on TLS1/TLS2
    - A detuning term delta1/delta2 * |e><e| for the TLS1/TLS2
    - Interaction terms between the two-level systems and left/right photonic modes
    (on the present and feedback bins).
    
    Parameters
    ----------
    params:InputParams
        Class containing the input parameters.
        
    omega1 : float/np.ndarray, default: 0
        Drive for two-level system 1 
        (can be a float for CW pumps or a time-dependent array for pulsed light).

    delta1 : float, default: 0
        Detuning for two-level system 1.

    omega2 : float/np.ndarray, default: 0
        Drive for two-level system 2
        (can be a float for CW pumps or a time-dependent array for pulsed light).

    delta2 : float, default: 0
        Detuning for two-level system 1.

    Returns
    -------
    Hamiltonian : np.ndarray | Callable[[int], np.ndarray]
        Hamiltonian as a numpy.ndarray (time-independent drive) or a callable that
        accepts a time index and returns the Hamiltonian (time-dependent drive).
    """
    delta_t,gamma_l1, gamma_r1, gamma_l2, gamma_r2,phase,d_sys_total, d_t_total \
    =params.delta_t,params.gamma_l, params.gamma_r,params.gamma_l2,params.gamma_r2,params.phase,params.d_sys_total,params.d_t_total
    
    
    d_sys1=d_sys_total[0]
    d_sys2=d_sys_total[1]
    d_t=np.prod(d_t_total)
    
    sigmaplus1=np.kron(sigmaplus(),np.eye(d_sys2))
    sigmaminus1=np.kron(sigmaminus(),np.eye(d_sys2))
    sigmaplus2=np.kron(np.eye(d_sys1),sigmaplus())
    sigmaminus2=np.kron(np.eye(d_sys1),sigmaminus())
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



