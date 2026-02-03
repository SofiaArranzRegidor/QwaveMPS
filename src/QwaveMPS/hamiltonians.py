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

__all__ = ['hamiltonian_1tls', 'hamiltonian_1tls_feedback', 'hamiltonian_2tls_mar', 'hamiltonian_2tls_nmar', 'Hamiltonian']

def hamiltonian_1tls(params:InputParams, omega:float|np.ndarray=0, delta:float=0) -> Hamiltonian:
    """
    Hamiltonian for 1 two-level system coupled to an infinite waveguide.
    
    The returned Hamiltonian includes:
        - A classical pump term (omega) acting on the TLS, :math:`\\Omega(\\sigma^+ + \\sigma^-)`
        - A detuning term for the TLS, :math:`\\delta |e\\rangle\\langle e|`.
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
    d_t = np.prod(d_t_total)
    d_sys=np.prod(d_sys_total)
    t1= np.sqrt(gamma_l)*(np.kron(delta_b_l(delta_t,d_t_total),sigmaplus()) + np.kron(delta_b_dag_l(delta_t,d_t_total),sigmaminus())) 
    t2= np.sqrt(gamma_r)*(np.kron(delta_b_r(delta_t,d_t_total),sigmaplus()) + np.kron(delta_b_dag_r(delta_t,d_t_total),sigmaminus())) 
    if isinstance(omega, np.ndarray):
        omegas = tuple(omega)
        def hm_total(t_k):
            hm_sys=omegas[t_k]/2*delta_t*(np.kron(np.eye(d_t),sigmaplus()) + np.kron(np.eye(d_t),sigmaminus())) +delta_t*delta*np.kron(np.eye(d_t),e(d_sys)) 
            hm = hm_sys+t1+t2
            return hm  
    else:
        hm_sys=omega/2*delta_t*(np.kron(np.eye(d_t),sigmaplus()) + np.kron(np.eye(d_t),sigmaminus())) +delta_t*delta*np.kron(np.eye(d_t),e(d_sys)) 
        hm_total=hm_sys+t1+t2
    return hm_total
 
def hamiltonian_1tls_feedback(params:InputParams,omega:float|np.ndarray=0, delta:float=0) -> Hamiltonian:
    """
    Hamiltonian for 1 two-level system in a semi-infinite waveguide with a side mirror (with feedback).   
    
    The returned Hamiltonian includes:
        - A classical pump term (omega) acting on the TLS, :math:`\\Omega(\\sigma^+ + \\sigma^-)`
        - A detuning term for the TLS, :math:`\\delta |e\\rangle\\langle e|`.
        - Interaction terms between the TLS and a single photonic mode (on the present and feedback bins).
    
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
    t1=np.sqrt(gamma_l)*np.kron(np.kron(delta_b(delta_t,d_t)*np.exp(-1j*phase),np.eye(d_t)),sigmaplus())
    t2=np.sqrt(gamma_r)*np.kron(np.kron(np.eye(d_t),delta_b(delta_t,d_t)),sigmaplus())
    t3=np.sqrt(gamma_l)*np.kron(np.kron(delta_b_dag(delta_t,d_t)*np.exp(1j*phase),np.eye(d_t)),sigmaminus())
    t4=np.sqrt(gamma_r)*np.kron(np.kron(np.eye(d_t),delta_b_dag(delta_t,d_t)),sigmaminus())   
    if isinstance(omega, np.ndarray):
        omegas = tuple(omega)
        def hm_total(t_k):  
            hm_sys=omegas[t_k]/2*delta_t*(np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus()) +np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaminus()))
            hm = hm_sys + t1 + t2 + t3 + t4
            return hm
    else:        
        hm_sys=omega/2*delta_t*(np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus()) +np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaminus()))
        hm_total = hm_sys + t1 + t2 + t3 + t4
    return hm_total

def hamiltonian_2tls_mar(params:InputParams, omega1:float|np.ndarray=0, delta1:float=0, omega2:float|np.ndarray=0, delta2:float=0) -> Hamiltonian:
    """
    Hamiltonian for 2 two-level systems in an infinite waveguide in the Markovian regime.
    
    The returned Hamiltonian includes:
        - Classical pump terms (omega1/omega2) acting on TLS1/TLS2, :math:`\\Omega_i(\\sigma_i^+ + \\sigma_i^-)`
        - A detuning term delta1/delta2 for TLS1/TLS2, :math:`\\delta_i |e\\rangle_i\\langle e|_i`.
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
    t1R = np.sqrt(gamma_r1)*(np.kron(delta_b_dag_r(delta_t,d_t_total),sigmaminus1) 
    + np.kron(delta_b_r(delta_t,d_t_total),sigmaplus1))
    t1L = np.sqrt(gamma_l1)*(np.kron(delta_b_dag_l(delta_t,d_t_total)*np.exp(1j*phase),sigmaminus1) 
    + np.kron(delta_b_l(delta_t,d_t_total)*np.exp(-1j*phase),sigmaplus1))
    t2R = np.sqrt(gamma_r2)*(np.kron(delta_b_dag_r(delta_t,d_t_total)*np.exp(1j*phase),sigmaminus2) 
    + np.kron(delta_b_r(delta_t,d_t_total)*np.exp(-1j*phase),sigmaplus2))                                                                                          
    t2L = np.sqrt(gamma_l2)*(np.kron(delta_b_dag_l(delta_t,d_t_total),sigmaminus2) 
    + np.kron(delta_b_l(delta_t,d_t_total),sigmaplus2))
    
    if isinstance(omega1, np.ndarray) and isinstance(omega2, np.ndarray):
        omega1s = tuple(omega1)
        omega2s = tuple(omega2)
        hm_total=[]
        def hm_total(t_k):
            hm_sys1=delta_t*omega1s[t_k]/2*(np.kron(np.eye(d_t),sigmaplus1) + np.kron(np.eye(d_t),sigmaminus1))
            +delta_t*delta1*np.kron(np.eye(d_t),e1) 
            
            hm_sys2=delta_t*omega2s[t_k]/2*(np.kron(np.eye(d_t),sigmaplus2) + np.kron(np.eye(d_t),sigmaminus2))
            +delta_t*delta2* np.kron(np.eye(d_t),e2) 
           
            return hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L
   
    elif isinstance(omega1, np.ndarray):
        omega1s = tuple(omega1)
        hm_sys2=delta_t*omega2/2*(np.kron(np.eye(d_t),sigmaplus2) + np.kron(np.eye(d_t),sigmaminus2))
        +delta_t*delta2* np.kron(np.eye(d_t),e2)  

        def hm_total(t_k):
            hm_sys1=delta_t*omega1s[t_k]/2*(np.kron(np.eye(d_t),sigmaplus1) + np.kron(np.eye(d_t),sigmaminus1))
            +delta_t*delta1*np.kron(np.eye(d_t),e1) 
            
            return hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L

    elif isinstance(omega2, np.ndarray):
        omega2s = tuple(omega2)
        hm_sys1=delta_t*omega1/2*(np.kron(np.eye(d_t),sigmaplus1) + np.kron(np.eye(d_t),sigmaminus1))
        +delta_t*delta1*np.kron(np.eye(d_t),e1) 
        
        def hm_total(t_k):
            hm_sys2=delta_t*omega2s[t_k]/2*(np.kron(np.eye(d_t),sigmaplus2) + np.kron(np.eye(d_t),sigmaminus2))
            +delta_t*delta2* np.kron(np.eye(d_t),e2) 
             
            return hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L
        
    else:
        hm_sys1=delta_t*omega1/2*(np.kron(np.eye(d_t),sigmaplus1) + np.kron(np.eye(d_t),sigmaminus1))
        +delta_t*delta1*np.kron(np.eye(d_t),e1) 
     
        hm_sys2=delta_t*omega2/2*(np.kron(np.eye(d_t),sigmaplus2) + np.kron(np.eye(d_t),sigmaminus2))
        +delta_t*delta2* np.kron(np.eye(d_t),e2) 
 
        hm_total = (hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L )
    return hm_total

def hamiltonian_2tls_nmar(params:InputParams,omega1:float|np.ndarray=0, delta1:float=0, omega2:float|np.ndarray=0, delta2:float=0) -> Hamiltonian:
    """
    Hamiltonian for 2 two-level systems in an infinite waveguide in the non-Markovian regime (feedback).
    
    The returned Hamiltonian includes:
        - Classical pump terms (omega1/omega2) acting on TLS1/TLS2, :math:`\\Omega_i(\\sigma_i^+ + \\sigma_i^-)`
        - A detuning term delta1/delta2 for TLS1/TLS2, :math:`\\delta_i |e\\rangle_i\\langle e|_i`.
        - Interaction terms between the two-level systems and left/right photonic modes (on the present and feedback bins).
    
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
    t11 = np.sqrt(gamma_l2)*np.kron(np.kron(np.eye(d_t),delta_b_dag_l(delta_t,d_t_total)),sigmaminus2)
    t11hc = +np.sqrt(gamma_l2)*np.kron(np.kron(np.eye(d_t),delta_b_l(delta_t,d_t_total)),sigmaplus2)
    t21 = np.sqrt(gamma_r2)*np.kron(np.kron(delta_b_dag_r(delta_t,d_t_total)*np.exp(1j*phase),np.eye(d_t)),sigmaminus2)
    t21hc = +np.sqrt(gamma_r2)*np.kron(np.kron(delta_b_r(delta_t,d_t_total)*np.exp(-1j*phase),np.eye(d_t)),sigmaplus2)
    t12 = np.sqrt(gamma_l1)*np.kron(np.kron(delta_b_dag_l(delta_t,d_t_total)*np.exp(1j*phase),np.eye(d_t)),sigmaminus1)
    t12hc = +np.sqrt(gamma_l1)*np.kron(np.kron(delta_b_l(delta_t,d_t_total)*np.exp(-1j*phase),np.eye(d_t)),sigmaplus1)
    t22 = np.sqrt(gamma_r1)*np.kron(np.kron(np.eye(d_t),delta_b_dag_r(delta_t,d_t_total)),sigmaminus1)
    t22hc = +np.sqrt(gamma_r1)*np.kron(np.kron(np.eye(d_t),delta_b_r(delta_t,d_t_total)),sigmaplus1)
     
    if isinstance(omega1, np.ndarray) and isinstance(omega2, np.ndarray):
        omega1s = tuple(omega1)
        omega2s = tuple(omega2)
        def hm_total(t_k):
            hm_sys1 = delta_t*omega1s[t_k]/2*(np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus1) + np.kron(np.kron(np.eye(d_t),np.eye(d_t))),sigmaminus1)
            +delta_t*delta1*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),e1) 
            hm_sys2 = delta_t*omega2s[t_k]/2*(np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus2) + np.kron(np.kron(np.eye(d_t),np.eye(d_t))),sigmaminus2)
            +delta_t*delta2* np.kron(np.kron(np.eye(d_t),np.eye(d_t)),e2) 
            
            return hm_sys1 + hm_sys2 + t11 + t11hc + t21 + t21hc + t12 + t12hc + t22 + t22hc
   
    elif isinstance(omega1, np.ndarray):
        omega1s = tuple(omega1)
        hm_sys2=delta_t*omega2/2*(np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus2) + np.kron(np.kron(np.eye(d_t),np.eye(d_t))),sigmaminus2)
        +delta_t*delta2* np.kron(np.kron(np.eye(d_t),np.eye(d_t)),e2) 
        
        def hm_total(t_k):
            hm_sys1=delta_t*omega1s[t_k]/2*(np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus1) + np.kron(np.kron(np.eye(d_t),np.eye(d_t))),sigmaminus1)
            +delta_t*delta1*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),e1)
            
            return hm_sys1 + hm_sys2 + t11 + t11hc + t21 + t21hc + t12 + t12hc + t22 + t22hc

    elif isinstance(omega2, np.ndarray):
        omega2s = tuple(omega2)
        hm_sys1=delta_t*omega1/2*(np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus1) + np.kron(np.kron(np.eye(d_t),np.eye(d_t))),sigmaminus1)
        +delta_t*delta1*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),e1) 

        def hm_total(t_k):
            hm_sys2=delta_t*omega2s[t_k]/2*(np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus2) + np.kron(np.kron(np.eye(d_t),np.eye(d_t))),sigmaminus2)
            +delta_t*delta2* np.kron(np.kron(np.eye(d_t),np.eye(d_t)),e2) 
             
            return hm_sys1 + hm_sys2 + t11 + t11hc + t21 + t21hc + t12 + t12hc + t22 + t22hc
        
    else:
        hm_sys1=delta_t*omega1/2*(np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus1) + np.kron(np.kron(np.eye(d_t),np.eye(d_t))),sigmaminus1)
        +delta_t*delta1*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),e1) 
        
        hm_sys2=delta_t*omega2/2*(np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus2) + np.kron(np.kron(np.eye(d_t),np.eye(d_t))),sigmaminus2)
        +delta_t*delta2* np.kron(np.kron(np.eye(d_t),np.eye(d_t)),e2) 
        
        hm_total=hm_sys1 + hm_sys2 + t11 + t11hc + t21 + t21hc + t12 + t12hc + t22 + t22hc
    return hm_total



# Fix style/standardization of below functions
def basicSigmaPlus(dim):
    return np.diag(np.ones(dim-1, dtype=complex), -1)

def sigmaPlus(dimList, index):
    identityDim1 = np.prod(dimList[:index], dtype=np.uint)
    identityDim2 = np.prod(dimList[index+1:], dtype=np.uint)
    return np.kron(np.kron(np.eye(identityDim1), basicSigmaPlus(dimList[index])), np.eye(identityDim2))

def hamN2LSChiral(t, deltaT, d_t_total, tlsNum, gammas=None, detuningLs=None, phases=None, omegaPumps=None, pumpStarts=None, pumpEnds=None):
    if gammas == None: gammas = np.ones(tlsNum)
    if detuningLs == None: detuningLs = np.zeros(tlsNum)
    if phases == None: phases = np.zeros(tlsNum)
    if omegaPumps == None: omegaPumps = np.zeros(tlsNum)
    if pumpStarts == None: pumpStarts = np.zeros(tlsNum)
    if pumpEnds == None: pumpEnds = np.zeros(tlsNum)
    
    deltaB = delta_b(deltaT, np.prod(d_t_total))
    deltaBDag = np.conj(deltaB).T
    dTime = deltaB.shape[0]
    dSys = 2
    
    dimList = (np.ones(tlsNum) * dSys).astype(int)
    pumpCoeffs = np.array(omegaPumps) * (t >= np.array(pumpStarts)) * (t<= np.array(pumpEnds))
    
    H = np.zeros((dTime**tlsNum * np.prod(dimList),)*2, dtype=complex)
    for i in range(tlsNum):
        sigmaP = sigmaPlus(dimList, i)
        
        sigmaP = np.kron(np.eye(dTime**(i)), sigmaP)
        sigmaM = sigmaP.T
    
        h2LS = deltaT * detuningLs[i]  * np.kron(np.eye(dTime), sigmaP @ sigmaM)                             
        hPump = deltaT * pumpCoeffs[i] * np.kron(np.eye(dTime), sigmaP + sigmaM)
        hInt = np.sqrt(gammas[i]) * ( np.kron(deltaB, sigmaP) + np.kron(deltaBDag, sigmaM))
        
        Hlocal = h2LS + hPump + hInt
        
        Hlocal = np.kron(np.eye(dTime**(tlsNum-i-1)), Hlocal)
        H += Hlocal
        
    return H