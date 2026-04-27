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

from .symmetrical_coupling_helper import Symmetrical_Coupling_Helper

# Type alias: Hamiltonian can be either a single ndarray or a callable indexed by time point for time dependent cases
Hamiltonian: TypeAlias = np.ndarray | Callable[[int], np.ndarray]

__all__ = ['hamiltonian_1tls', 'hamiltonian_1tls_feedback', 'hamiltonian_2tls_mar', 'hamiltonian_2tls_nmar', 'Hamiltonian',
           'hamN2LSChiral', 'hamN2LS', 'hamiltonian_1tls_chiral', 'hamiltonian_Ntls_sym_eff']

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
            hm_sys=delta_t*(omegas[t_k]/2*np.kron(np.eye(d_t),sigmaplus()) + np.conj(omegas[t_k])/2*np.kron(np.eye(d_t),sigmaminus())) + delta_t*delta*np.kron(np.eye(d_t),e(d_sys)) 
            hm = hm_sys+t1+t2
            return hm  
    else:
        hm_sys=delta_t*(omega/2*np.kron(np.eye(d_t),sigmaplus()) + np.conj(omega)/2*np.kron(np.eye(d_t),sigmaminus())) +delta_t*delta*np.kron(np.eye(d_t),e(d_sys)) 
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
            hm_sys=delta_t*(omegas[t_k]/2*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus()) +np.conj(omegas[t_k])/2*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaminus()))
            hm = hm_sys + t1 + t2 + t3 + t4
            return hm
    else:        
        hm_sys=delta_t*(np.kron(omega/2*np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus()) + np.conj(omega)/2*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaminus()))
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
    
    j_12 = 0.5*(np.sqrt(gamma_r1 * gamma_r2) + np.sqrt(gamma_l1 * gamma_l2)) * np.imag(np.exp(1j*phase))
    h_exch = delta_t*j_12*np.kron(np.eye(d_t), sigmaplus1 @ sigmaminus2 + sigmaminus1 @ sigmaplus2)
    
    if isinstance(omega1, np.ndarray) and isinstance(omega2, np.ndarray):
        omega1s = tuple(omega1)
        omega2s = tuple(omega2)
        hm_total=[]
        def hm_total(t_k):
            hm_sys1=(delta_t*(omega1s[t_k]/2*np.kron(np.eye(d_t),sigmaplus1) + np.conj(omega1s[t_k])/2*np.kron(np.eye(d_t),sigmaminus1))
            +delta_t*delta1*np.kron(np.eye(d_t),e1)) 
            
            hm_sys2=(delta_t*(omega2s[t_k]/2*np.kron(np.eye(d_t),sigmaplus2) + np.conj(omega2s[t_k])/2*np.kron(np.eye(d_t),sigmaminus2))
            +delta_t*delta2* np.kron(np.eye(d_t),e2)) 
           
            return hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L + h_exch
   
    elif isinstance(omega1, np.ndarray):
        omega1s = tuple(omega1)
        hm_sys2=(delta_t*(omega2/2*np.kron(np.eye(d_t),sigmaplus2) + np.conj(omega2)/2*np.kron(np.eye(d_t),sigmaminus2))
        +delta_t*delta2* np.kron(np.eye(d_t),e2))  

        def hm_total(t_k):
            hm_sys1=(delta_t*(omega1s[t_k]/2*np.kron(np.eye(d_t),sigmaplus1) + np.conj(omega1s[t_k])/2*np.kron(np.eye(d_t),sigmaminus1))
            +delta_t*delta1*np.kron(np.eye(d_t),e1)) 
            
            return hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L + h_exch

    elif isinstance(omega2, np.ndarray):
        omega2s = tuple(omega2)
        hm_sys1=(delta_t*(omega1/2*np.kron(np.eye(d_t),sigmaplus1) + np.conj(omega1)/2*np.kron(np.eye(d_t),sigmaminus1))
        +delta_t*delta1*np.kron(np.eye(d_t),e1)) 
        
        def hm_total(t_k):
            hm_sys2=(delta_t*(omega2s[t_k]/2*np.kron(np.eye(d_t),sigmaplus2) + np.conj(omega2s[t_k])/2*np.kron(np.eye(d_t),sigmaminus2))
            +delta_t*delta2* np.kron(np.eye(d_t),e2)) 
             
            return hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L + h_exch
        
    else:
        hm_sys1=(delta_t*(omega1/2*np.kron(np.eye(d_t),sigmaplus1) + np.conj(omega1)/2*np.kron(np.eye(d_t),sigmaminus1))
        +delta_t*delta1*np.kron(np.eye(d_t),e1)) 
     
        hm_sys2=(delta_t*(omega2/2*np.kron(np.eye(d_t),sigmaplus2) + np.conj(omega2)/2*np.kron(np.eye(d_t),sigmaminus2))
        +delta_t*delta2* np.kron(np.eye(d_t),e2)) 
 
        hm_total = hm_sys1 + hm_sys2 + t1R + t1L + t2R + t2L + h_exch
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
    t11 = np.sqrt(gamma_l1)*np.kron(np.kron(delta_b_dag_l(delta_t,d_t_total),np.eye(d_t)),sigmaminus1)    
    t11hc = +np.sqrt(gamma_l1)*np.kron(np.kron(delta_b_l(delta_t,d_t_total),np.eye(d_t)),sigmaplus1)
    t21 = np.sqrt(gamma_r1)*np.kron(np.kron(np.eye(d_t)*np.exp(1j*phase),delta_b_dag_r(delta_t,d_t_total)),sigmaminus1)    
    t21hc = +np.sqrt(gamma_r1)*np.kron(np.kron(np.eye(d_t)*np.exp(-1j*phase),delta_b_r(delta_t,d_t_total)),sigmaplus1)
    t12 = np.sqrt(gamma_l2)*np.kron(np.kron(np.eye(d_t)*np.exp(1j*phase),delta_b_dag_l(delta_t,d_t_total)),sigmaminus2)
    t12hc = +np.sqrt(gamma_l2)*np.kron(np.kron(np.eye(d_t)*np.exp(-1j*phase),delta_b_l(delta_t,d_t_total)),sigmaplus2)  
    t22 = np.sqrt(gamma_r2)*np.kron(np.kron(delta_b_dag_r(delta_t,d_t_total),np.eye(d_t)),sigmaminus2)
    t22hc = +np.sqrt(gamma_r2)*np.kron(np.kron(delta_b_r(delta_t,d_t_total),np.eye(d_t)),sigmaplus2)
     
    if isinstance(omega1, np.ndarray) and isinstance(omega2, np.ndarray):
        omega1s = tuple(omega1)
        omega2s = tuple(omega2)
        def hm_total(t_k):
            hm_sys1 = (delta_t*(omega1s[t_k]/2*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus1) + np.conj(omega1s[t_k])/2*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaminus1))
            +delta_t*delta1*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),e1)) 
            hm_sys2 = (delta_t*(omega2s[t_k]/2*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus2) + np.conj(omega2s[t_k])/2*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaminus2))
            +delta_t*delta2* np.kron(np.kron(np.eye(d_t),np.eye(d_t)),e2)) 
            
            return hm_sys1 + hm_sys2 + t11 + t11hc + t21 + t21hc + t12 + t12hc + t22 + t22hc
   
    elif isinstance(omega1, np.ndarray):
        omega1s = tuple(omega1)
        hm_sys2=(delta_t*(np.kron(omega2/2*np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus2) + np.conj(omega2)/2*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaminus2))
        +delta_t*delta2* np.kron(np.kron(np.eye(d_t),np.eye(d_t)),e2)) 
        
        def hm_total(t_k):
            hm_sys1=(delta_t*(omega1s[t_k]/2*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus1) + np.conj(omega1s[t_k])/2*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaminus1))
            +delta_t*delta1*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),e1)) 
            
            return hm_sys1 + hm_sys2 + t11 + t11hc + t21 + t21hc + t12 + t12hc + t22 + t22hc

    elif isinstance(omega2, np.ndarray):
        omega2s = tuple(omega2)
        hm_sys1=(delta_t*(np.kron(omega1/2*np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus1) + np.conj(omega1)/2*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaminus1))
        +delta_t*delta1*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),e1)) 

        def hm_total(t_k):
            hm_sys2=(delta_t*(omega2s[t_k]/2*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus2) + np.conj(omega2s[t_k])/2*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaminus2))
            +delta_t*delta2* np.kron(np.kron(np.eye(d_t),np.eye(d_t)),e2)) 
             
            return hm_sys1 + hm_sys2 + t11 + t11hc + t21 + t21hc + t12 + t12hc + t22 + t22hc
        
    else:
        hm_sys1=(delta_t*(omega1/2*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus1) + np.conj(omega1/2)*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaminus1))
        +delta_t*delta1*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),e1)) 
        
        hm_sys2=(delta_t*(omega2/2*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaplus2) + np.conj(omega2/2)*np.kron(np.kron(np.eye(d_t),np.eye(d_t)),sigmaminus2))
        +delta_t*delta2* np.kron(np.kron(np.eye(d_t),np.eye(d_t)),e2)) 
        
        hm_total=hm_sys1 + hm_sys2 + t11 + t11hc + t21 + t21hc + t12 + t12hc + t22 + t22hc
    return hm_total



# Fix style/standardization of below functions
def basicSigmaPlus(dim):
    return np.diag(np.ones(dim-1, dtype=complex), -1)

def sigmaPlus(dimList, index):
    identityDim1 = np.prod(dimList[:index], dtype=np.uint)
    identityDim2 = np.prod(dimList[index+1:], dtype=np.uint)
    return np.kron(np.kron(np.eye(identityDim1), basicSigmaPlus(dimList[index])), np.eye(identityDim2))

def hamN2LSChiral(params:InputParams, gammas=None, detuningLs=None, phases=None, omegaPumps=None):
    delta_t,d_sys_total, d_t_total = params.delta_t,params.d_sys_total,params.d_t_total
    tlsNum = len(d_sys_total)
   
    if gammas is None: gammas = np.ones(tlsNum)
    if detuningLs is None: detuningLs = np.zeros(tlsNum)
    if phases is None: phases = np.zeros(tlsNum)
    else: phases = np.insert(phases,0,0)
    if omegaPumps is None: omegaPumps = np.zeros(tlsNum)
    
    dTime = np.prod(d_t_total)
    deltaB = delta_b(delta_t, dTime)
    deltaBDag = np.conj(deltaB).T
    dSys = 2
    
    dimList = (np.ones(tlsNum) * dSys).astype(int)
    pumpCoeffs = np.array(omegaPumps)
    
    H = np.zeros((dTime**tlsNum * np.prod(dimList),)*2, dtype=complex)
    for i in range(tlsNum):
        sigmaP = sigmaPlus(dimList, i)
        
        # Baking the time bin identities for the tensor spaces left of the system bin to the sys ops
        sigmaP = np.kron(np.eye(dTime**(i)), sigmaP)
        sigmaM = sigmaP.T
    
        h2LS = delta_t * detuningLs[i]  * np.kron(np.eye(dTime), sigmaP @ sigmaM)                             
        hPump = delta_t * pumpCoeffs[i] * np.kron(np.eye(dTime), sigmaP + sigmaM)
        hInt = np.sqrt(gammas[i]) * ( np.exp(1j*phases[i])*np.kron(deltaB, sigmaP)
                                      + np.exp(-1j*phases[i])*np.kron(deltaBDag, sigmaM))
        
        Hlocal = h2LS + hPump + hInt
        
        Hlocal = np.kron(np.eye(dTime**(tlsNum-i-1)), Hlocal)
        H += Hlocal
        
    return H

#TODO Add in pump logic
def hamN2LS(params:InputParams, gamma_ls=None, gamma_rs=None, detuningLs=None, phases=None, omegaPumps=None):
    delta_t,d_sys_total, d_t_total = params.delta_t,params.d_sys_total,params.d_t_total
    tlsNum = len(d_sys_total)

    if gamma_ls is None: gamma_ls = np.ones(tlsNum)*0.5
    if gamma_rs is None: gamma_rs = np.ones(tlsNum)*0.5
    if detuningLs is  None: detuningLs = np.zeros(tlsNum)
    if phases is None: phases = np.zeros(tlsNum) 
    else: phases = np.insert(phases,0,0)
    if omegaPumps is None: omegaPumps = np.zeros(tlsNum)
    
    dTime = np.prod(d_t_total)

    delta_br = delta_b_r(delta_t, d_t_total)
    delta_bl = delta_b_l(delta_t, d_t_total)
    dSys = 2
    
    sys_dim_list = (np.ones(tlsNum) * dSys).astype(int)
    time_dim_list = (np.ones(tlsNum) * dTime).astype(int)
    omegaPumps = np.array(omegaPumps)
    H = np.zeros((np.prod(time_dim_list) * np.prod(sys_dim_list),)*2, dtype=complex)
    
    deltaBr_list = []
    deltaBDagr_list = []
    deltaBl_list = []
    deltaBDagl_list = []

    for i in range(tlsNum):
        br = extend_op(delta_br, time_dim_list, i, reverse_dims=True)
        bl = extend_op(delta_bl, time_dim_list, i, reverse_dims=True)
        deltaBr_list.append(br)
        deltaBl_list.append(bl)

        deltaBDagr_list.append(np.conj(br).T)
        deltaBDagl_list.append(np.conj(bl).T)
    
    total_time_eye = np.eye(np.prod(time_dim_list))
    
    for i in range(tlsNum):
        sigmaP = extend_op(sigmaplus(), sys_dim_list, i, reverse_dims=True)
        sigmaM = sigmaP.T
        phase_ind_l = (i+1) % tlsNum

        h2LS = delta_t * detuningLs[i]  * np.kron(total_time_eye, sigmaP @ sigmaM)                             
        hPump = delta_t * omegaPumps[i] * np.kron(total_time_eye, sigmaP + sigmaM)
        hInt_r = np.sqrt(gamma_rs[i]) * (np.exp(1j*phases[i]) * np.kron(deltaBr_list[i], sigmaP)
                                    + np.exp(-1j*phases[i]) * np.kron(deltaBDagr_list[i], sigmaM))
        hInt_l = np.sqrt(gamma_ls[i]) * (np.exp(1j*phases[phase_ind_l]) * np.kron(deltaBl_list[tlsNum-1-i], sigmaP)
                                    + np.exp(-1j*phases[phase_ind_l]) * np.kron(deltaBDagl_list[tlsNum-1-i], sigmaM))

        Hlocal = h2LS + hPump + hInt_r + hInt_l
        H += Hlocal
        
    return H


def hamiltonian_1tls_chiral(params:InputParams, gamma=1, phase=None):
    delta_t,d_sys_total, d_t_total = params.delta_t,params.d_sys_total,params.d_t_total

    if phase == None: phase = 0

    sigmaP = sigmaplus()
    sigmaM = sigmaminus()
    d_t = np.prod(d_t_total)
    return np.sqrt(gamma) * (np.kron(np.exp(1j*phase)*delta_b(delta_t, d_t), sigmaP) \
        + np.exp(-1j*phase)* np.kron(delta_b_dag(delta_t, d_t), sigmaM))


# Sym case, returns LIST of local hamiltonians
def hamiltonian_Ntls_sym_eff(params:InputParams, gamma_ls:list[float], gamma_rs:list[float], phases=None):
    delta_t,d_sys_total, d_t_total = params.delta_t,params.d_sys_total,params.d_t_total
    helper_obj = Symmetrical_Coupling_Helper(d_sys_total)
    if phases is None: phases = np.zeros(len(d_sys_total))
    else: phases = np.insert(phases,0,0)

    delta_b_dag_l_single = delta_b_dag_l(delta_t, d_t_total)
    delta_b_l_single = delta_b_l(delta_t, d_t_total)
    delta_b_dag_r_single = delta_b_dag_r(delta_t, d_t_total)
    delta_b_r_single = delta_b_r(delta_t, d_t_total)
    d_t_eye = np.eye(params.d_t)
    sys_eye = np.eye(2)

    delta_b_dag_l_0 = np.kron(delta_b_dag_l_single, d_t_eye)
    delta_b_dag_l_1 = np.kron(d_t_eye, delta_b_dag_l_single)

    delta_b_l_0 = np.kron(delta_b_l_single, d_t_eye)
    delta_b_l_1 = np.kron(d_t_eye, delta_b_l_single)

    delta_b_dag_r_0 = np.kron(delta_b_dag_r_single, d_t_eye)
    delta_b_dag_r_1 = np.kron(d_t_eye, delta_b_dag_r_single)

    delta_b_r_0 = np.kron(delta_b_r_single, d_t_eye)
    delta_b_r_1 = np.kron(d_t_eye, delta_b_r_single)

    sigmap = sigmaplus()
    sigmam = sigmaminus()

    N = len(d_sys_total)

    hams = [None] * helper_obj.interaction_num

    for i in range(int(helper_obj.sys_num/2)):
        sys_ind_r = helper_obj.ordered_indices[2*i]
        sys_ind_l = helper_obj.ordered_indices[2*i+1]

        phase_ind_r2 = (sys_ind_r+1) % N
        phase_ind_l2 = (sys_ind_l+1) % N


        ham = np.sqrt(gamma_rs[sys_ind_r])*(np.exp(-1j*phases[sys_ind_r])*np.kron(delta_b_dag_r_1, np.kron(sys_eye,sigmam))
                            + np.exp(1j*phases[sys_ind_r])*np.kron(delta_b_r_1, np.kron(sys_eye,sigmap)))
        ham += np.sqrt(gamma_ls[sys_ind_r])*(np.exp(-1j*phases[phase_ind_r2])*np.kron(delta_b_dag_l_0, np.kron(sys_eye,sigmam))
                            + np.exp(1j*phases[phase_ind_r2])*np.kron(delta_b_l_0, np.kron(sys_eye,sigmap)))

        ham += np.sqrt(gamma_rs[sys_ind_l])*(np.exp(-1j*phases[sys_ind_l])*np.kron(delta_b_dag_r_0, np.kron(sigmam, sys_eye)) 
                          + np.exp(1j*phases[sys_ind_l])*np.kron(delta_b_r_0, np.kron(sigmap, sys_eye)))
        ham += np.sqrt(gamma_ls[sys_ind_l])*(np.exp(-1j*phases[phase_ind_l2])*np.kron(delta_b_dag_l_1, np.kron(sigmam,sys_eye)) 
                           + np.exp(1j*phases[phase_ind_l2])*np.kron(delta_b_l_1, np.kron(sigmap,sys_eye)))

        hams[i] = ham

    # Final hamiltonian coupling single emitter to single time bin
    if helper_obj.odd_end:
        sys_ind = helper_obj.ordered_indices[-1]
        phase_ind_2 = (sys_ind+1) % N

        ham = np.sqrt(gamma_rs[sys_ind])*(np.exp(1j*phases[sys_ind])*np.kron(delta_b_r_single, sigmap)
                         + np.exp(-1j*phases[sys_ind])*np.kron(delta_b_dag_r_single, sigmam))
        
        ham += np.sqrt(gamma_ls[sys_ind])*(np.exp(1j*phases[phase_ind_2])*np.kron(delta_b_l_single, sigmap)
                        + np.exp(-1j*phases[phase_ind_2])*np.kron(delta_b_dag_l_single, sigmam))

        hams[-1] = ham

    return hams