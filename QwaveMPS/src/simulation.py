#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the simulations to evolve the systems 
and calculate the main observables.

It provides time-evolution routines (Markovian and non-Markovian) for systems
coupled to a 1D field, together with observable
calculations (populations, correlations, spectra and entanglement).

It requires the module ncon (pip install --user ncon)

"""


import numpy as np
import copy
from ncon import ncon
from scipy.linalg import svd,norm
from . import states as states
from collections.abc import Iterator
from QwaveMPS.src.parameters import *
from typing import Callable, TypeAlias
from QwaveMPS.src.hamiltonians import Hamiltonian
from QwaveMPS.src.operators import *

# -----------------------------------
# Singular Value Decomposition helper
# -----------------------------------

def _svd_tensors(tensor:np.ndarray, bond:int, d_1:int, d_2:int) -> np.ndarray:
    """
    Perform a SVD, reshape the tensors and return left tensor, 
    normalized Schmidt vector, and right tensor.

    Parameters
    ----------
    tensor : ndarray
        tensor to decompose
    
    bond : int
        max. bond dimension
    
    d_1 : int
        physical dimension of first tensor
    
    d_2 : int
        physical dimension of second tensor

    Returns
    -------
    u : ndarray
        left normalized tensor

    s_norm : ndarray
        smichdt coefficients normalized 
    
    vt : ndarray
        transposed right normalized tensor
    """
    u, s, vt = svd(tensor.reshape(tensor.shape[0]*d_1, tensor.shape[-1]*d_2), full_matrices=False)
    chi = min(bond, len(s))
    epsilon = 1e-12 #to avoid dividing by zero
    s_norm = s[:chi] / (norm(s[:chi])+ epsilon)
    u = u[:, :chi].reshape(tensor.shape[0],d_1,chi)
    vt = vt[:chi, :].reshape(chi,d_2,tensor.shape[-1])
    return u, s_norm, vt

# ------------------------------------------------------
# Time evolution: Markovian and non-Markovian evolutions
# ------------------------------------------------------

def t_evol(ham:Hamiltonian, i_s0:np.ndarray, i_n0:np.ndarray,params:InputParams) -> Bins:
    """ 
    Appropriate time evolution of the system (chooses Markovian or non-markovian based on presence of delay times).
    
    Parameters
    ----------
    ham : ndarray or callable
        Either a fixed evolution operator/tensor or a callable returning the
        evolution operator for time-step k: ham(k).
        
     i_s0 : ndarray
         Initial system bin (tensor).
         
     i_n0: ndarray 
         Initial field bin.
         Seed for the input time-bin generator.

     params:InputParams
         Class containing the input parameters
         (contains delta_t, tmax, bond, d_t_total, d_sys_total, tau.).

    Returns
    -------
    Bins:  Dataclass (from parameters.py) 
        containing:
          - sys_b: list of system bins
          - time_b: list of time bins
          - tau_b: list of feedback bins 
          - cor_b: list of tensors used for correlations
          - schmidt, schmidt_tau: lists of Schmidt coefficient arrays
    """
    if len(params.tau) == 0 or params.tau is None:
        return t_evol_mar(ham, i_s0, i_n0, params)
    else:
        return t_evol_nmar(ham, i_s0, i_n0, params)

def t_evol_mar(ham:Hamiltonian, i_s0:np.ndarray, i_n0:np.ndarray, params:InputParams) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """ 
    Time evolution of the system without delay times (Markovian regime)
    
    Parameters
    ----------
    ham : ndarray or callable
        Either a fixed evolution operator/tensor or a callable returning the
        evolution operator for time-step k: ham(k).
    
    i_s0 : ndarray
        Initial system bin (tensor).
        
    i_n0: ndarray 
        Initial field bin.
        Seed for the input time-bin generator.
        
    params:InputParams
        Class containing the input parameters
        (contains delta_t, tmax, bond, d_t_total, d_sys_total).

    Returns
    -------
    Bins:  Dataclass (from parameters.py) 
        containing:
            - sys_b: list of system bins
            - time_b: list of time bins
            - cor_b: list of tensors used for correlations
            - schmidt: list of Schmidt coefficient arrays (for entanglement calculation)
    """
    
    delta_t = params.delta_t
    tmax=params.tmax
    bond=params.bond_max
    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total

    
    d_t=np.prod(d_t_total)
    d_sys=np.prod(d_sys_total)
    n=int(tmax/delta_t)
    t_k=0
    i_s=i_s0
    sbins=[] 
    sbins.append(i_s0)
    tbins=[]
    tbins.append(states.wg_ground(d_t))
    schmidt=[]
    schmidt.append(np.zeros(1))
    tbins_in = []
    tbins_in.append(states.wg_ground(d_t))
    if not callable(ham):
        evol=u_evol(ham,d_sys,d_t)
    swap_sys_t=swap(d_sys,d_t)
    input_field=states.input_state_generator(d_t_total, i_n0)
    cor_list=[]
    for k in range(n):   
        i_nk = next(input_field)   
        if callable(ham):
            evol=u_evol(ham(k),d_sys,d_t)

        # Put OC in input bin to calculate input field observables
        phi1 = ncon([i_s, i_nk], [[-1,-2,1], [1,-3,-4]])
        i_s, stemp, i_nk = _svd_tensors(phi1, bond, d_sys, d_t)
        i_nk = stemp[:,None,None] * i_nk # OC in input bin
        tbins_in.append(i_nk)

        phi1=ncon([i_s,i_nk,evol],[[-1,2,3],[3,4,-4],[-2,-3,2,4]]) #system bin, time bin + u operator contraction  
        i_s,stemp,i_n=_svd_tensors(phi1, bond,d_sys,d_t)
        i_s=i_s*stemp[None,None,:] #OC system bin
        sbins.append(i_s)
        tbins.append(stemp[:,None,None]*i_n)
                    
        phi2=ncon([i_s,i_n,swap_sys_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #system bin, time bin + swap contraction
        i_n,stemp,i_st=_svd_tensors(phi2, bond,d_t,d_sys)
        i_s=stemp[:,None,None]*i_st   #OC system bin
        t_k += delta_t
        
        schmidt.append(stemp)
        
        if k < (n-1):
            cor_list.append(i_n)
        if k == n-1:
            cor_list.append(ncon([i_n,np.diag(stemp)],[[-1,-2,1],[1,-3]]))
        
    return Bins(system_states=sbins,output_field_states=tbins, input_field_states=tbins_in,
                correlation_bins=cor_list,schmidt=schmidt)

def t_evol_nmar(ham:Hamiltonian, i_s0:np.ndarray, i_n0:np.ndarray,params:InputParams) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """ 
    Time evolution of the system with finite delays/feedback (non-Markovian regime).
    Requires tau to be at least delta_t.
    
    Parameters
    ----------
    ham : ndarray or callable
        Either a fixed evolution operator/tensor or a callable returning the
        evolution operator for time-step k: ham(k).
        
     i_s0 : ndarray
         Initial system bin (tensor).
         
     i_n0: ndarray 
         Initial field bin.
         Seed for the input time-bin generator.

     params:InputParams
         Class containing the input parameters
         (contains delta_t, tmax, bond, d_t_total, d_sys_total, tau.).

    Returns
    -------
    Bins:  Dataclass (from parameters.py) 
        containing:
          - sys_b: list of system bins
          - time_b: list of time bins
          - tau_b: list of feedback bins 
          - cor_b: list of tensors used for correlations
          - schmidt, schmidt_tau: lists of Schmidt coefficient arrays
    """
    delta_t = params.delta_t
    tmax=params.tmax
    bond=params.bond_max
    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total
    tau=params.tau
    
    d_t=np.prod(d_t_total)
    d_sys=np.prod(d_sys_total)
    sbins=[] 
    tbins=[]
    tbins_in = []
    taubins=[]
    nbins=[]
    cor_list=[]
    schmidt=[]
    schmidt_tau=[]
    sbins.append(i_s0)   
    tbins.append(states.wg_ground(d_t))
    tbins_in.append(states.wg_ground(d_t))
    taubins.append(states.wg_ground(d_t))
    schmidt.append(np.zeros(1))
    schmidt_tau.append(np.zeros(1))
    input_field=states.input_state_generator(d_t_total, i_n0)
    n=int(round(tmax/delta_t,0))
    t_k=0
    t_0=0
    if not callable(ham):
        evol=u_evol(ham,d_sys,d_t,2) #Feedback loop means time evolution involves an input and a feedback time bin. Can generalize this later, leaving 2 for now so it runs.
    swap_t_t=swap(d_t,d_t)
    swap_sys_t=swap(d_sys,d_t)
    l=int(round(tau/delta_t,0)) #time steps between system and feedback
    
    for i in range(l):
        nbins.append(states.wg_ground(d_t))
        t_0+=delta_t
    
    i_stemp=i_s0      
    
    for k in range(n):   
        #swap of the feedback until being next to the system
        i_tau= nbins[k] #starting from the feedback bin
        for i in range(k,k+l-1): 
            i_n=nbins[i+1] 
            swaps=ncon([i_tau,i_n,swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) 
            i_n2,stemp,i_t=_svd_tensors(swaps,bond,d_t,d_t)
            i_tau = ncon([np.diag(stemp),i_t],[[-1,1],[1,-3,-4]]) 
            nbins[i]=i_n2 
            
        #Make the system bin the OC
        i_1=ncon([i_tau,i_stemp],[[-1,-2,1],[1,-3,-4]]) #feedback-system contraction
        i_t,stemp,i_stemp=_svd_tensors(i_1, bond,d_t,d_sys)
        i_s=stemp[:,None,None]*i_stemp #OC system bin
        
        i_nk = next(input_field)                
        if callable(ham):
            evol=u_evol(ham(k),d_sys,d_t, 2)

        # Put OC in input bin to calculate input field observables
        phi1 = ncon([i_s, i_nk], [[-1,-2,1], [1,-3,-4]])
        i_s, stemp, i_nk = _svd_tensors(phi1, bond, d_sys, d_t)
        i_nk = stemp[:,None,None] * i_nk # OC in input bin
        tbins_in.append(i_nk)

        #now contract the 3 bins and apply u, followed by 2 svd to recover the 3 bins 
        phi1=ncon([i_t,i_s,i_nk,evol],[[-1,3,1],[1,4,2],[2,5,-5],[-2,-3,-4,3,4,5]]) #tau bin, system bin, future time bin + u operator contraction
        i_t,stemp,i_2=_svd_tensors(phi1, bond,d_t,d_t*d_sys)
        i_2=stemp[:,None,None]*i_2
        i_stemp,stemp,i_n=_svd_tensors(i_2, bond,d_sys,d_t)
        i_s = i_stemp*stemp[None,None,:]
        sbins.append(i_s) 
        
        #swap system and i_n
        phi2=ncon([i_s,i_n,swap_sys_t],[[-1,3,2],[2,4,-4],[-2,-3,3,4]]) #system bin, time bin + swap contraction
        i_n,stemp,i_stemp=_svd_tensors(phi2, bond,d_sys,d_t)   
        i_n=i_n*stemp[None,None,:] #the OC in time bin     
        
        cont= ncon([i_t,i_n],[[-1,-2,1],[1,-3,-4]]) 
        i_t,stemp,i_n=_svd_tensors(cont, bond,d_t,d_t)   
        i_tau = i_t*stemp[None,None,:] #OC in feedback bin     
        tbins.append(stemp[:,None,None]*i_n)
        
        #feedback bin, time bin contraction
        taubins.append(i_tau) 
        nbins[k+l-1]=i_tau #update of the feedback bin
        nbins.append(i_n)         
        t_k += delta_t
        schmidt.append(stemp) #storing the Schmidt coeff for calculating the entanglement

        #swap back of the feedback bin      
        for i in range(k+l-1,k,-1): #goes from the last time bin to first one
            i_n=nbins[i-1] #time bin
            swaps=ncon([i_n,i_tau,swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #time bin, feedback bin + swap contraction
            i_t,stemp,i_n2=_svd_tensors(swaps, bond,d_t,d_t)   
            i_tau = i_t*stemp[None,None,:] #OC tau bin         
            nbins[i]=i_n2    #update nbins  
        schmidt_tau.append(stemp)  
        if k<(n-1):         
            nbins[k+1] = stemp[:,None,None]*i_n2 #new tau bin for the next time step       
            cor_list.append(i_t)
        if k == n-1:
            cor_list.append(i_t*stemp[None,None,:])   
            
    return Bins(system_states=sbins,loop_field_states=tbins,output_field_states=taubins,
                input_field_states=tbins_in,correlation_bins=cor_list,
                schmidt=schmidt,schmidt_tau=schmidt_tau)

