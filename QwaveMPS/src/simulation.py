#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the simulations to evolve the systems 
and calculate the main observables.

Irovides time-evolution routines (Markovian and non-Markovian) for systems
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
    if len(params.tau) == 0 or params.tau == None:
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
    Time evolution of the system with finite delays/feedback (non-Markovian regime)
    
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

# ---------------------------------------------------------------
# Observables: populations, entanglement, spectrum
# ---------------------------------------------------------------
def loop_integrated_statistics(time_dependent_func:np.ndarray[complex], params:InputParams) -> np.ndarray:
    """
    Calculates the main population dynamics for a single TLS in a semi-infinite waveguide,
    with non-Markovian feedback (one channel).

    Parameters
    ----------
    bins : Bins
        Bins returned by t_evol_nmar
    
    params : InputParams
        Simulation parameters

    Returns
    -------
    Pop1Channel: Dataclass
         containing:
            - pop: TLS population
            - tbins: photon flux per time bin
            - trans: integrated transmitted flux
            - loop: feedback-loop photon count
            - total: total excitations at each time
    """

    tau = params.tau
    delta_t = params.delta_t

    n=len(time_dependent_func) 
    observable_integrated_in_loop = np.zeros(n,dtype=complex)
    
    l=int(round(tau/delta_t,0))

    cumulative_sum = np.cumsum(time_dependent_func)
    observable_integrated_in_loop[:l+1] = cumulative_sum[:l+1]
    observable_integrated_in_loop[l:] = cumulative_sum[l:] - cumulative_sum[:-l]
    
    return observable_integrated_in_loop * delta_t

def entanglement(sch:list[np.ndarray]) -> list[float]:
    """
    Compute von Neumann entanglement entropy across a list of Schmidt coefficient arrays.

    Parameters
    ----------
    sch : list[np.ndarray]
        List of Schmidt coefficient arrays (s) for each bipartition.

    Returns
    -------
    list[float]
        Entanglement entropies computed as -sum(p * log2 p) where p = s**2.
    """
    ent_list=[]
    for s in sch:
        sqrd_sch=s**2   
        sqrd_sch=np.trim_zeros(sqrd_sch) 
        log_sqrd_sch=np.log2(sqrd_sch)
        prod=sqrd_sch*log_sqrd_sch
        ent=-sum(prod)
        ent_list.append(ent)
    return ent_list

def spectrum_w(delta_t:float, g1_list: np.ndarray) -> list[np.ndarray, np.ndarray]:
    """
    Compute the (discrete) spectrum in the long-time limit via Fourier transform 
    of the two-time first-order correlation (steady-state solution).

    Parameters
    ----------
    delta_t : float
        Time step used in the simulation; used to set frequency sampling.
   
    g1_list : np.ndarray
        Steady-state first order correlation.

    Returns
    -------
    s_w : np.ndarray
        Spectrum in the long-time limit (steady state solution)
    wlist : np.ndarray
        Corresponding frequency list.
    """
    s_w = np.fft.fftshift(np.fft.fft(g1_list))
    n=s_w.size
    wlist = np.fft.fftshift(np.fft.fftfreq(n,d=delta_t))*2*np.pi   
    return s_w,wlist

# ----------------------
# Two time point Correlation functions
# ----------------------

def correlation_2op_2t(correlation_bins:list[np.ndarray], a_op_list:list[np.ndarray], b_op_list:list[np.ndarray], params:InputParams, completion_print_flag:bool=True) -> list[np.ndarray]|np.ndarray:
    """ 
    Calculates the two time correlation function <A(t)B(t+tau)> for each A/B in a_op_list/b_op_list.
            
    Parameters
    ----------
    correlation_bins : list[ndarray]
        List of time bins with the OC in either the initial or final bin in the list.
    
    a_op_list : [ndarray]
        List of operators of which correlation functions should be calculated in the case that tau=0 (same time). These should exist in a single time-bin tensor space.

    ops_two_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that tau > 0. These should be ordered in a corresponding order to
        ops_same_time and should exist in a tensor space that is the outer product of two time bin tensor spaces, with the right space corresponding to the greater time.

    params : InputParams
        Simulation parameters 

    Returns
    -------
    result : [np.ndarray]
        List of 2D arrays, each a two time correlation function corresponding by index to the operators in ops_same_time and ops_two_time.
        The two time correlation function is stored as f[t,tau], with non-negative tau and time increments between points given by the simulation.
    """
    list_flag = op_list_check(a_op_list)

    if list_flag and len(a_op_list) != len(b_op_list):
        raise ValueError("Lengths of operators lists are not equals")
    
    ops_same_time = []; ops_two_time = []

    if list_flag:
        for i in range(len(a_op_list)):
            ops_same_time.append(a_op_list[i] @ b_op_list[i])
            ops_two_time.append(np.kron(a_op_list[i], b_op_list[i]))
    else:
        ops_same_time.append(a_op_list @ b_op_list)
        ops_two_time.append(np.kron(a_op_list, b_op_list))
    
    results, t_list =  correlations_2t(correlation_bins, ops_same_time, ops_two_time, params, completion_print_flag=completion_print_flag)

    if not list_flag:
        results = results[0]

    return results, t_list

def correlation_4op_2t(correlation_bins:list[np.ndarray], a_op_list:list[np.ndarray], b_op_list:list[np.ndarray], c_op_list:list[np.ndarray], d_op_list:list[np.ndarray], params:InputParams, completion_print_flag:bool=True) -> list[np.ndarray]|np.ndarray:
    """ 
    Calculates the two time correlation function <A(t)B(t+tau)C(t+tau)D(t)> for each operator in the respective lists.
            
    Parameters
    ----------
    correlation_bins : list[ndarray]
        List of time bins with the OC in either the initial or final bin in the list.
    
    a_op_list : [ndarray]
        List of operators of which correlation functions should be calculated in the case that tau=0 (same time). These should exist in a single time-bin tensor space.

    ops_two_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that tau > 0. These should be ordered in a corresponding order to
        ops_same_time and should exist in a tensor space that is the outer product of two time bin tensor spaces, with the right space corresponding to the greater time.

    params : InputParams
        Simulation parameters 

    Returns
    -------
    result : [np.ndarray]
        List of 2D arrays, each a two time correlation function corresponding by index to the operators in ops_same_time and ops_two_time.
        The two time correlation function is stored as f[t,tau], with non-negative tau and time increments between points given by the simulation.
    """
    list_flag = op_list_check(a_op_list)

    if list_flag and not (len(a_op_list) == len(b_op_list) == len(c_op_list) == len(d_op_list)):
        raise ValueError("Lengths of operators lists are not equal")


    ops_same_time = []; ops_two_time = []

    if list_flag:
        for i in range(len(a_op_list)):
            ops_same_time.append(a_op_list[i] @ b_op_list[i] @ c_op_list[i] @ d_op_list[i])
            ops_two_time.append(np.kron(a_op_list[i] @ d_op_list[i], b_op_list[i] @ c_op_list[i]))
    else:
        ops_same_time.append(a_op_list @ b_op_list @ c_op_list @ d_op_list)
        ops_two_time.append(np.kron(a_op_list @ d_op_list, b_op_list @ c_op_list))
    
    results, t_list = correlations_2t(correlation_bins, ops_same_time, ops_two_time, params, completion_print_flag=completion_print_flag)

    # Don't return as list
    if not list_flag:
        results = results[0]
    return results, t_list


def correlation_2op_1t(correlation_bins:list[np.ndarray], a_op_list:list[np.ndarray], b_op_list:list[np.ndarray], params:InputParams):
    """ 
    Calculates the two time correlation function <A(t)B(t+tau)C(t+tau)D(t)> for each operator in the respective lists.
            
    Parameters
    ----------
    correlation_bins : list[ndarray]
        List of time bins with the OC in either the initial or final bin in the list.
    
    a_op_list : [ndarray]
        List of operators of which correlation functions should be calculated in the case that tau=0 (same time). These should exist in a single time-bin tensor space.

    ops_two_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that tau > 0. These should be ordered in a corresponding order to
        ops_same_time and should exist in a tensor space that is the outer product of two time bin tensor spaces, with the right space corresponding to the greater time.

    params : InputParams
        Simulation parameters 

    Returns
    -------
    result : [np.ndarray]
        List of 2D arrays, each a two time correlation function corresponding by index to the operators in ops_same_time and ops_two_time.
        The two time correlation function is stored as f[t,tau], with non-negative tau and time increments between points given by the simulation.
    """
    

def correlations_2t(correlation_bins:list[np.ndarray], ops_same_time:list[np.ndarray], ops_two_time:list[np.ndarray], params:InputParams, oc_end_list_flag:bool=True, completion_print_flag:bool=False) -> tuple[list[np.ndarray], np.ndarray]:
    """ 
    General two-time correlation calculator.
    Take in list of time ordered normalized (with OC) time bins at position of relevance.
    Calculate a list of arbitrary two time point correlation functions at t and t+tau for nonnegative tau. 
        
    Parameters
    ----------
    time_bin_list : [ndarray]
        List of time bins with the OC in either the initial or final bin in the list.
    
    ops_same_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that tau=0 (same time). These should exist in a single time-bin tensor space.

    ops_two_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that tau > 0. These should be ordered in a corresponding order to
        ops_same_time and should exist in a tensor space that is the outer product of two time bin tensor spaces, with the right space corresponding to the greater time.

    params : InputParams
        Simulation parameters 

    oc_end_list_flag : bool, default=True 
        Leaves the chain of bins unchanged for the calculation if the OC is at the end of the chain. Else assumes it is at the beginning and moves it to the end at start of calculation.
       
    completion_print_flag : bool, default=True
        Flag to print completion loop number percent of the calculation (note this is not the percent completion, and later loops complete faster than earlier ones). 

    Returns
    -------
    result : [np.ndarray]
        List of 2D arrays, each a two time correlation function corresponding by index to the operators in ops_same_time and ops_two_time.
        The two time correlation function is stored as f[t,tau], with non-negative tau and time increments between points given by the simulation.
    """
    d_t_total=params.d_t_total
    bond=params.bond_max
    d_t=np.prod(d_t_total)
    
    time_bin_list_copy = copy.deepcopy(correlation_bins)
    swap_matrix = swap(d_t, d_t)
    
    # Resize two_time_ops if needed
    for i in range(len(ops_two_time)):
        ops_two_time[i] = ops_two_time[i].reshape((d_t,)*(2*2))
    
    correlations = [np.zeros((len(time_bin_list_copy), len(time_bin_list_copy)), dtype=complex) for i in ops_two_time]
    
    # If the OC is at end of the time bin list, move it to the start (shifts OC from one end to other, index 0)
    if oc_end_list_flag:
        for i in range(len(time_bin_list_copy)-1,0,-1):
            bin_contraction = ncon([time_bin_list_copy[i-1],time_bin_list_copy[i]],[[-1,-2,1],[1,-3,-4]])
            left_bin, stemp, right_bin = _svd_tensors(bin_contraction, bond, d_t, d_t)
            time_bin_list_copy[i] = right_bin #right normalized system bin    
            time_bin_list_copy[i-1]= left_bin * stemp[None,None,:] #OC on left bin
    
    # Loop over to fill in correlation matrices values
    print('Correlation Calculation Completion:')
    loop_num = len(time_bin_list_copy) - 1
    print_rate = max(round(loop_num / 100.0), 1)
    for i in range(len(time_bin_list_copy)-1):

        i_1=time_bin_list_copy[0]
        i_2=time_bin_list_copy[1] 
        
        #for the first raw (tau=0)
        for k in range(len(correlations)):
            correlations[k][i,0] = expectation_1bin(i_1, ops_same_time[k]) #this means I'm storing [t,tau] 
        
        #for the rest of the rows (column by column)
        for j in range(len(time_bin_list_copy)-1):       
            state=ncon([i_1,i_2],[[-1,-2,1],[1,-3,-4]]) 
            for k in range(len(correlations)):
                correlations[k][i,j+1] = expectation_nbins(state, ops_two_time[k]) #this means I'm storing [t,tau] 

            
            swapped_tensor=ncon([i_1,i_2,swap_matrix],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #swapping the time bin down the line
            i_t2, stemp, i_t1 = _svd_tensors(swapped_tensor, bond, d_t, d_t)

            i_1 = stemp[:,None,None] * i_t1 #OC tau bin            

            if j < (len(time_bin_list_copy)-2):                
                i_2=time_bin_list_copy[j+2] #next time bin for the next correlation
                time_bin_list_copy[j]=i_t2 #update of the increasing bin
            if j == len(time_bin_list_copy)-2:
                time_bin_list_copy[j]=i_t2
                time_bin_list_copy[j+1]= i_1
        
        #after the last value of the column we bring back the first time
        for j in range(len(time_bin_list_copy)-1,0,-1):            
            swapped_tensor=ncon([time_bin_list_copy[j-1],time_bin_list_copy[j],swap_matrix],[[-1,5,2],[2,6,-4],[-2,-3,5,6]])
            returning_bin, stemp, right_bin = _svd_tensors(swapped_tensor, bond, d_t, d_t)
            if j>1:
                #timeBinListCopy[j] = vt[range(chi),:].reshape(chi,dTime,timeBinListCopy[i].shape[-1]) #right normalized system bin    
                time_bin_list_copy[j] = right_bin #right normalized system bin    
                time_bin_list_copy[j-1]= returning_bin * stemp[None,None,:] #OC on left bin
            # Final iteration drop the returning bin
            if j == 1:
               time_bin_list_copy[j] = stemp[:,None,None] * right_bin
        time_bin_list_copy=time_bin_list_copy[1:]    #Truncating the start of the list now that are done with that bin (t=i)
        
        if i % print_rate == 0 and completion_print_flag == True:
            print(round((float(i)/loop_num)*100,2), '%')
    
    t_list = np.arange(len(correlation_bins)) * params.delta_t
    return correlations, t_list

def correlations_1t(correlation_bins:list[np.ndarray], ops_same_time:list[np.ndarray], ops_two_time:list[np.ndarray], params:InputParams) -> list[np.ndarray]:
    """ 
    General two-time correlation calculator along a single axis.
    Take in list of time ordered normalized (with OC) time bins at position of relevance.
    Calculate a list of arbitrary two time point correlation functions at t and t+tau for nonnegative tau. 
        
    Parameters
    ----------
    time_bin_list : [ndarray]
        List of time bins with the OC in either the initial or final bin in the list.
    
    ops_same_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that tau=0 (same time). These should exist in a single time-bin tensor space.

    ops_two_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that tau > 0. These should be ordered in a corresponding order to
        ops_same_time and should exist in a tensor space that is the outer product of two time bin tensor spaces, with the right space corresponding to the greater time.

    params : InputParams
        Simulation parameters 

    Returns
    -------
    result : [np.ndarray]
        List of 1D arrays, each a two time correlation function corresponding by index to the operators in ops_same_time and ops_two_time.
        The two time correlation function is stored as f[t,tau], with non-negative tau and time increments between points given by the simulation.
    """
    return


#-------------------------------------------
#Steady-state index helper, and correlations
#-------------------------------------------

def steady_state_index(pop:list,window: int=10, tol: float=1e-5) -> int|None:
    """
    Steady-state index helper function to find the time step 
    when the steady state is reached in the population dynamics.
    
    Parameters
    ----------
    pop : list
        List of population values
    
    window : int, default: 10
        Number of recent points to analyze
    
    tol : float, default: 1e-5
        Maximum deviation allowed in the final window
    
    Returns
    -------
    int or None
        The index of the start of the steady window, or None if none found.
    """
    pop = np.asarray(pop)
    for i in range(window, len(pop)):
        tail = pop[i-window:i]
        if tail.max() - tail.min() > tol:
            continue
        if np.max(np.abs(np.diff(tail))) > tol:
            continue
        print('Steady state found at list index i = ', i - window)
        return i - window
    return None

def steady_state_correlations(bins:Bins, ops_same_time:list[np.ndarray], ops_two_time:list[np.ndarray], params:InputParams) -> list[np.ndarray]:
    """
    Efficient steady-state correlation calculation for continuous-wave pumping.
    This computes time differences starting from a convergence index (steady-state
    index). It returns a compact structure (SSCorrel or SSCorrel1Channel) 
    containing normalized g1 and g2 lists and raw correlation arrays.
    
    Parameters
    ----------
    bins : Bins
        Bins returned by time evolution functions
        cor_b field must contain the correlation tensors.
    
    params : InputParams
        Simulation parameters 
    
    pop : Pop1TLS
        Population dataclass used to detect steady-state   
        (it can be extended for other cases)
        
    Returns
    -------
    SSCorrel or SSCorrel1Channel
        Dataclass with precomputed time correlation lists for steady-state
        (for more info see the corresponding dataclasses in parameters.py)
    """
    
    pop=pop.pop
    cor_list=bins.correlation_bins
    delta_t, d_t_total,bond=params.delta_t,params.d_t_total,params.bond_max
    #First check convergence:
    conv_index =  steady_state_index(pop,10)  
    if conv_index is None:
        raise ValueError("tmax not long enough for steady state")
    # cor_list1=cor_list
    cor_list1=cor_list[conv_index:]
    t_cor=[0]
    i_1=cor_list1[-2]
    i_2=cor_list1[-1] #OC is in here
    p=0
    d_t=np.prod(d_t_total)
    swap_t_t=swap(d_t,d_t)
    
    if d_t==2:
        exp_0=expectation_1bin(i_2,delta_b_dag(delta_t, d_t_total))
        exp2_0=expectation_1bin(i_2, delta_b(delta_t, d_t_total))
        c1=[expectation_1bin(i_2, delta_b_dag(delta_t, d_t_total)@ delta_b(delta_t, d_t_total))]
        c2=[expectation_1bin(i_2, delta_b_dag(delta_t, d_t_total) @ delta_b_dag(delta_t, d_t_total) @ delta_b(delta_t, d_t_total) @delta_b(delta_t, d_t_total))]
        coher_list=[exp_0*exp2_0]
        denom=expectation_1bin(i_2,  delta_b_dag(delta_t, d_t_total)@ delta_b(delta_t, d_t_total))
        
        for i in range(len(cor_list1)-2,0,-1):
            state=ncon([i_1,i_2],[[-1,-2,1],[1,-3,-4]]) 
            # Separating between left and right spectra
            c1.append(expectation_2bins(state, g1(delta_t,d_t_total))) #for calculating the total spectra
            c2.append(expectation_2bins(state, g2(delta_t,d_t_total)))
            coher_list.append(exp_0*expectation_1bin(i_2, delta_b(delta_t, d_t_total))) #for calculating the coherent spectra           
            swaps=ncon([i_1,i_2,swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #swapping the feedback bin to the left so it is next to the next bin
            i_t1,stemp,i_t2=_svd_tensors(swaps,bond,d_t,d_t)
            i_2 = ncon([i_t1,np.diag(stemp)],[[-1,-2,1],[1,-3]]) #OC tau bin
            i_1=cor_list1[i-1] #next past bin for the next time step
            p+=1    
            t_cor.append(p*delta_t)       
        g1_list=c1/denom
        g2_list=c2/denom**2
        return SSCorrel1Channel(t_cor=t_cor,g1_list=g1_list,g2_list=g2_list,c1=c1,c2=c2)

    else:
        exp_0l=expectation_1bin(i_2,delta_b_dag_l(delta_t, d_t_total))
        exp2_0l=expectation_1bin(i_2, delta_b_l(delta_t, d_t_total))
        exp_0r=expectation_1bin(i_2, delta_b_dag_r(delta_t, d_t_total))
        exp2_0r=expectation_1bin(i_2, delta_b_r(delta_t, d_t_total))
        c1_l=[expectation_1bin(i_2, delta_b_dag_l(delta_t, d_t_total)@ delta_b_l(delta_t, d_t_total))]
        c1_r=[expectation_1bin(i_2, delta_b_dag_r(delta_t, d_t_total) @ delta_b_r(delta_t, d_t_total))]
        c2_l=[expectation_1bin(i_2, delta_b_dag_l(delta_t, d_t_total) @ delta_b_dag_l(delta_t, d_t_total) @ delta_b_l(delta_t, d_t_total) @delta_b_l(delta_t, d_t_total))]
        c2_r=[expectation_1bin(i_2,  delta_b_dag_r(delta_t, d_t_total) @ delta_b_dag_r(delta_t, d_t_total) @ delta_b_r(delta_t, d_t_total) @delta_b_r(delta_t, d_t_total))]
        coher_listl=[exp_0l*exp2_0l]
        coher_listr=[exp_0r*exp2_0r]    
        denoml=expectation_1bin(i_2,  delta_b_dag_l(delta_t, d_t_total)@ delta_b_l(delta_t, d_t_total))
        denomr = expectation_1bin(i_2,  delta_b_dag_r(delta_t, d_t_total)@ delta_b_r(delta_t, d_t_total))
        
        for i in range(len(cor_list1)-2,0,-1):
            state=ncon([i_1,i_2],[[-1,-2,1],[1,-3,-4]]) 
            # Separating between left and right spectra
            c1_l.append(expectation_2bins(state, g1_ll(delta_t,d_t_total))) #for calculating the total spectra
            c1_r.append(expectation_2bins(state, g1_rr(delta_t,d_t_total)))
            c2_l.append(expectation_2bins(state, g2_ll(delta_t,d_t_total)))
            c2_r.append(expectation_2bins(state, g2_rr(delta_t,d_t_total)))
            coher_listl.append(exp_0l*expectation_1bin(i_2, delta_b_l(delta_t, d_t_total))) #for calculating the coherent spectra
            coher_listr.append(exp_0r*expectation_1bin(i_2, delta_b_r(delta_t, d_t_total)))
            
            swaps=ncon([i_1,i_2,swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #swapping the feedback bin to the left so it is next to the next bin
            i_t1,stemp,i_t2=_svd_tensors(swaps,bond,d_t,d_t)
            i_2 = ncon([i_t1,np.diag(stemp)],[[-1,-2,1],[1,-3]]) #OC tau bin
            i_1=cor_list1[i-1] #next past bin for the next time step
            p+=1    
            t_cor.append(p*delta_t)
        
        g1_listl=c1_l/denoml
        g2_listl=c2_l/denoml**2
        g1_listr=c1_r/denomr
        g2_listr=c2_r/denomr**2
        
        return SSCorrel(t_cor=t_cor,g1_listl=g1_listl,g1_listr=g1_listr,g2_listl=g2_listl,g2_listr=g2_listr,c1_l=c1_l,c1_r=c1_r,c2_l=c2_l,c2_r=c2_r)
    
