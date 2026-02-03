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
from QwaveMPS import states as states
from collections.abc import Iterator
from QwaveMPS.parameters import InputParams, Bins
from typing import Callable, TypeAlias
from QwaveMPS.hamiltonians import Hamiltonian
from QwaveMPS.operators import *
from QwaveMPS.operators import u_evol, swap

__all__ = ['t_evol_mar', 't_evol_nmar', 't_evol_nmar_old', 't_evol_nmar_chiral']

# -----------------------------------
# Singular Value Decomposition helper
# -----------------------------------

def _svd_tensors(tensor:np.ndarray, bond_max:int, d_1:int, d_2:int) -> np.ndarray:
    """
    Perform a SVD, reshape the tensors and return left tensor, 
    normalized Schmidt vector, and right tensor.

    Parameters
    ----------
    tensor : ndarray
        tensor to decompose
    
    bond_max : int
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
    chi = min(bond_max, len(s))
    epsilon = 1e-12 #to avoid dividing by zero
    s_norm = s[:chi] / (norm(s[:chi])+ epsilon)
    u = u[:, :chi].reshape(tensor.shape[0],d_1,chi)
    vt = vt[:chi, :].reshape(chi,d_2,tensor.shape[-1])
    return u, s_norm, vt

# ------------------------------------------------------
# Time evolution: Markovian and non-Markovian evolutions
# ------------------------------------------------------

def t_evol_mar(ham:Hamiltonian, i_s0:np.ndarray, i_n0:np.ndarray, params:InputParams) -> Bins:
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
        
    params : InputParams
        Class containing the input parameters
        (contains delta_t, tmax, bond, d_t_total, d_sys_total).

    Returns
    -------
    results : Bins (from parameters.py) 
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

    # Prepare for results and store initial states
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

    # Time Evolution loop
    for k in range(n):   
        i_nk = next(input_field)   
        if callable(ham):
            evol=u_evol(ham(k),d_sys,d_t)

        # Swap system bin to right of time bin      
        phi2=ncon([i_s,i_nk,swap_sys_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #system bin, time bin + swap contraction
        i_n,stemp,i_st=_svd_tensors(phi2, bond,d_t,d_sys)
        i_n = i_n * stemp[None,None,:] # OC in input bin
        tbins_in.append(i_n)

        # Time evolution
        phi1=ncon([i_n,i_st,evol],[[-1,2,3],[3,4,-4],[-2,-3,2,4]]) #system bin, time bin + u operator contraction  
        i_n,stemp,i_s=_svd_tensors(phi1, bond,d_t,d_sys)
        i_s = stemp[:,None,None] * i_s #OC system bin

        sbins.append(i_s)
        tbins.append(i_n * stemp[None,None,:])
        
        schmidt.append(stemp)
        cor_list.append(i_n)
        t_k += delta_t

    # Overwrite last entry with the OC
    cor_list[-1] = i_n * stemp[None,None,:]

    return Bins(system_states=sbins,output_field_states=tbins, input_field_states=tbins_in,
                correlation_bins=cor_list,schmidt=schmidt)

def t_evol_nmar_old(ham:Hamiltonian, i_s0:np.ndarray, i_n0:np.ndarray,params:InputParams) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """ 
    Time evolution of the system with finite delays/feedback (non-Markovian regime).
    Requires tau to be at least delta_t.
    
    Parameters
    ----------
    ham : ndarray/callable
        Either a fixed evolution operator/tensor or a callable returning the
        evolution operator for time-step k: ham(k).
        
     i_s0 : ndarray
         Initial system bin (tensor).
         
     i_n0: ndarray 
         Initial field bin.
         Seed for the input time-bin generator.

     params : InputParams
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
    
    # Lists for storing results
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
    
    # Fill the feedback loop with vacuum bins
    for i in range(l):
        nbins.append(states.wg_ground(d_t))
        t_0+=delta_t
    
    i_stemp=i_s0      
    
    # Simulation loop for time evolution
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
        i_n,stemp,i_stemp=_svd_tensors(phi2, bond,d_t,d_sys)   
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
        
        nbins[k+1] = stemp[:,None,None]*i_n2 #new tau bin for the next time step       
        cor_list.append(i_t)    

    # Rewrite the last result time bin with the OC in it
    cor_list[-1] = i_t*stemp[None,None,:]

    return Bins(system_states=sbins,loop_field_states=tbins,output_field_states=taubins,
                input_field_states=tbins_in,correlation_bins=cor_list,
                schmidt=schmidt,schmidt_tau=schmidt_tau)


def _feedback_swapping(bin_list:list[np.ndarray], l_list:list[int], d_t:int, schmidt_list:list[np.ndarray], bond:int):
    """
    Private helper procedure used in the non-markovian simulation for swapping the various tau bins to the appropriate locations in the bin list after time evolution

    Parameters
    ----------
    bin_list : list[ndarray]
        List of bins to be reorganized after time evolution, with the OC located in the final time bin in the list.
    
    l_list : list[int]
        Ordered list of the feedback lengths in terms of bin numbers.
    
    d_t : int
        Total scaler dimension of the time bins
    
    bond : int
        The maximum bond dimension
    """
    
    n = len(l_list)
    cummulative_l = np.cumsum(np.array(l_list))
    time_bins_swap = swap(d_t, d_t)
    
    # Repeat process for each feedback bin to get in right places
    for j in range(n-1, -1 ,-1):
        
        # Move OC N bins to the left
        for k in range(n):   
            right_oc_bin = bin_list[-1-k]
            left_bin = bin_list[-2-k]
            contraction=ncon([left_bin,right_oc_bin],[[-1,-2,2],[2,-3,-4]])
            left_bin, stemp, right_bin = _svd_tensors(contraction, bond, d_t, d_t)
            left_oc_bin = left_bin * stemp[None,None,:]
            
            bin_list[-1-k] = right_bin
            bin_list[-2-k] = left_oc_bin
            #print('Moved OC from bin', -1-k, 'to', -2-k)
    
        # Note at this point have OC in binList[-n-1], the jth last temporal feedback bin that was most recently interacted with
        # In the case that j=n-1, this is the output field time bin
        
        # Swap current bin cummulativeL[j] bins to the left (leave OC bin in right bin at end)
        ocBin = bin_list[-1]
        start_index = -1-n
        for k in range(cummulative_l[j]-j-1): #make cleaner #for k in range(cummulativeL[j]-n):
            right_oc_bin = bin_list[start_index-k]
            left_bin = bin_list[start_index-k-1]
            
            swaps=ncon([left_bin,right_oc_bin,time_bins_swap],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #feedback bin, time bin + swap contraction
            left_bin, stemp, right_bin = _svd_tensors(swaps, bond, d_t, d_t)

            if k < cummulative_l[j] -n-1:
                left_oc_bin = left_bin * stemp[None,None,:] #OC tau bin
                bin_list[start_index-k] = right_bin
                bin_list[start_index-k-1] = left_oc_bin
                #print('if', k)
            else:
                right_oc_bin = stemp[:,None,None] * right_bin #OC tau bin            
                bin_list[start_index-k] = right_oc_bin
                bin_list[start_index-k-1] = left_bin
                #print('else', k)
            
            #print('Swap bin at', startIndex-k, 'with', startIndex-k-1)

        # Save the schmidts!
        schmidt_list[j].append(stemp)
        
        # Swap the new OC bin cummulativeL[j]+n-1 bins to right (Leave OC in right bin at end of swapping)
        
        start_index = -cummulative_l[j] - (n-j-1) # make cleaner
        for k in range(cummulative_l[j] + (n-j-2)): #-1
            left_oc_bin = bin_list[start_index + k]
            rightBin = bin_list[start_index + k + 1]
            
            swaps=ncon([left_oc_bin,rightBin,time_bins_swap],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #feedback bin, time bin + swap contraction

            left_bin, stemp, right_bin = _svd_tensors(swaps, bond, d_t, d_t)        
            right_oc_bin = stemp[:,None,None] * right_bin #OC tau bin
            
            bin_list[start_index+k] = left_bin
            bin_list[start_index+k+1] = right_oc_bin
            
            #print('Swap bins', startIndex+k, 'and', startIndex+k+1)
    #return bin_list

def _initialize_feedback_loop(bin_list:list[np.ndarray], sys_bin:np.ndarray, l_list:list[int], d_t:int, d_sys:int, bond:int, input_field_generator:Iterator=None, bond0:int=1) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Private helper function used in the non-markovian simulation to initially fill the feedback channel(s) with time bins.

    Parameters
    ----------
    bin_list : list[ndarray]
        List of bins to be reorganized after time evolution.
    
    sys_bin : ndarray
        Initial system bin at the start of the simulation.
        
    l_list : list[int]
        Ordered list of the feedback lengths in terms of bin numbers.
    
    d_t : int
        Total scaler dimension of the time bins

    d_sys : int
        Total scalar dimension of the system

    input_field : Iterator, default: None
        Generator of the time bins present in the feedback spaces in the waveguide prior to the start of the simulation. If None then the space is filled with vacuum bins. 
    
    bond : int
        The maximum bond dimension

    bond0 : int, default: 1
        Initial bond dimension size.

    Returns
    -------
    bin_list : list[ndarray]
        A reference to the inplace, modified list of time bins.

    sys_bin : ndarray
        Updated system bin with the OC after adding it to the matrix product states of the waveguide.
    """
    n = len(l_list) # number of feedback loops
    cummulative_l = np.cumsum(np.array(l_list))
    time_bin_swap = swap(d_t, d_t)
    sys_time_swap = swap(d_sys, d_t)
    vacuum_flag = False
    
    if input_field_generator is None:
        vacuum_flag = True
        input_field_generator = states.input_state_generator(d_t)
    
    # Add in the bins from the right using the generator, and swap to left side of the sys bin
    for i in range(cummulative_l[-1]):
        new_time_bin = next(input_field_generator)

        contraction = ncon([sys_bin, new_time_bin, sys_time_swap], [[-1,5,1],[1,6,-4], [-2,-3,5,6]])
        time_bin, stemp, sys_bin = _svd_tensors(contraction, bond, d_t, d_sys)
        
        sys_bin = stemp[:,None,None]*sys_bin
        bin_list.append(time_bin)

    # Reorder the bins correctly according to feedback (don't have to do this with vacuum bins)
    # Yet to verify this works correctly!!
    if not vacuum_flag:
        # Reposition the bins. Iterate over feedback loops
        for i in range(n-1, -1, -1):
            
            # Move OC from binList[-1] to cummulativeL[i]+n-1 bins to the left (no swapping)
            for j in range(cummulative_l[i] + (i-n)):
                right_time_bin = bin_list[-1-j]
                left_time_bin = bin_list[-2-j]
                contraction = ncon([left_time_bin, right_time_bin], [[-1,-2,1],[1,-3,-4]])
                left_time_bin, stemp, right_time_bin = _svd_tensors(contraction, bond, d_t, d_t)
                
                left_time_bin = left_time_bin * stemp[None,None,:]
                bin_list[-1-j] = right_time_bin
                bin_list[-2-j] = left_time_bin

        
            # DSwap the bin all the way to just right of the system bin
            for j in range(cummulative_l[i]+ (i-n), 1, -1):
                left_time_bin = bin_list[-j]
                right_time_bin = bin_list[-j+1]

                
                swapsContraction = ncon([left_time_bin, right_time_bin, time_bin_swap], [[-1,5,1],[1,6,-4], [-2,-3,5,6]])
                left_time_bin, stemp, right_time_bin = _svd_tensors(swapsContraction, bond, d_t, d_t)
                
                right_time_bin = stemp[:,None,None]*right_time_bin
                bin_list[-j] = left_time_bin
                bin_list[-j+1] = right_time_bin

        
        
        # Make sure OC is in sys bin at end and not time bin to right of sys bin
        time_bin = bin_list[-1]
        contraction = ncon([time_bin, sys_bin], [[-1,-2,1],[1,-3,-4]])
        time_bin, stemp, sys_bin = _svd_tensors(contraction, bond, d_t, d_sys)
        
        sys_bin = stemp[:,None,None]*sys_bin
        bin_list[-1] = time_bin
    
    return bin_list, sys_bin

def _ncon_contraction_index_list(bin_num:int, operator_flag:bool=True)->list[list[np.ndarray]]:
    """
    Private helper function used in the non-markovian simulation to create the index list for the contraction with the time evolution operator.
    Can be used more generally for contraction of many bins with a single operator (or just together).

    Parameters
    ----------
    bin_num : int
        The total number of bins being contracted together. Must be greater than 1.
    
    operator_flag : bool, default: True
        Boolean that dictates whether the list of bins will be contracted with a single operator tensor at the end.
        
    Returns
    -------
    indices : list[list[int]]
        The list of index lists describing the contractions of the various adjacent bins, possibly with their physical index with an operator.
    """
    if operator_flag:
        return [[-1,bin_num,1]] + [[i+1,bin_num+1+i, i+2] for i in range(bin_num-2)] + [[bin_num-1,2*bin_num-1, -(bin_num + 2)]] \
        + [[-2-i for i in range(bin_num)] + [bin_num+i for i in range(bin_num)]]
    else:
        return [[-1,-2,1]] + [[i+1, -(i+3), i+2] for i in range(bin_num-2)] + [[bin_num-1, -(bin_num+1), -(bin_num+2)]]
    
def _separate_bins(bins:list[np.ndarray], phi:np.ndarray, feedback_loop_num:int, d_t:int, d_sys:int, oc_normed_bins_lists:list[list[np.ndarray]], bond:int)->tuple[list[np.ndarray], np.ndarray]:
    """
    Private helper function used in the non-markovian simulation to separate the various bins immediately after the time evolution operator has been used.

    Parameters
    ----------
    bins : list[ndarray]
        List of bins to have the end rewritten inplace with the separated out tau bins.
    
    phi : ndarray
        Tensor produced after the time evolution contraction, containing physical indices of all the interacting time bins and the system bin.
        
    feedback_loop_num : int
        Number of feedback channels/extra time bins interacting with the system bin.
    
    d_t : int
        Total scaler dimension of the time bins

    d_sys : int
        Total scalar dimension of the system

    input_field : Iterator, default: None
        Generator of the time bins present in the feedback spaces in the waveguide prior to the start of the simulation. If None then the space is filled with vacuum bins. 
    
    oc_normed_bins_lists : list[list[np.ndarray]]
        List of lists containing the OC normalized time bins that are updated in this step and can be used for single time observables.
    
    bond : int
        The maximum bond dimension

    Returns
    -------
    bin_list : list[ndarray]
        A reference to the inplace, modified list of time bins.

    sys_bin : ndarray
        Updated system bin with the OC.
    """
    for i in range(feedback_loop_num+1):
        dim2 = d_sys * d_t**(feedback_loop_num - i)
        i_t, stemp, i_2 = _svd_tensors(phi, bond, d_t, dim2)

        bins[-feedback_loop_num-1+i] = i_t 
        phi = stemp[:,None,None]*i_2 #ncon([np.diag(stemp),i_2],[[-1,1],[1,-2,-3]])
        oc_normed_bins_lists[i].append(i_t*stemp[None,None,:])

    i_s = phi    
    return bins, i_s

def t_evol_nmar(ham:Hamiltonian, i_s0:np.ndarray, i_n0:np.ndarray, taus:list[float], params:InputParams) -> Bins:
    """ 
    Time evolution of the system with delay times
    
    Parameters
    ----------
    i_s0 : ndarray
        Initial system bin
    
    input_field : Iterator
        Generator of time bins incident the system.

    taus : list[float]
        List of feedback times, the order of the times coincide with the order of the feedback tensor spaces from left to right.
    
    delta_t : float
        time step

    tmax : float
        max time

    bond : int
        max bond dimension
    
    d_sys_total : ndarray
        List of sizes of system Hilbert spaces.

    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

    taus : int, default: 1
        List of delay times for nonmarkovian channels. Ordering coincides to order of feedback tensor spaces in the Hamiltonian from right to left.

    Returns
    -------
    sbins : list[ndarray]
        A list with the system bins.
    
    oc_normed_bins_lists : list[list[ndarray]]
        A list of time dependent lists of OC normalized time bins, with the first index coinciding to the bin leaving the waveguide, and higher indices being in deeper feedback loops.
    
    tbins_in : list[ndarray]
        A list of the input time bins (with OC).

    nbins : list[ndarray]
        A list of the numpy array time bins representing the entire Matrix Product State with the OC in the final bin.

    schmidt : [ndarray]
        A list of the Schmidt coefficients
        return ,oc_normed_bins_lists
    """
    delta_t = params.delta_t
    tmax=params.tmax
    bond=params.bond_max
    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total

    d_t=np.prod(d_t_total)
    d_sys=np.prod(d_sys_total)
    feedback_bin_num = len(taus)
    interacting_time_bin_num = len(taus) + 1

    sbins=[] 
    tbins_in = []
    nbins=[]
    sbins.append(i_s0)   
    tbins_in.append(states.i_ng(d_t))
    # indexed with 0 being out of the WG, greater index is greater depth
    oc_normed_bins_lists = [[states.i_ng(d_t)] for x in range(interacting_time_bin_num)]
    schmidts = [np.zeros(1) for x in range(interacting_time_bin_num)]

    input_field=states.input_state_generator(d_t_total, i_n0)
    n=int(round(tmax/delta_t,0))
    t_k=0
    t_0=0
    if not callable(ham):
        evol=u_evol(ham,d_sys,d_t,interacting_time_bin_num) #Feedback loop means time evolution involves an input and a feedback time bin. Can generalize this later, leaving 2 for now so it runs.
    swap_t_t=swap(d_t,d_t)
    swap_sys_t=swap(d_sys,d_t)
    taus = np.array(taus)
    l_list=np.array(np.round(taus/delta_t, 0).astype(int)) #time steps between system and feedback
        
    i_s=i_s0      
    
    #first l time bins in vacuum (for no feedback part)    
    
    # Create the time contraction indices and initialize the feedback loop with bins
    time_evolution_ncon_indices = _ncon_contraction_index_list(interacting_time_bin_num+1, True) # Plus 1 for the system bin!
    nbins, i_s = _initialize_feedback_loop(nbins, i_s, l_list, d_t, d_sys, bond, input_field_generator=None) # Modifies nbins in place, need to return i_s as it is reassigned in function
    for k in range(n):
        if callable(ham):
            evol=u_evol(ham(k),d_sys,d_t)

        in_state = next(input_field)
        
        # Swap with inState and put OC in inState to save it
        phi1 = ncon([i_s, in_state, swap_sys_t], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
        in_state, stemp, i_s = _svd_tensors(phi1, bond, d_t, d_sys)
        in_state = in_state * stemp[None,None,:] #OC time bin

        tbins_in.append(in_state)
        nbins.append(in_state)
        
        # Time Evolution
        phi1=ncon(nbins[-feedback_bin_num-1:-1] + [in_state,i_s,evol],time_evolution_ncon_indices) #tau bin, system bin, future time bin + U operator contraction
        
        # Separate the bins into individual components, end with OC in sys bin
        nbins, i_s = _separate_bins(nbins, phi1, feedback_bin_num, d_t, d_sys, oc_normed_bins_lists, bond=bond)

        #swap OC into i_n
        i_n = nbins[-1]
        phi2=ncon([i_n,i_s],[[-1,-2,1],[1,-3,-4]]) #system bin, time bin + swap contraction
        i_n, stemp, i_s = _svd_tensors(phi2, bond, d_t, d_sys)

        i_n = i_n * stemp[None,None,:] #the OC in time bin     
        nbins[-1] = i_n                  
        #nbins = _feedback_swapping(nbins, l_list, d_t, bond=bond)
        _feedback_swapping(nbins, l_list, d_t, schmidts, bond=bond) # Mutates nbins inplace

                
        #Put OC in system bin
        phi1 = ncon([nbins[-1], i_s], [[-1,-2,1],[1,-3,-4]])
        i_t, stemp, i_s = _svd_tensors(phi1, bond, d_t, d_sys)
        i_s = stemp[:,None,None] * i_s #OC sys bin
        nbins[-1] = i_t
        # Save the last case of the schmidts, system bin with everything else
        schmidts[-1].append(stemp)
        sbins.append(i_s) #the system bin is computed here as it is the moment it is the OC

        t_k += delta_t
        #tlist.append(currT)

    # Put OC in end of nbins chain at end 
    phi1 = ncon([nbins[-1], i_s], [[-1,-2,1],[1,-3,-4]])
    i_t, stemp, i_s = _svd_tensors(phi1, bond, d_t, d_sys)
    i_t = i_t * stemp[None,None,:] #OC in time bin
    nbins[-1] = i_t

    # Reverse the normbed bins lists so that start is at 0 and output is -1
    oc_normed_bins_lists.reverse()

    return Bins(system_states=sbins,output_field_states=oc_normed_bins_lists, input_field_states=tbins_in,
            correlation_bins=nbins,schmidt=schmidts)


def _initialize_feedback_loop_chiral(nbins, l_list, d_t, d_sys_total, bond, input_field_generator=None):
    """
    Private helper function used in the non-markovian simulation to initially fill the feedback channel(s) with time bins.

    Parameters
    ----------
    bin_list : list[ndarray]
        List of bins to be reorganized after time evolution.
    
    sys_bin : ndarray
        Initial system bin at the start of the simulation.
        
    l_list : list[int]
        Ordered list of the feedback lengths in terms of bin numbers.
    
    d_t : int
        Total scaler dimension of the time bins

    d_sys : int
        Total scalar dimension of the system

    input_field : Iterator, default: None
        Generator of the time bins present in the feedback spaces in the waveguide prior to the start of the simulation. If None then the space is filled with vacuum bins. 
    
    bond : int
        The maximum bond dimension

    bond0 : int, default: 1
        Initial bond dimension size.

    Returns
    -------
    bin_list : list[ndarray]
        A reference to the inplace, modified list of time bins.

    sys_bin : ndarray
        Updated system bin with the OC after adding it to the matrix product states of the waveguide.
    """
    sys_num = len(d_sys_total) # number of feedback loops
    swap_t_t=swap(d_t,d_t)
    swap_sys_t= []
    for i in range(sys_num):
        swap_sys_t.append(swap(d_sys_total[i],d_t))

    if input_field_generator is None:
        input_field_generator = states.input_state_generator(d_t)
    
    # Add in the bins from the right using the generator, and swap to left side of the sys bins appropriately
    for i in range(sys_num-1):
        for j in range(l_list[-1-i]):
            new_time_bin = next(input_field_generator)
            nbins.append(new_time_bin)
            oc_ind = -1

            # Swap input_bin sys_num-i bins to the left
            for k in range(sys_num-i):
                phi1 = ncon([nbins[oc_ind-1], nbins[oc_ind], swap_sys_t[k]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
                oc_ind = oc_ind-1
                nbins[oc_ind], stemp, nbins[oc_ind+1] = _svd_tensors(phi1, bond, d_t, d_sys_total[k])
                nbins[oc_ind] = nbins[oc_ind] * stemp[None,None,:] #OC time bin

            # Move OC sys_num-i bins to the right, back to nbins[-1]
            phi1 = ncon([nbins[oc_ind], nbins[oc_ind+1]], [[-1,-2,1],[1,-3,-4]])
            oc_ind = oc_ind+1
            nbins[oc_ind-1], stemp, nbins[oc_ind] = _svd_tensors(phi1, bond, d_t, d_sys_total[-1-i])
            nbins[oc_ind] = stemp[:,None,None] * nbins[oc_ind] #OC time bin

            for k in range(sys_num-i-1):
                phi1 = ncon([nbins[oc_ind], nbins[oc_ind+1]], [[-1,-2,1],[1,-3,-4]])
                oc_ind = oc_ind + 1
                nbins[oc_ind-1], stemp, nbins[oc_ind] = _svd_tensors(phi1, bond, d_sys_total[-1-i-k], d_sys_total[-2-i-k])
                nbins[oc_ind] = stemp[:,None,None] * nbins[oc_ind] #OC time bin

    return nbins

def _separate_sys_bins_chiral(i_s, d_sys_total, sbins, bond):
    nbins = []
    sys_num = len(d_sys_total)

    for i in range(-1, -sys_num, -1):
        sys_i, stemp, i_s = _svd_tensors(i_s, bond, d_sys_total[i], np.prod(d_sys_total[:i]))
        nbins.append(sys_i)
        i_s = stemp[:,None,None] * i_s #OC time bin
        sbins[i].append(sys_i * stemp[None,None,:])

    nbins.append(i_s)
    sbins[0].append(i_s)
    return nbins

def t_evol_nmar_chiral(hams:list[np.ndarray], i_s0:np.ndarray, i_n0:np.ndarray, taus:list[float], params:InputParams) -> Bins:
    """ 
    Time evolution of a chiral waveguide system with delay times (no feedbacks)
    Structured memory-efficiently with separated system bins in the MPS chain
    
    Parameters
    ----------
    i_s0 : ndarray
        Initial system bin
    
    input_field : Iterator
        Generator of time bins incident the system.

    taus : list[float]
        List of feedback times, the order of the times coincide with the order of the feedback tensor spaces from left to right.
    
    delta_t : float
        time step

    tmax : float
        max time

    bond : int
        max bond dimension
    
    d_sys_total : ndarray
        List of sizes of system Hilbert spaces.

    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

    taus : int, default: 1
        List of delay times for nonmarkovian channels. Ordering coincides to order of feedback tensor spaces in the Hamiltonian from right to left.

    Returns
    -------
    sbins : list[ndarray]
        A list with the system bins.
    
    oc_normed_bins_lists : list[list[ndarray]]
        A list of time dependent lists of OC normalized time bins, with the first index coinciding to the bin leaving the waveguide, and higher indices being in deeper feedback loops.
    
    tbins_in : list[ndarray]
        A list of the input time bins (with OC).

    nbins : list[ndarray]
        A list of the numpy array time bins representing the entire Matrix Product State with the OC in the final bin.

    schmidt : [ndarray]
        A list of the Schmidt coefficients
        return ,oc_normed_bins_lists
    """
    delta_t = params.delta_t
    tmax=params.tmax
    bond=params.bond_max
    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total


    d_t=np.prod(d_t_total)
    d_sys=np.prod(d_sys_total)
    feedback_bin_num = len(taus)
    sys_num = len(d_sys_total)

    tbins_in = []
    schmidt=[]

    
    input_field=states.input_state_generator(d_t_total, i_n0)
    n=int(round(tmax/delta_t,0))
    t_k=0
    t_0=0
    evols = [] * sys_num
    callable_ham_indices = []
    for i in range(len(hams)):
        if not callable(hams[i]):
            evols[i] = u_evol(hams[i],d_sys_total[i],d_t) #Feedback loop means time evolution involves an input and a feedback time bin. Can generalize this later, leaving 2 for now so it runs.
        else:
            callable_ham_indices.append(i)
    swap_t_t=swap(d_t,d_t)
    swap_sys_t= []
    for i in range(sys_num):
        swap_sys_t.append(swap(d_sys_total[i],d_t))
    taus = np.array(taus)
    l_list=np.array(np.round(taus/delta_t, 0).astype(int)) #time steps between system and feedback
        
    # indexed with 0 being out of the WG, greater index is greater depth
    oc_normed_bins_lists = [[states.i_ng(d_t)] for x in range(sys_num)]
    schmidts = [np.zeros(1) for x in range(sys_num)]
    sbins=[[] for x in range(sys_num)] 
    
    i_s=i_s0      
    
    #first l time bins in vacuum (for no feedback part)    
    
    # Separate the system bins and initialize the feedback loop with bins
    nbins = _separate_sys_bins_chiral(i_s, d_sys_total, sbins, bond)
    nbins = _initialize_feedback_loop_chiral(nbins, l_list, d_t, d_sys_total, bond, input_field_generator=None) # Modifies nbins in place
    
    for k in range(n):
        print(str(round(k/n*100,2)) + '%')
        for i in callable_ham_indices:
            evols[i] = u_evol(hams[i](k), d_sys_total[i], d_t)

        sbin_0 = nbins[-1]
        in_state = next(input_field)
        nbins.append(in_state)
        
        # Swap with inState and put OC in inState to save it
        phi1 = ncon([sbin_0, in_state, swap_sys_t[0]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
        nbins[-2], stemp, nbins[-1] = _svd_tensors(phi1, bond, d_t, d_sys_total[0])
        nbins[-2] = nbins[-2] * stemp[None,None,:] #OC time bin

        tbins_in.append(nbins[-2])
        
        # Time Evolution of first sys bin
        phi1=ncon(nbins[-2:] + [evols[0]],[[-1,2,1],[1,3,-4],[-2,-3,2,3]]) #t_in bin, system bin, + U operator contraction

        # Separate the bins with OC in time bin to left, also saving OC normalized sys[0] and flux out
        nbins[-2], stemp, nbins[-1] = _svd_tensors(phi1, bond, d_t, d_sys_total[0])
        nbins[-2] = nbins[-2] * stemp[None,None,:] # OC time bin

        oc_normed_bins_lists[0].append(nbins[-2])
        sbins[0].append(stemp[:,None,None] * nbins[-1])

        # Loop over time evolution process for all interacting sysbins in the chain
        for j in range(feedback_bin_num):
            # Determine the relative -1 index after tbin is swapped into this section of chain
            sys_ind = -(2 + np.sum(l_list[:j]+1))
            oc_ind = sys_ind - 1
            
            # Swap OC time bin on R of sysbin with current sysbin
            phi1 = ncon([nbins[sys_ind-1], nbins[sys_ind], swap_sys_t[j+1]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
            nbins[oc_ind], stemp, nbins[sys_ind] = _svd_tensors(phi1, bond, d_t, d_sys_total[j+1])
            nbins[oc_ind] = nbins[oc_ind] * stemp[None,None,:] #OC time bin


            # Move OC tau_j bins left
            for i in range(l_list[j]):
                phi1 = ncon([nbins[oc_ind-1], nbins[oc_ind]], [[-1,-2,1],[1,-3,-4]])
                oc_ind = oc_ind - 1
                nbins[oc_ind], stemp, nbins[oc_ind+1] = _svd_tensors(phi1, bond, d_t, d_t)
                nbins[oc_ind] = nbins[oc_ind] * stemp[None,None,:] #OC time bin

            # swap new OC bin tau_j bins right
            for i in range(l_list[j]):
                phi1 = ncon([nbins[oc_ind], nbins[oc_ind+1], swap_t_t], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
                oc_ind = oc_ind+1
                nbins[oc_ind-1], stemp, nbins[oc_ind] = _svd_tensors(phi1, bond, d_t, d_t)
                nbins[oc_ind] = stemp[:,None,None] * nbins[oc_ind] #OC time bin

            # time evolve OC bin with bin on right (sys bin) with evol[j]
            phi1=ncon([nbins[oc_ind], nbins[oc_ind+1] ,evols[j+1]],[[-1,2,1],[1,3,-4],[-2,-3,2,3]]) #t_in bin, system bin, + U operator contraction

            # Separate bins and save
            nbins[oc_ind], stemp, nbins[oc_ind+1] = _svd_tensors(phi1, bond, d_t, d_sys_total[j+1])
            nbins[oc_ind] = nbins[oc_ind] * stemp[None,None,:] # OC time bin

            oc_normed_bins_lists[j+1].append(nbins[oc_ind])
            sbins[j+1].append(stemp[:,None,None] * nbins[oc_ind+1])

            # Swap OC bin tau_j bins left
            for i in range(l_list[j]):
                phi1 = ncon([nbins[oc_ind-1], nbins[oc_ind], swap_t_t], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
                oc_ind = oc_ind - 1
                nbins[oc_ind], stemp, nbins[oc_ind+1] = _svd_tensors(phi1, bond, d_t, d_t)
                nbins[oc_ind] = nbins[oc_ind] * stemp[None,None,:] #OC time bin


        # Move OC to the right to bring it back into the first sys bin in the chain
        # Start by moving it one to the right (requires tau of at least 1 bin)
        # Save schmidts during this part/return trip
        phi1 = ncon([nbins[oc_ind], nbins[oc_ind+1]], [[-1,-2,1],[1,-3,-4]])
        oc_ind = oc_ind + 1
        nbins[oc_ind-1], stemp, nbins[oc_ind] = _svd_tensors(phi1, bond, d_t, d_t)
        nbins[oc_ind] = stemp[:,None,None] * nbins[oc_ind]  #OC time bin
        schmidts[0].append(stemp)
        
        for j in range(feedback_bin_num):
            # Move OC adjacent to next sys bin
            for k in range(l_list[-j]-1):
                phi1 = ncon([nbins[oc_ind], nbins[oc_ind+1]], [[-1,-2,1],[1,-3,-4]])
                oc_ind = oc_ind + 1
                nbins[oc_ind-1], stemp, nbins[oc_ind] = _svd_tensors(phi1, bond, d_t, d_t)
                nbins[oc_ind] = stemp[:,None,None] * nbins[oc_ind]  #OC time bin
            # Move it past the sys bin
            phi1 = ncon([nbins[oc_ind], nbins[oc_ind+1]], [[-1,-2,1],[1,-3,-4]])
            oc_ind = oc_ind + 1
            nbins[oc_ind-1], stemp, nbins[oc_ind] = _svd_tensors(phi1, bond, d_t, d_sys_total[-1-j])
            nbins[oc_ind] = stemp[:,None,None] * nbins[oc_ind]  #OC time bin


            # Move to next time bin chain
            if j == feedback_bin_num-1:
                phi1 = ncon([nbins[oc_ind], nbins[oc_ind+1]], [[-1,-2,1],[1,-3,-4]])
                oc_ind = oc_ind + 1
                nbins[oc_ind-1], stemp, nbins[oc_ind] = _svd_tensors(phi1, bond, d_sys_total[-1-j], d_t)
                nbins[oc_ind] = stemp[:,None,None] * nbins[oc_ind]  #OC time bin
             
                # Save next schmidt, containing all but previous systems and their loops
                schmidts[j+1].append(stemp)
            
            # Move to final sys bin
            else:
                phi1 = ncon([nbins[oc_ind], nbins[oc_ind+1]], [[-1,-2,1],[1,-3,-4]])
                oc_ind = oc_ind + 1
                nbins[oc_ind-1], stemp, nbins[oc_ind] = _svd_tensors(phi1, bond, d_sys_total[1], d_sys_total[0])
                nbins[oc_ind] = stemp[:,None,None] * nbins[oc_ind]  #OC time bin
                # Save next schmidt, containing all but previous systems and their loops
                schmidts[j+1].append(stemp)


        t_k += delta_t
        #tlist.append(currT)

    # Put OC in end of nbins chain at end of output (past all sys bins/feedback loops)
    # Have to be careful about dimensionality with the system bins
    # Swap OC from sysbin_0 to sysbin_1 to be next to first delay loop
    phi1 = ncon([nbins[-2], nbins[-1]], [[-1,-2,1],[1,-3,-4]])
    nbins[-2], stemp, nbins[-1] = _svd_tensors(phi1, bond, d_sys_total[-2], d_sys_total[-1])
    nbins[-2] = nbins[oc_ind] * stemp[None,None,:]  #OC left bin
    oc_ind = -2

    for j in range(feedback_bin_num):
        # Move OC into next chain of feedback time bins, past the system bin
        phi1 = ncon([nbins[oc_ind-1], nbins[oc_ind]], [[-1,-2,1],[1,-3,-4]])
        nbins[oc_ind-1], stemp, nbins[oc_ind] = _svd_tensors(phi1, bond, d_t, d_sys_total[-2-j])
        oc_ind = oc_ind - 1
        nbins[oc_ind] = nbins[oc_ind] * stemp[None,None,:]  #OC left bin

        # Loop to move OC adjacent to the next sys bin
        for k in range(l_list[j]-1):
            phi1 = ncon([nbins[oc_ind-1], nbins[oc_ind]], [[-1,-2,1],[1,-3,-4]])
            nbins[oc_ind-1], stemp, nbins[oc_ind] = _svd_tensors(phi1, bond, d_t, d_t)
            oc_ind = oc_ind - 1
            nbins[oc_ind] = nbins[oc_ind] * stemp[None,None,:]  #OC left bin

        # Move OC into the system bin (if there is one, if not into the time bin at very end)

        if j < feedback_bin_num-1:
            phi1 = ncon([nbins[oc_ind-1], nbins[oc_ind]], [[-1,-2,1],[1,-3,-4]])
            nbins[oc_ind-1], stemp, nbins[oc_ind] = _svd_tensors(phi1, bond, d_sys_total[-3-j], d_t)
            oc_ind = oc_ind - 1
            nbins[oc_ind] = stemp[:,None,None] * nbins[oc_ind]  #OC time bin 
        
        # Move to end time bin
        else:
            phi1 = ncon([nbins[oc_ind-1], nbins[oc_ind]], [[-1,-2,1],[1,-3,-4]])
            nbins[oc_ind-1], stemp, nbins[oc_ind] = _svd_tensors(phi1, bond, d_t, d_t)
            oc_ind = oc_ind - 1
            nbins[oc_ind] = nbins[oc_ind] * stemp[None,None,:]  #OC left bin
    
    truncation_number = np.sum(l_list) + sys_num
    nbins = nbins[:-truncation_number]

    return Bins(system_states=sbins,output_field_states=oc_normed_bins_lists, input_field_states=tbins_in,
        correlation_bins=nbins,schmidt=schmidts)
