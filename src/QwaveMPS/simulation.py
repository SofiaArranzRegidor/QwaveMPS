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
from QwaveMPS.operators import u_evol, swap, u_evol_multi_emitter_bins
from tqdm import tqdm

__all__ = ['t_evol_mar', 't_evol_nmar', 't_evol_nmar_chiral', 't_evol_nmar_sym']

# -----------------------------------
# Singular Value Decomposition helper
# -----------------------------------

#TODO Make the epsilon used by _svd_tensors a global parameter configurable by user, like rcparams. Can have issues sometimes
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
   
    # Block for bond size check
    # TODO: Should benchmark speed impact, this function is called a lot!
    total_weight = np.sum(s**2)
    discarded_weight = np.sum(s[chi:]**2)
    if total_weight > 0:
        disc_weight = discarded_weight / total_weight
    else:
        disc_weight = 0.0
    if disc_weight > 1e-6:
        print(f"Warning: SVD discarded weight = {disc_weight:.3e}; max bond may be too small")
    # End of block

    epsilon = 1e-10 #to avoid dividing by zero
    s_norm = s[:chi] / (norm(s[:chi])+ epsilon)
    u = u[:, :chi].reshape(tensor.shape[0],d_1,chi)
    vt = vt[:chi, :].reshape(chi,d_2,tensor.shape[-1])
    return u, s_norm, vt

# ------------------------------------------------------
# Time evolution: Markovian and non-Markovian evolutions
# ------------------------------------------------------

def t_evol_mar(ham:Hamiltonian, i_s0:np.ndarray, i_n0:np.ndarray, params:InputParams, show_progress:bool = True) -> Bins:
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
    
    show_progress : bool, default: True
        Flag to show progress bar of the simulation.

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
    for k in tqdm(range(n), disable = not show_progress):   
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

#-------------------------
# non-Markovian Evolution: All emitters in 1 system bin
#-------------------------

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

    show_progress : bool, default: True
        Flag to show progress bar of the simulation.

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

def t_evol_nmar(ham:Hamiltonian, i_s0:np.ndarray, i_n0:np.ndarray, taus:list[float], params:InputParams, show_progress:bool=True) -> Bins:
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

    show_progress : bool, default: True
        Display the progress bar for the completion of the simulation

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
    tbins_in.append(states.wg_ground(d_t))
    # indexed with 0 being out of the WG, greater index is greater depth
    oc_normed_bins_lists = [[states.wg_ground(d_t)] for x in range(interacting_time_bin_num)]
    schmidts = [[np.zeros(1)] for x in range(interacting_time_bin_num)]
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
    for k in tqdm(range(n), disable = not show_progress):   
        if callable(ham):
            evol=u_evol(ham(k),d_sys,d_t,interacting_time_bin_num)

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

#-------------------------
# Efficient Chiral N-Emitter Implementation
#  (1 time and sys bin interacting per local time evolution)
#-------------------------

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

def _separate_sys_bins(i_s, d_sys_total, sbins, bond):
    nbins = []
    sys_num = len(d_sys_total)

    for i in range(sys_num-1):
        sys_i, stemp, i_s = _svd_tensors(i_s, bond, d_sys_total[i], np.prod(d_sys_total[i+1:]))
        nbins.append(sys_i)
        i_s = stemp[:,None,None] * i_s #OC sys bin
        sbins[-1-i].append(sys_i * stemp[None,None,:])

    nbins.append(i_s)
    sbins[0].append(i_s)

    '''
    # Do swaps to take in s0, s1, s2, ... but rearrange to have ..., s2, s1, s0 (s0 on left)
    swap_matrices = np.empty((sys_num, sys_num), dtype=object)
    for i in range(sys_num):
        for j in range(i):
            swap_matrices[i][j] = swap(d_sys_total[j], d_sys_total[i]) # Always have i > j
    # Swap N-1 bins left
    for i in range(sys_num-1):
        # Swap the leftmost bin to right of s0
        for j in range(sys_num-1-i):
            phi1 = ncon([nbins[-2-j], nbins[-1-j], swap_matrices[][]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
            nbins[-2-j], stemp, nbins[-1-j] = _svd_tensors(phi1, bond, d_sys_total[], d_sys_total[k])
            nbins[-2-j] = nbins[-2-j] * stemp[None,None,:] #OC left bin


        # Move OC back to the right
        for j in range(sys_num-1-i):

    '''
    return nbins

def t_evol_nmar_chiral(hams:list[np.ndarray], i_s0:np.ndarray, i_n0:np.ndarray, taus:list[float], params:InputParams, show_progress:bool=True,store_total_sys_flag:bool=False) -> Bins:
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

    show_progress : bool, default: True
        Flag to print a progress bar.

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
    tbins_in.append(states.wg_ground(d_t))

    
    input_field=states.input_state_generator(d_t_total, i_n0)
    n=int(round(tmax/delta_t,0))
    t_k=0
    t_0=0
    evols = [0] * sys_num
    callable_ham_indices = []
    for i in range(len(hams)):
        if not callable(hams[i]):
            evols[i] = u_evol(hams[i],d_sys_total[i],d_t)
        else:
            callable_ham_indices.append(i)
    swap_t_t=swap(d_t,d_t)
    swap_sys_t= []
    swap_t_sys = []
    for i in range(sys_num):
        swap_sys_t.append(swap(d_sys_total[i],d_t))
        swap_t_sys.append(swap(d_t, d_sys_total[i]))
    taus = np.array(taus)
    l_list=np.array(np.round(taus/delta_t, 0).astype(int)) #time steps between system and feedback
    
    # indexed with 0 being out of the WG, greater index is greater depth
    oc_normed_bins_lists = [[states.wg_ground(d_t)] for x in range(sys_num)]
    schmidts = [[np.zeros(1)] for x in range(sys_num)]
    sbins=[[] for x in range(sys_num)] 
    
    i_s=i_s0      

    # Vars stored when storing total/grouped up sys information
    schmidt_sysInput_vs_wgOutput = []
    sbins_total = []
    total_sys_contraction_indices = _ncon_contraction_index_list(sys_num, operator_flag=False)
    
    cumsum_ls = np.cumsum(l_list)
    cumprod_dims = np.cumprod(d_sys_total[:-1])[::-1]

    #first l time bins in vacuum (for no feedback part)    
    
    # Separate the system bins and initialize the feedback loop with bins
    nbins = _separate_sys_bins(i_s, d_sys_total, sbins, bond)
    nbins = _initialize_feedback_loop_chiral(nbins, l_list, d_t, d_sys_total, bond, input_field_generator=None) # Modifies nbins in place
    
    for k in tqdm(range(n), disable = not show_progress):           
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
            for k in range(l_list[-j-1]-1):
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
            if j < feedback_bin_num-1:
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

        #OC is in rightmost sys bin (sys_0)
        if store_total_sys_flag:
            # Create a copy so no need to return
            copy_nbins = copy.deepcopy(nbins[-(sys_num + cumsum_ls[-1]):])

            oc_ind = -1
            # Loop over the various emitter bins to be fetched
            for i in range(feedback_bin_num):
                # Move the OC to the right of the sys bin
                for j in range(cumsum_ls[i]):
                    phi1 = ncon([copy_nbins[oc_ind-1], copy_nbins[oc_ind]], [[-1,-2,1],[1,-3,-4]])
                    copy_nbins[oc_ind-1], stemp, copy_nbins[oc_ind] = _svd_tensors(phi1, bond, copy_nbins[oc_ind-1].shape[1], copy_nbins[oc_ind].shape[1])
                    oc_ind -= 1
                    copy_nbins[oc_ind] = copy_nbins[oc_ind] * stemp[None,None,:]  #OC left bin
                # OC is in bin left of next emitter bin
                # Swap the emitter bin to the right to be grouped together
                for j in range(cumsum_ls[i]):
                    phi1 = ncon([copy_nbins[oc_ind-1], copy_nbins[oc_ind], swap_sys_t[i+1]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
                    copy_nbins[oc_ind-1], stemp, copy_nbins[oc_ind] = _svd_tensors(phi1, bond, d_t, d_sys_total[i+1])
                    copy_nbins[oc_ind] = stemp[:,None,None] * copy_nbins[oc_ind] #OC right bin
                    oc_ind += 1
                oc_ind -= 1

            # Save the collective emitter bin (OC is in rightmost bin)
            # Last stemp is schmidts between emitters and ingoing field vs. rest of WG and outgoing field
            # total_emitter_state.append(ncon(nbins[oc_ind:], inds))
            schmidt_sysInput_vs_wgOutput.append(stemp)
            sbins_total.append(ncon(copy_nbins[oc_ind:], total_sys_contraction_indices))

        t_k += delta_t
        #tlist.append(currT)

    # Put OC in end of nbins chain at end of output (past all sys bins/feedback loops)
    # Have to be careful about dimensionality with the system bins
    # Swap OC from sysbin_0 to sysbin_1 to be next to first delay loop
    phi1 = ncon([nbins[-2], nbins[-1]], [[-1,-2,1],[1,-3,-4]])
    nbins[-2], stemp, nbins[-1] = _svd_tensors(phi1, bond, d_sys_total[1], d_sys_total[0])
    oc_ind = -2
    nbins[-2] = nbins[oc_ind] * stemp[None,None,:]  #OC left bin

    for j in range(feedback_bin_num):
        # Move OC into next chain of feedback time bins, past the system bin
        phi1 = ncon([nbins[oc_ind-1], nbins[oc_ind]], [[-1,-2,1],[1,-3,-4]])
        nbins[oc_ind-1], stemp, nbins[oc_ind] = _svd_tensors(phi1, bond, d_t, d_sys_total[j+1])
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
            nbins[oc_ind-1], stemp, nbins[oc_ind] = _svd_tensors(phi1, bond, d_sys_total[j+2], d_t)
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
    result = Bins(system_states=sbins,output_field_states=oc_normed_bins_lists, input_field_states=tbins_in,
        correlation_bins=nbins,schmidt=schmidts)

    if store_total_sys_flag:
        return result, sbins_total, schmidt_sysInput_vs_wgOutput
    else:
        return result

#-------------------------
# Efficient Symmetrical N-Emitter Implementation
#  (2 time and sys bins interacting per local time evolution)
#-------------------------
from .symmetrical_coupling_helper import Symmetrical_Coupling_Helper

def _reorder_sys_bins_sym_efficient(nbins, d_sys_total, help_obj:Symmetrical_Coupling_Helper, bond):
    # OC starts in rightmost sys bin [-1]
    # Sysbins are initially ordered w.r.t. d_sys_total dims (N-1, N-2, N-3, ..., 2, 1, 0)
    # Want to reorder to (..., N-3, 2, N-2, 1, N-1, 0)
    # Need to swap N-1 to pos. -2, N2 to pos. -4, etc.

    # Creation of swap matrices (indexed in sys_dim order, as this is prior to sys_bin reordering)
    swap_loops = int((help_obj.sys_num - 1)/2)
    moved_indices = 2*np.arange(1, swap_loops+1) - 1 #(in reversed indexing)

    swap_matrices = np.empty((help_obj.sys_num, help_obj.sys_num), dtype=object)
    for loop_num, i in enumerate(range(help_obj.sys_num-1, help_obj.sys_num-1-swap_loops,-1)):
        for j in range(loop_num+1, i):
            swap_matrices[i][j] = swap(d_sys_total[i], d_sys_total[j]) # Always have i > j

    # Loop over bins that will be brought to the right
    for i in range(swap_loops):
        # Move OC to the leftmost bin (one that will be moved), OC starts in bin 2*i from right end
        for j in range(2*i, help_obj.sys_num-1):
            indexr = -1-j; indexl = -2-j
            right_oc_bin = nbins[indexr]
            left_bin = nbins[indexl]
            contraction=ncon([left_bin,right_oc_bin],[[-1,-2,2],[2,-3,-4]])
            left_bin, stemp, right_bin = _svd_tensors(contraction, bond, d_sys_total[j+1-i], d_sys_total[j-i])
            left_oc_bin = left_bin * stemp[None,None,:]
            
            nbins[indexr] = right_bin
            nbins[indexl] = left_oc_bin
            
        # Move this OC bin to the right to correct location (starts at 0th bin everytime)
        for j in range(help_obj.sys_num - moved_indices[i] - 1):
            swapping_sys_ind = help_obj.sys_num - 1 - i
            phi1 = ncon([nbins[j], nbins[j+1], swap_matrices[swapping_sys_ind][swapping_sys_ind-1-j]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
            nbins[j], stemp, nbins[j+1] = _svd_tensors(phi1, bond, d_sys_total[swapping_sys_ind-1-j], d_sys_total[swapping_sys_ind])
            nbins[j+1] = stemp[:,None,None] * nbins[j+1] #OC right bin

        # Move the OC 1 index to the left (to be in an even indexed bin from the right end, prepares for next loop)
        indexr = -1-moved_indices[i]; indexl = indexr - 1
        right_oc_bin = nbins[indexr]
        left_bin = nbins[indexl]
        contraction=ncon([left_bin,right_oc_bin],[[-1,-2,2],[2,-3,-4]])
        left_bin, stemp, right_bin = _svd_tensors(contraction, bond, d_sys_total[i+1], d_sys_total[-1-i])
        left_oc_bin = left_bin * stemp[None,None,:]
        
        nbins[indexr] = right_bin
        nbins[indexl] = left_oc_bin

    # Move OC to end of the chain    
    for j in range(int(not help_obj.odd_end), help_obj.sys_num - 1):
        contraction=ncon([nbins[j],nbins[j+1]],[[-1,-2,2],[2,-3,-4]])
        nbins[j], stemp, nbins[j+1] = _svd_tensors(contraction, bond, help_obj.d_sys_ordered[-1-j], help_obj.d_sys_ordered[-2-j])
        nbins[j+1] = stemp[:,None,None] * nbins[j+1] # OC in right bin 

    return nbins

# Assume OC is in right most system bin
#TODO Currently only supports initializing feedback loops with vacuum bins, in future can expand to have option of incident pulse already inside of the WG.
def _initialize_feedback_loop_sym_efficient(nbins, l_list, d_t, d_sys_total, bond, help_obj:Symmetrical_Coupling_Helper, input_field_generator=None):
    sys_num = len(d_sys_total) 
    
    swap_t_t=swap(d_t,d_t)
    swap_sys_t= []
    for i in range(sys_num):
        swap_sys_t.append(swap(help_obj.d_sys_ordered[i],d_t))

    if input_field_generator is None:
        input_field_generator = states.input_state_generator(d_t)

    # Add in the bins from the right using the generator, and swap to left side of the sys bins appropriately
    # End with OC in the right most bin on each loop, prep for next loop/after loop
    for i in range(help_obj.fback_subchain_num):
        # Loop over the number of bins for that subcain
        if i == 0:
            num_bins_swap_over = help_obj.sys_num
        else:
            num_bins_swap_over = help_obj.sys_num - 2*i + int(help_obj.odd_end)

        for j in range(help_obj.fback_subchain_lengths[-1-i]):
            new_time_bin = next(input_field_generator)
            nbins.append(new_time_bin)

            # Swap the incident time bin num_bins_swap_over to the left down the chain
            for k in range(num_bins_swap_over):
                phi1 = ncon([nbins[-2-k], nbins[-1-k], swap_sys_t[k]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
                nbins[-2-k], stemp, nbins[-1-k] = _svd_tensors(phi1, bond, d_t, help_obj.d_sys_ordered[k])
                nbins[-2-k] = nbins[-2-k] * stemp[None,None,:] #OC Left time bin
            
            # Move OC from time bin to first sys bin on the right
            phi1 = ncon([nbins[-1-num_bins_swap_over], nbins[-num_bins_swap_over]], [[-1,-2,1],[1,-3,-4]])
            nbins[-1-num_bins_swap_over], stemp, nbins[-num_bins_swap_over] = _svd_tensors(phi1, bond, d_t, help_obj.d_sys_ordered[num_bins_swap_over-1])
            nbins[-num_bins_swap_over] = stemp[:,None,None] * nbins[-num_bins_swap_over] #OC In right sys bin

            # Move the OC back to the inital bin to prepare for the next loop
            for k in range(num_bins_swap_over-2, -1, -1):
                phi1 = ncon([nbins[-2-k], nbins[-1-k]], [[-1,-2,1],[1,-3,-4]])
                nbins[-2-k], stemp, nbins[-1-k] = _svd_tensors(phi1, bond, help_obj.d_sys_ordered[k+1], help_obj.d_sys_ordered[k])
                nbins[-1-k] = stemp[:,None,None] * nbins[-1-k] #OC in right sys bin
        
    return nbins


# Setup as pairs of systems of 0,N-1, 1,N-2, 2,N-3, ...
# Have to be careful of edge cases for first pair, when N=2, and the last pair in case N is even vs odd
#TODO Still have to appropriately save the Schmidt coefficients
def t_evol_nmar_sym(hams:list[np.ndarray], i_s0:np.ndarray, i_n0:np.ndarray, taus:list[float],params:InputParams,show_progress:bool=True,store_total_sys_flag:bool=False) -> Bins:
    delta_t = params.delta_t
    tmax=params.tmax
    bond=params.bond_max
    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total


    d_t=np.prod(d_t_total)
    d_sys=np.prod(d_sys_total)
    feedback_bin_num = len(taus)
    sys_num = len(d_sys_total)
    taus = np.array(taus)

    '''
    # In case of half taus specified, infer rest of taus by symmetry requirements
    if len(taus) == int(round((sys_num-1)/2)) and sys_num % 2 != 0:
        taus = np.append(taus, taus[::-1])
    elif len(taus) == int(round(sys_num/2)) and sys_num % 2 == 0:
        taus = np.append(taus, taus[len(taus)-2::-1])

    l_list=np.array(np.round(taus/delta_t, 0).astype(int)) #time steps between system and feedback


    # Check errors
    if not (l_list == l_list[::-1]).all():
        raise ValueError("Delay times tau list must be symmetric over reversal.")
    '''

    l_list = Symmetrical_Coupling_Helper.calc_l_list(taus, d_sys_total, delta_t)
    help_obj = Symmetrical_Coupling_Helper(d_sys_total)
    help_obj.set_fback_subchain_lengths(l_list)

    tbins_in = []
    tbins_in.append(states.wg_ground(d_t))
    schmidt=[]
    
    input_field=states.input_state_generator(d_t_total, i_n0)
    n=int(round(tmax/delta_t,0))
    t_k=0
    t_0=0
    evols = [0] * len(hams)
    callable_ham_indices = []

    # Determine the number of sys/time legs and sys dims for each Hamiltonian (right to left / outside to inside)
    # Eg. 0, 6, 1, 5, 2, 4, 3
    # Also want to store the feedback lengths between each pair of bins
    # Helpful to store these things in an interior class that can get passed around

    #TODO Have to account for two system bin legs, and possibility of single sys/time leg
    for i in range(len(hams)):
        if not callable(hams[i]):
            if i == len(hams)-1 and help_obj.odd_end:
                evols[i] = u_evol_multi_emitter_bins(hams[i], [help_obj.d_sys_ordered[2*i]], d_t, 1)
            else:
                evols[i] = u_evol_multi_emitter_bins(hams[i], [help_obj.d_sys_ordered[2*i],help_obj.d_sys_ordered[2*i+1]], d_t, 2)
        else:
            callable_ham_indices.append(i)
    swap_t_t=swap(d_t,d_t)
    swap_sys_t= []
    swap_t_sys = []
    for i in range(sys_num):
        swap_sys_t.append(swap(help_obj.d_sys_ordered[i],d_t))
        swap_t_sys.append(swap(d_t, help_obj.d_sys_ordered[i]))

        
    # indexed with 0 being out of the WG, greater index is greater depth
    oc_normed_bins_lists = [[states.wg_ground(d_t)] for x in range(sys_num)]
    schmidts = [[np.zeros(1)] for x in range(sys_num)]
    sbins=[[] for x in range(sys_num)] 
    
    i_s=i_s0     

    # Vars stored when storing total/grouped up sys information
    schmidt_sysInput_vs_wgOutput = []
    sbins_total = []
    total_sys_contraction_indices = _ncon_contraction_index_list(sys_num, operator_flag=False)
    
    cumsum_ls = np.repeat(np.cumsum(help_obj.l_list_ordered)[::2], 2)
    if help_obj.odd_end:
        cumsum_ls = cumsum_ls[:-1]
    else:
        cumsum_ls = cumsum_ls[:-2]
    left_indices = (sys_num-2) - 2*np.arange(int(sys_num/2))
    right_indices = 2*np.arange(int((sys_num+1)/2)) + int(sys_num%2 == 0)
    transpose_indices = np.append([left_indices], right_indices) + 1
    transpose_indices = np.insert(transpose_indices, 0, 0)
    transpose_indices = np.append(transpose_indices, sys_num+1)

    # cumprod_dims = np.cumprod(d_sys_total[:-1])[::-1]
 
    
    #first l time bins in vacuum (for no feedback part)    
    
    # Separate the system bins and initialize the feedback loop with bins
    nbins = _separate_sys_bins(i_s, d_sys_total, sbins, bond)
    nbins = _reorder_sys_bins_sym_efficient(nbins, d_sys_total, help_obj, bond)
    nbins = _initialize_feedback_loop_sym_efficient(nbins, l_list, d_t, d_sys_total, bond, help_obj, input_field_generator=None) # Modifies nbins in place
    
    # N=2 is a special case, so programmed separately as its own subroutine
    if sys_num == 2:
        # Precalculations
        time_evol_ncon_indices = _ncon_contraction_index_list(4)

        # Prefetch the first feedback bin so that all iterations start with the feedback bin placed correct
        # Move OC from s0 to s1
        phi1 = ncon([nbins[-2], nbins[-1]], [[-1,-2,1],[1,-3,-4]])
        nbins[-2], stemp, nbins[-1] = _svd_tensors(phi1, bond, help_obj.d_sys_ordered[1], help_obj.d_sys_ordered[0])
        nbins[-2] = nbins[-2] * stemp[None,None,:] #OC in left bin

        # Move OC from s1 to first fback bin
        phi1 = ncon([nbins[-3], nbins[-2]], [[-1,-2,1],[1,-3,-4]])
        nbins[-3], stemp, nbins[-2] = _svd_tensors(phi1, bond, d_t, help_obj.d_sys_ordered[1])
        nbins[-3] = nbins[-3] * stemp[None,None,:] #OC in left bin

        # Loop over moving the OC down the feedback chain
        for j in range(l_list[0]-1):
            phi1 = ncon([nbins[-(4+j)], nbins[-(3+j)]], [[-1,-2,1],[1,-3,-4]])
            nbins[-(4+j)], stemp, nbins[-(3+j)] = _svd_tensors(phi1, bond, d_t, d_t)
            nbins[-(4+j)] = nbins[-(4+j)] * stemp[None,None,:] #OC in left bin

        # Loop over swapping the bin back up the feedback chain
        for j in range(l_list[0]-2,-1,-1):
            phi1 = ncon([nbins[-(4+j)], nbins[-(3+j)], swap_t_t], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
            nbins[-(4+j)], stemp, nbins[-(3+j)] = _svd_tensors(phi1, bond, d_t, d_t)
            nbins[-(3+j)] = stemp[:,None,None] * nbins[-(3+j)] #OC in right bin

        # Move OC from s1 to first fback bin
        phi1 = ncon([nbins[-3], nbins[-2]], [[-1,-2,1],[1,-3,-4]])
        nbins[-3], stemp, nbins[-2] = _svd_tensors(phi1, bond, d_t, help_obj.d_sys_ordered[1])
        nbins[-2] = stemp[:,None,None] * nbins[-2]  #OC in Right bin

        # Move OC from s0 to s1
        phi1 = ncon([nbins[-2], nbins[-1]], [[-1,-2,1],[1,-3,-4]])
        nbins[-2], stemp, nbins[-1] = _svd_tensors(phi1, bond, help_obj.d_sys_ordered[1], help_obj.d_sys_ordered[0])
        nbins[-1] = stemp[:,None,None] * nbins[-1] #OC in right bin

        # Loop over, with the feedback bin beside the emitters at the start of each iteration
        for k in tqdm(range(n), disable = not show_progress):   
            #TODO Have to account for two system bin legs, and possibility of single sys/time leg
            for i in callable_ham_indices:
                evols[i] = u_evol_multi_emitter_bins(hams[i](k), help_obj.d_sys_ordered, d_t, 2)

            in_state = next(input_field)
            nbins.append(in_state)
            t_k += delta_t

            # Swap the incident bin with s0
            phi1 = ncon([nbins[-2], nbins[-1], swap_sys_t[0]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
            nbins[-2], stemp, nbins[-1] = _svd_tensors(phi1, bond, d_t, help_obj.d_sys_ordered[0])
            nbins[-2] = nbins[-2] * stemp[None,None,:] #OC in left bin

            # Swap the incident bin with s1
            phi1 = ncon([nbins[-3], nbins[-2], swap_sys_t[1]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
            nbins[-3], stemp, nbins[-2] = _svd_tensors(phi1, bond, d_t, help_obj.d_sys_ordered[1])
            nbins[-3] = nbins[-3] * stemp[None,None,:] #OC in left bin

            # Store the OC normed input bin
            tbins_in.append(nbins[-3])

            # Time evolve the system
            phi1=ncon(nbins[-4:] + [evols[0]],time_evol_ncon_indices) #tau bin, system bin, future time bin + U operator contraction
        
            # Separate the bins and store OC normed bins, leave OC in old tau bin
            phi1, stemp, nbins[-1] = _svd_tensors(phi1, bond, d_t*d_t*help_obj.d_sys_ordered[1], help_obj.d_sys_ordered[0])
            phi1 = phi1 * stemp[None,None,:] #OC in left bin
            sbins[0].append(stemp[:,None,None] * nbins[-1]) #Store OC normed sys0 bin
            
            phi1, stemp, nbins[-2] = _svd_tensors(phi1, bond, d_t*d_t, help_obj.d_sys_ordered[1])
            phi1 = phi1 * stemp[None,None,:] #OC in left bin
            sbins[1].append(stemp[:,None,None] * nbins[-2]) #Store OC normed sys1 bin

            nbins[-4], stemp, nbins[-3] = _svd_tensors(phi1, bond, d_t, d_t)
            nbins[-4] = nbins[-4] * stemp[None,None,:] #OC in left bin
            oc_normed_bins_lists[0].append(stemp[:,None,None] * nbins[-3]) #Store OC normed output time bin
            oc_normed_bins_lists[1].append(nbins[-4]) #Store OC normed outgoing field bin

            # Swap the old tau bin beyond the end of the chain, l-1 bins left
            for j in range(4, l_list[0]+3):
                phi1 = ncon([nbins[-(1+j)], nbins[-(j)], swap_t_t], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
                nbins[-(1+j)], stemp, nbins[-(j)] = _svd_tensors(phi1, bond, d_t, d_t)
                nbins[-(1+j)] = nbins[-(1+j)] * stemp[None,None,:] #OC in left bin

            # Put OC in one bin to the right, the next tau bin
            phi1 = ncon([nbins[-(l_list[0]+3)], nbins[-(l_list[0]+2)]], [[-1,-2,1],[1,-3,-4]])
            nbins[-(l_list[0]+3)], stemp, nbins[-(l_list[0]+2)] = _svd_tensors(phi1, bond, d_t, d_t)
            nbins[-(l_list[0]+2)] = stemp[:,None,None] * nbins[-(l_list[0]+2)] #OC in right bin

            # Swap the next tau bin back, l-1 bins right
            for j in range(l_list[0]+1, 2, -1):
                phi1 = ncon([nbins[-(1+j)], nbins[-j], swap_t_t], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
                nbins[-(1+j)], stemp, nbins[-j] = _svd_tensors(phi1, bond, d_t, d_t)
                nbins[-j] = stemp[:,None,None] * nbins[-j] #OC in right bin

            # Put the OC in S1  
            phi1 = ncon([nbins[-3], nbins[-2]], [[-1,-2,1],[1,-3,-4]])
            nbins[-3], stemp, nbins[-2] = _svd_tensors(phi1, bond, d_t, help_obj.d_sys_ordered[1])
            nbins[-2] = stemp[:,None,None] * nbins[-2] #OC in right bin

            # Put OC in S0
            phi1 = ncon([nbins[-2], nbins[-1]], [[-1,-2,1],[1,-3,-4]])
            nbins[-2], stemp, nbins[-1] = _svd_tensors(phi1, bond, help_obj.d_sys_ordered[1], help_obj.d_sys_ordered[0])
            nbins[-1] = stemp[:,None,None] * nbins[-1] #OC in right bin

            #OC is in rightmost sys bin (sys_0)
            if store_total_sys_flag:
                sbins_total.append(ncon(nbins[-2:], total_sys_contraction_indices))


        # Move the OC to the first outgoing field and truncate the list to prepare nbins for correlation calculations
        phi1 = ncon([nbins[-2], nbins[-1]], [[-1,-2,1],[1,-3,-4]])
        nbins[-2], stemp, nbins[-1] = _svd_tensors(phi1, bond, help_obj.d_sys_ordered[1], help_obj.d_sys_ordered[0])
        nbins[-2] = nbins[-2] * stemp[None,None,:] #OC in left bin

        phi1 = ncon([nbins[-3], nbins[-2]], [[-1,-2,1],[1,-3,-4]])
        nbins[-3], stemp, nbins[-2] = _svd_tensors(phi1, bond, d_t, help_obj.d_sys_ordered[1])
        nbins[-3] = nbins[-3] * stemp[None,None,:] #OC in left bin

        # Move the OC over the feedback bins
        for j in range(3,l_list[0]+3):
            phi1 = ncon([nbins[-(j+1)], nbins[-j]], [[-1,-2,1],[1,-3,-4]])
            nbins[-(j+1)], stemp, nbins[-j] = _svd_tensors(phi1, bond, d_t, d_t)
            nbins[-(j+1)] = nbins[-(j+1)] * stemp[None,None,:] #OC in left bin

        nbins = nbins[:-(l_list[0]+2)] # Truncate the nbins list

        result =  Bins(system_states=sbins,output_field_states=oc_normed_bins_lists, input_field_states=tbins_in,
                        correlation_bins=nbins,schmidt=schmidts)
        if store_total_sys_flag:
            return result, sbins_total, schmidt_sysInput_vs_wgOutput
        else:
            return result


    # Otherwise have N\neq 2, and just have to careful around the edges, and especially for the N=3 case
    # Precalculate required numbers
    group_num = int((sys_num+1)/2)
    two_groups_int = int(group_num == 2)
    sys_group_num = [2] * group_num
    if help_obj.odd_end:
        sys_group_num[-1] = 1
    
    # Calculate the fback_distances
    fback_distances_j = [0] * group_num # The forward grabbed feedback bin (from the left of current emitter pair)
    fback_distances_k = [0] * group_num # The backward grabbed feedback bin (from the right of current emitter pair)

    # Case of first ones
    if sys_num != 3:
        fback_distances_j[0] = min(help_obj.fback_subchain_lengths[1], 2*help_obj.l_list_ordered[-1])
    else:
        fback_distances_j[0] = help_obj.fback_subchain_lengths[1]
    
    # Case of last ones
    if group_num > 2:
        if not help_obj.odd_end:
            fback_distances_j[-1] = max(0, help_obj.l_list_ordered[-1] - help_obj.l_list_ordered[-2] + 1)
            fback_distances_k[-1] = max(0, help_obj.l_list_ordered[-3] - help_obj.l_list_ordered[-4] + 1)
        else:
            fback_distances_k[-1] = max(0, help_obj.l_list_ordered[-2] - help_obj.l_list_ordered[-3] + 1)
    # Special case if there's only two system groups
    else:
        if not help_obj.odd_end:
            fback_distances_j[-1] = max(0, help_obj.l_list_ordered[-1] - help_obj.l_list_ordered[-2])
        fback_distances_k[-1] = 0

    # Intermediate values
    # loop over calculating the left numbers
    for j in range(1,group_num-1):
        # Special case for the second last group
        if j == group_num -2:
            if help_obj.odd_end:
                fback_distances_j[j] = help_obj.l_list_ordered[2*j+1]
            else:
                fback_distances_j[j] = min(help_obj.fback_subchain_lengths[j+1]-1, 2*help_obj.l_list_ordered[2*j+2])
            continue

        fback_distances_j[j] = min(help_obj.fback_subchain_lengths[j+1]-1, 2*help_obj.l_list_ordered[2*j+2])
        
    for j in range(1,group_num-1):    
        # Special cases for second group
        if j == 1:
            fback_distances_k[j] = 0
            continue
        fback_distances_k[j] = max(0,help_obj.l_list_ordered[2*j-2]-help_obj.l_list_ordered[2*j-3]+1)
        
    # Adjust the second fback_distances_k due to first loop having one taken by first time evolution
    if group_num > 2:
        fback_distances_k[2] = max(0, fback_distances_k[2]-1)

    time_evol_ncon_indices = _ncon_contraction_index_list(4)
    if help_obj.odd_end:
        end_time_evol_ncon_indices = _ncon_contraction_index_list(2)
    else:
        end_time_evol_ncon_indices = _ncon_contraction_index_list(4)


    for k in tqdm(range(n), disable = not show_progress):   
        #TODO Have to account for two system bin legs, and possibility of single sys/time leg, and correct emitter dims
        for i in callable_ham_indices:
            if i == len(hams)-1 and help_obj.odd_end:
                evols[i] = u_evol_multi_emitter_bins(hams[i](k), [help_obj.d_sys_ordered[2*i]], d_t, 1)
            else:
                evols[i] = u_evol_multi_emitter_bins(hams[i](k), [help_obj.d_sys_ordered[2*i],help_obj.d_sys_ordered[2*i+1]], d_t, 2)

        sbin_0 = nbins[-1]
        in_state = next(input_field)
        nbins.append(in_state)
        t_k += delta_t

        #=====================================================
        # Time evolve the first emitter pair, (sN-1,s0)
        #=====================================================

        # Swap the incident field around s0
        # Swap the incident bin with s0
        phi1 = ncon([nbins[-2], nbins[-1], swap_sys_t[0]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
        nbins[-2], stemp, nbins[-1] = _svd_tensors(phi1, bond, d_t, help_obj.d_sys_ordered[0])
        nbins[-2] = nbins[-2] * stemp[None,None,:] #OC in left bin

        # Swap the incident bin with s1
        phi1 = ncon([nbins[-3], nbins[-2], swap_sys_t[1]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
        nbins[-3], stemp, nbins[-2] = _svd_tensors(phi1, bond, d_t, help_obj.d_sys_ordered[1])
        nbins[-3] = nbins[-3] * stemp[None,None,:] #OC in left bin

        # Store the OC normed input bin
        tbins_in.append(nbins[-3])

        # Fetch the feedback bin
        # Move the OC l[0] + 1/2 + j bins left
        for j in range(3,help_obj.l_list_ordered[0] + sys_group_num[1] + fback_distances_j[0] + 3):
            phi1 = ncon([nbins[-(j+1)], nbins[-j]], [[-1,-2,1],[1,-3,-4]])
            nbins[-(j+1)], stemp, nbins[-j] = _svd_tensors(phi1, bond, nbins[-(j+1)].shape[1],nbins[-j].shape[1])
            nbins[-(j+1)] = nbins[-(j+1)] * stemp[None,None,:] #OC in left bin

        # Swap this feedback bin right (j-1) (over time bins), 1/2 over sys bins, l[0] over time bins
        for j in range(-(help_obj.l_list_ordered[0] + sys_group_num[1] + fback_distances_j[0] + 3), -(help_obj.l_list_ordered[0] + sys_group_num[1] + 4)):
            phi1 = ncon([nbins[j], nbins[j+1], swap_t_t], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
            nbins[j], stemp, nbins[j+1] = _svd_tensors(phi1, bond, d_t, d_t)
            nbins[j+1] = stemp[:,None,None] * nbins[j+1] #OC in right bin

        # Swap over 1/2 sys bins
        ind = -(help_obj.l_list_ordered[0] + sys_group_num[1] + 4)
        if sys_num != 3:
            phi1 = ncon([nbins[ind], nbins[ind+1], swap_t_sys[3]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
            nbins[ind], stemp, nbins[ind+1] = _svd_tensors(phi1, bond, help_obj.d_sys_ordered[3], d_t)
            nbins[ind+1] = stemp[:,None,None] * nbins[ind+1] #OC in right bin
            ind += 1

        phi1 = ncon([nbins[ind], nbins[ind+1], swap_t_sys[2]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
        nbins[ind], stemp, nbins[ind+1] = _svd_tensors(phi1, bond, help_obj.d_sys_ordered[2], d_t)
        nbins[ind+1] = stemp[:,None,None] * nbins[ind+1] #OC in right bin


        # Swap over l[0] time bins
        for j in range(-(4 + help_obj.l_list_ordered[0]), -4):
            phi1 = ncon([nbins[j], nbins[j+1], swap_t_t], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
            nbins[j], stemp, nbins[j+1] = _svd_tensors(phi1, bond, d_t, d_t)
            nbins[j+1] = stemp[:,None,None] * nbins[j+1] #OC in right bin

        # Contract and time evolve
        phi1=ncon(nbins[-4:] + [evols[0]],time_evol_ncon_indices) #tau bin, system bin, future time bin + U operator contraction
    
        # Separate the bins and store OC normed bins, leave OC in old tau bin
        phi1, stemp, nbins[-1] = _svd_tensors(phi1, bond, d_t*d_t*help_obj.d_sys_ordered[1], help_obj.d_sys_ordered[0])
        phi1 = phi1 * stemp[None,None,:] #OC in left bin
        sbins[0].append(stemp[:,None,None] * nbins[-1]) #Store OC normed sys0 bin
        
        phi1, stemp, nbins[-2] = _svd_tensors(phi1, bond, d_t*d_t, help_obj.d_sys_ordered[1])
        phi1 = phi1 * stemp[None,None,:] #OC in left bin
        sbins[-1].append(stemp[:,None,None] * nbins[-2]) #Store OC normed sys1 bin

        nbins[-4], stemp, nbins[-3] = _svd_tensors(phi1, bond, d_t, d_t)
        nbins[-4] = nbins[-4] * stemp[None,None,:] #OC in left bin
        oc_normed_bins_lists[0].append(stemp[:,None,None] * nbins[-3]) #Store OC normed output time bin
        oc_normed_bins_lists[-1].append(nbins[-4]) #Store OC normed outgoing field bin

        # Move the old feedback bin ALL the way left over the MPS chain (being careful of swaps over sys bins)
        index = -4
        for loop in range(help_obj.fback_subchain_num):
            if loop == 1:
                bin_num = help_obj.fback_subchain_lengths[loop] - 1
            else:
                bin_num = help_obj.fback_subchain_lengths[loop]
            
            for j in range(index, index-bin_num,-1):
                phi1 = ncon([nbins[j-1], nbins[j], swap_t_t], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
                nbins[j-1], stemp, nbins[j] = _svd_tensors(phi1, bond, d_t, d_t)
                nbins[j-1] = nbins[j-1] * stemp[None,None,:] #OC in left bin
            index -= bin_num

            # At emitters (if present)
            if loop == help_obj.fback_subchain_num - 1:
                break # Are complete (no emitters follow final feedback chain)
            
            phi1 = ncon([nbins[index-1], nbins[index], swap_sys_t[2*loop+2]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
            nbins[index-1], stemp, nbins[index] = _svd_tensors(phi1, bond, d_t,help_obj.d_sys_ordered[2*loop+2])
            nbins[index-1] = nbins[index-1] * stemp[None,None,:] #OC in left bin
            index -= 1

            # Check if this is the last group and have an odd number of emitters.
            if sys_group_num[loop+1] != 1:
                phi1 = ncon([nbins[index-1], nbins[index], swap_sys_t[2*loop+3]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
                nbins[index-1], stemp, nbins[index] = _svd_tensors(phi1, bond, d_t, help_obj.d_sys_ordered[2*loop+3])
                nbins[index-1] = nbins[index-1] * stemp[None,None,:] #OC in left bin
                index -= 1

        #=====================================================
        # Time evolve the last emitter pair, (s_N-1/2, s_int(N-1/2))
        #=====================================================
        # Must be careful, have extra step of grabbing extra bin if N is even

        # Even case
        if not help_obj.odd_end:
            # Move the OC k+1 bins to the right (over time bins, getting to the fback bin)
            for j in range(index, index + fback_distances_j[-1] + 1):
                phi1 = ncon([nbins[j], nbins[j+1]], [[-1,-2,1],[1,-3,-4]])
                nbins[j], stemp, nbins[j+1] = _svd_tensors(phi1, bond, d_t, d_t)
                nbins[j+1] = stemp[:,None,None] * nbins[j+1] #OC in right bin
            index += fback_distances_j[-1] + 1

            # OC is now in Feedback bin, swap the OC to the right to get it right of emitter bins (chain num - (k+1))
            # If there are only two sysgroups need to move one bin fewer
            for j in range(index, index + help_obj.fback_subchain_lengths[-1] - (fback_distances_j[-1] + 1) - two_groups_int):
                phi1 = ncon([nbins[j], nbins[j+1], swap_t_t], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
                nbins[j], stemp, nbins[j+1] = _svd_tensors(phi1, bond, d_t, d_t)
                nbins[j+1] = stemp[:,None,None] * nbins[j+1] #OC in right bin

            index += help_obj.fback_subchain_lengths[-1] - (fback_distances_j[-1] + 1)- two_groups_int

        # Odd cases
        else:
            # Move the OC into the time bin immediately to the left of last emitter bin
            # If there are only two sysgroups need to move one bin fewer
            for j in range(index, index+help_obj.fback_subchain_lengths[-1]- two_groups_int):
                phi1 = ncon([nbins[j], nbins[j+1]], [[-1,-2,1],[1,-3,-4]])
                nbins[j], stemp, nbins[j+1] = _svd_tensors(phi1, bond, d_t, d_t)
                nbins[j+1] = stemp[:,None,None] * nbins[j+1] #OC in right bin
            index += help_obj.fback_subchain_lengths[-1]- two_groups_int

        # Move OC sys_group_num[-1] + (j+1) bins to the right (over emitter and time bins)
        for j in range(index, index + sys_group_num[-1] + fback_distances_k[-1] + 1):
            phi1 = ncon([nbins[j], nbins[j+1]], [[-1,-2,1],[1,-3,-4]])
            nbins[j], stemp, nbins[j+1] = _svd_tensors(phi1, bond, nbins[j].shape[1], nbins[j+1].shape[1])
            nbins[j+1] = stemp[:,None,None] * nbins[j+1] #OC in right bin
        index += sys_group_num[-1] + fback_distances_k[-1] + 1

        # Have OC in appropriate bin, swap left over j bins to be right of emitter bin(s)
        for j in range(index, index - fback_distances_k[-1], -1):
            phi1 = ncon([nbins[j-1], nbins[j], swap_t_t], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
            nbins[j-1], stemp, nbins[j] = _svd_tensors(phi1, bond, d_t, d_t)
            nbins[j-1] = nbins[j-1] * stemp[None,None,:] #OC in left bin

        index -= fback_distances_k[-1]

        # Swap over the emitter bin(s)
        if sys_group_num[-1] == 2:
            phi1 = ncon([nbins[index-1], nbins[index], swap_sys_t[-2]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
            nbins[index-1], stemp, nbins[index] = _svd_tensors(phi1, bond, d_t, help_obj.d_sys_ordered[-2])
            nbins[index-1] = nbins[index-1] * stemp[None,None,:] #OC in left bin
            index -= 1
        phi1 = ncon([nbins[index-1], nbins[index], swap_sys_t[-1]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
        nbins[index-1], stemp, nbins[index] = _svd_tensors(phi1, bond, d_t, help_obj.d_sys_ordered[-1])
        nbins[index-1] = nbins[index-1] * stemp[None,None,:] #OC in left bin
        index -= 1

        # Contract and separate bins (3 vs 4 bins dependent on even vs odd N)
        if help_obj.odd_end:
            #  2 bins contraction, index/OC in time bin
            phi1=ncon(nbins[index:index+2] + [evols[-1]], end_time_evol_ncon_indices) #tau bin, system bin, future time bin + U operator contraction
        
            # Separate the bins and store OC normed bins, leave OC in old tau bin
            nbins[index], stemp, nbins[index+1] = _svd_tensors(phi1, bond, d_t, help_obj.d_sys_ordered[-1])
            nbins[index] = nbins[index] * stemp[None,None,:] #OC in left bin

            sbins[help_obj.ordered_indices[-1]].append(stemp[:,None,None] * nbins[index+1]) #Store OC normed sys end bin
            oc_normed_bins_lists[help_obj.ordered_indices[-1]].append(nbins[index]) #Store OC normed outgoing field bin

        else:
            index -= 1
            # Regular 4 bins, index/OC is right time bin
            phi1=ncon(nbins[index:index+4] + [evols[-1]], end_time_evol_ncon_indices) #tau bin, system bin, future time bin + U operator contraction
        
            # Separate the bins and store OC normed bins, leave OC in old tau bin
            phi1, stemp, nbins[index+3] = _svd_tensors(phi1, bond, d_t*d_t*help_obj.d_sys_ordered[-1], help_obj.d_sys_ordered[-2])
            phi1 = phi1 * stemp[None,None,:] #OC in left bin
            sbins[help_obj.ordered_indices[-2]].append(stemp[:,None,None] * nbins[index+3]) #Store OC normed sys second from end bin
            
            phi1, stemp, nbins[index+2] = _svd_tensors(phi1, bond, d_t*d_t, help_obj.d_sys_ordered[-1])
            phi1 = phi1 * stemp[None,None,:] #OC in left bin
            sbins[help_obj.ordered_indices[-1]].append(stemp[:,None,None] * nbins[index+2]) #Store OC normed sys end bin

            nbins[index], stemp, nbins[index+1] = _svd_tensors(phi1, bond, d_t, d_t)
            nbins[index] = nbins[index] * stemp[None,None,:] #OC in left bin
            oc_normed_bins_lists[help_obj.ordered_indices[-2]].append(stemp[:,None,None] * nbins[index+1]) #Store OC normed output time bin
            oc_normed_bins_lists[help_obj.ordered_indices[-1]].append(nbins[index]) #Store OC normed outgoing field bin

        #=====================================================
        # Iterate: Work backwards over all other emitter pairs, time evolving them, (sN-1-i, si)
        #=====================================================
        # Mostly the same for all cases, just have to be a little careful in case of the second last and second emitter pair
        for l in range(group_num-2, 0, -1):
            
            # Move the OC j bins to the left of index (current OC) over time bins
            for j in range(index, index - fback_distances_j[l], -1):
                phi1 = ncon([nbins[j-1], nbins[j]], [[-1,-2,1],[1,-3,-4]])
                nbins[j-1], stemp, nbins[j] = _svd_tensors(phi1, bond, d_t, d_t)
                nbins[j-1] = nbins[j-1] * stemp[None,None,:] #OC in left bin
            index -= fback_distances_j[l]

            # Swap this OC bin (j-1) + sys_group_num[l+1] time bins right
            for j in range(index, index + (fback_distances_j[l]-1) + sys_group_num[l+1] ):
                phi1 = ncon([nbins[j], nbins[j+1], swap_t_t], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
                nbins[j], stemp, nbins[j+1] = _svd_tensors(phi1, bond, d_t, d_t)
                nbins[j+1] = stemp[:,None,None] * nbins[j+1] #OC in right bin
            index += (fback_distances_j[l]-1) + sys_group_num[l+1]

            # Swap the OC bin over the sys_group_num[l+1] (either 1 or 2) emitter bins right
            if sys_group_num[l+1] == 2:
                phi1 = ncon([nbins[index], nbins[index+1], swap_t_sys[2*l+3]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
                nbins[index], stemp, nbins[index+1] = _svd_tensors(phi1, bond, help_obj.d_sys_ordered[2*l+3],d_t)
                nbins[index+1] = stemp[:,None,None] * nbins[index+1] #OC in right bin
                index += 1
            phi1 = ncon([nbins[index], nbins[index+1], swap_t_sys[2*l+2]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
            nbins[index], stemp, nbins[index+1] = _svd_tensors(phi1, bond, help_obj.d_sys_ordered[2*l+2], d_t)
            nbins[index+1] = stemp[:,None,None] * nbins[index+1] #OC in right bin
            index += 1


            # Swap this OC bin over the tau_i + tau_{N-i-1}-1 bins to the right (placing it to the left of the relevant emitter bins)
            # Move 1 fewer right on last case, already grabbed a bin for the time evolution for first emitter pair, therefore have a shorter chain
            for j in range(index, index + help_obj.fback_subchain_lengths[l]-1 - int(l==1) ):
                phi1 = ncon([nbins[j], nbins[j+1], swap_t_t], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
                nbins[j], stemp, nbins[j+1] = _svd_tensors(phi1, bond, d_t, d_t)
                nbins[j+1] = stemp[:,None,None] * nbins[j+1] #OC in right bin
            index += help_obj.fback_subchain_lengths[l]-1 - int(l==1)

            # Move the OC 2 + (k+1) bins to the right (over 2 emitter bins)
            for j in range(index, index + fback_distances_k[l] + 3 ):
                phi1 = ncon([nbins[j], nbins[j+1]], [[-1,-2,1],[1,-3,-4]])
                nbins[j], stemp, nbins[j+1] = _svd_tensors(phi1, bond, nbins[j].shape[1], nbins[j+1].shape[1])
                nbins[j+1] = stemp[:,None,None] * nbins[j+1] #OC in right bin
            index += fback_distances_k[l] + 3

            # Swap the OC bin k bins to the left over time bins
            for j in range(index, index - fback_distances_k[l], -1):
                phi1 = ncon([nbins[j-1], nbins[j], swap_t_t], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
                nbins[j-1], stemp, nbins[j] = _svd_tensors(phi1, bond, d_t, d_t)
                nbins[j-1] = nbins[j-1] * stemp[None,None,:] #OC in left bin
            index -= fback_distances_k[l]

            # Swap the OC bin over the 2 current, relevant emitter bins (places it immediately left of the emitters)
            phi1 = ncon([nbins[index-1], nbins[index], swap_sys_t[2*l]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
            nbins[index-1], stemp, nbins[index] = _svd_tensors(phi1, bond, d_t, help_obj.d_sys_ordered[2*l])
            nbins[index-1] = nbins[index-1] * stemp[None,None,:] #OC in left bin
            index -= 1

            phi1 = ncon([nbins[index-1], nbins[index], swap_sys_t[2*l+1]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
            nbins[index-1], stemp, nbins[index] = _svd_tensors(phi1, bond, d_t, help_obj.d_sys_ordered[2*l+1])
            nbins[index-1] = nbins[index-1] * stemp[None,None,:] #OC in left bin
            index -= 1


            # Contract and time evolve the bins, saving OC normed bin and leaving OC in leftmost time bin
            index -= 1
            phi1=ncon(nbins[index:index+4] + [evols[l]],time_evol_ncon_indices) #tau bin, system bin, future time bin + U operator contraction
        
            # Separate the bins and store OC normed bins, leave OC in old tau bin
            phi1, stemp, nbins[index+3] = _svd_tensors(phi1, bond, d_t*d_t*help_obj.d_sys_ordered[2*l+1], help_obj.d_sys_ordered[2*l])
            phi1 = phi1 * stemp[None,None,:] #OC in left bin
            sbins[help_obj.ordered_indices[2*l]].append(stemp[:,None,None] * nbins[index+3]) #Store OC normed sys second from end bin
            
            phi1, stemp, nbins[index+2] = _svd_tensors(phi1, bond, d_t*d_t, help_obj.d_sys_ordered[2*l+1])
            phi1 = phi1 * stemp[None,None,:] #OC in left bin
            sbins[help_obj.ordered_indices[2*l+1]].append(stemp[:,None,None] * nbins[index+2]) #Store OC normed sys end bin

            nbins[index], stemp, nbins[index+1] = _svd_tensors(phi1, bond, d_t, d_t)
            nbins[index] = nbins[index] * stemp[None,None,:] #OC in left bin
            oc_normed_bins_lists[help_obj.ordered_indices[2*l]].append(stemp[:,None,None] * nbins[index+1]) #Store OC normed output time bin
            oc_normed_bins_lists[help_obj.ordered_indices[2*l+1]].append(nbins[index]) #Store OC normed outgoing field bin


        #=====================================================
        # Move OC back into the s0 bin
        #=====================================================
        # Move the OC over emitter/time bins, (sys_group_num[1]*2 - 1) + l[0] + 2 bins right (end in index -1)

        for j in range(-(2*sys_group_num[1] + help_obj.fback_subchain_lengths[0] + 2), -1, 1):
            phi1 = ncon([nbins[j], nbins[j+1]], [[-1,-2,1],[1,-3,-4]])
            nbins[j], stemp, nbins[j+1] = _svd_tensors(phi1, bond, nbins[j].shape[1], nbins[j+1].shape[1])
            nbins[j+1] = stemp[:,None,None] * nbins[j+1] #OC in right bin
        index = -1

        #OC is in rightmost sys bin (sys_0)
        if store_total_sys_flag:
            # Create a copy so no need to return
            copy_nbins = copy.deepcopy(nbins[-(sys_num + cumsum_ls[-1]):])

            # Move the OC into the left system bin, S_{N-1}
            phi1 = ncon([copy_nbins[-2], copy_nbins[-1]], [[-1,-2,1],[1,-3,-4]])
            copy_nbins[-2], stemp, copy_nbins[-1] = _svd_tensors(phi1, bond, help_obj.d_sys_ordered[1],help_obj.d_sys_ordered[0])
            copy_nbins[-2] = copy_nbins[-2] * stemp[None,None,:]  #OC left bin

            oc_ind = -2
            # Loop over the various emitter bins to be fetched
            for i in range(sys_num-2):
                # Move the OC to the right of the sys bin
                for j in range(cumsum_ls[i]):
                    phi1 = ncon([copy_nbins[oc_ind-1], copy_nbins[oc_ind]], [[-1,-2,1],[1,-3,-4]])
                    copy_nbins[oc_ind-1], stemp, copy_nbins[oc_ind] = _svd_tensors(phi1, bond, copy_nbins[oc_ind-1].shape[1], copy_nbins[oc_ind].shape[1])
                    oc_ind -= 1
                    copy_nbins[oc_ind] = copy_nbins[oc_ind] * stemp[None,None,:]  #OC left bin
                # OC is in bin left of next emitter bin
                # Swap the emitter bin to the right to be grouped together
                for j in range(cumsum_ls[i]):
                    phi1 = ncon([copy_nbins[oc_ind-1], copy_nbins[oc_ind], swap_sys_t[i+1]], [[-1,2,1],[1,3,-4], [-2,-3,2,3]])
                    copy_nbins[oc_ind-1], stemp, copy_nbins[oc_ind] = _svd_tensors(phi1, bond, d_t, d_sys_total[i+2])
                    copy_nbins[oc_ind] = stemp[:,None,None] * copy_nbins[oc_ind] #OC right bin
                    oc_ind += 1
                oc_ind -= 1

            # Save the collective emitter bin (OC is in rightmost bin)
            # Last stemp is schmidts between emitters and ingoing field vs. rest of WG and outgoing field
            # total_emitter_state.append(ncon(nbins[oc_ind:], inds))
            schmidt_sysInput_vs_wgOutput.append(stemp)
            # Reindex the sysbins to be 0-> N from right to left before contraction
            grouped_sys_bins = np.transpose(copy_nbins[oc_ind:], transpose_indices)

            sbins_total.append(ncon(grouped_sys_bins, total_sys_contraction_indices))



    # Move OC beyond all feedback loops and truncate nbins
    # Move it N + \sum_i l[i] bins left (starts in -1)
    truncation_num = -(sys_num + np.sum(l_list))
    for j in range(-1, truncation_num, -1):
        phi1 = ncon([nbins[j-1], nbins[j]], [[-1,-2,1],[1,-3,-4]])
        nbins[j-1], stemp, nbins[j] = _svd_tensors(phi1, bond, nbins[j-1].shape[1], nbins[j].shape[1])
        nbins[j-1] = nbins[j-1] * stemp[None,None,:] #OC in left bin


    #truncation_number = np.sum(l_list) + sys_num
    nbins = nbins[:truncation_num]

    result = Bins(system_states=sbins,output_field_states=oc_normed_bins_lists, input_field_states=tbins_in,
        correlation_bins=nbins,schmidt=schmidts)

    if store_total_sys_flag:
        return result, sbins_total, schmidt_sysInput_vs_wgOutput
    else:
        return result
