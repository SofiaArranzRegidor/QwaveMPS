#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains initial states, coupling function 
and pulse constructors for the waveguide and the TLSs.

Note 
----    
It requires the module ncon (pip install --user ncon)

"""

import numpy as np
import scipy as sci
from ncon import ncon
from collections.abc import Iterator
from typing import Callable
from QwaveMPS import simulation as sim
from QwaveMPS.parameters import InputParams

__all__ = ['wg_ground', 'tls_ground', 'tls_excited', 'vacuum', 'basis', 'input_state_generator', 'coupling',
            'tophat_envelope', 'gaussian_envelope','exp_decay_envelope',
            'normalize_pulse_envelope_integral', 'normalize_pulse_envelope','left_normalize_bins',
            'fock_pulse', 'create_pulse', 'calc_coherent_val', 'coherent_pulse', 'addMPSs']

#--------------------
#Initial basic states
#--------------------

def wg_ground(d_t:int, bond0:int=1) -> np.ndarray:
    """
    Waveguide vacuum state for a single time bin.

    Parameters
    ----------
    d_t : int
        Size of the truncated Hilbert space of the light field.

    bond0 : int, default: 1
        Initial size of the bond dimension.
    
    Returns
    -------
    state : ndarray
        Waveguide vacuum state.
    """ 
    state = np.zeros([bond0,d_t,bond0],dtype=complex) 
    state[:,0,:]=1.
    return state

def tls_ground(bond0:int=1) -> np.ndarray:
    """
    Two level system ground state tensor.

    Parameters
    ----------
    bond0 : int, default: 1
        Initial size of the bond dimension.
    
    Returns
    -------
    state : ndarray
        Ground state of the two level system.
    """ 
    i_s = np.zeros([bond0,2,bond0],dtype=complex) 
    i_s[:,0,:]=1.
    return i_s
    
def tls_excited(bond0:int=1) -> np.ndarray:
    """
    Two level system excited state tensor.

    Parameters
    ----------
    bond0 : int, default: 1
        Initial size of the bond dimension.
    
    Returns
    -------
    state : ndarray
        Excited state of the two level system.
    """ 
    i_s = np.zeros([bond0,2,bond0],dtype=complex) 
    i_s[:,1,:]=1.
    return i_s

def vacuum(time_length:float, params:InputParams) -> list[np.ndarray]:
    """
    Produces an array of vacuum time bins for a given time_length.

    Parameters
    ----------

    time_length : float
        Length of the vacuum pulse (units of inverse coupling).

    params : InputParams
        Class containing the input parameters.

    Returns
    -------
    state : list[np.ndarray]
        List of vacuum states for time_length.
    """ 
    delta_t = params.delta_t
    d_t_total = params.d_t_total
    
    bond0=1
    l = int(round(time_length/delta_t, 0))
    d_t=np.prod(d_t_total)
    
    
    return [wg_ground(d_t, bond0) for i in range(l)]

def basis(dim:int, n:int=0) -> np.ndarray:
    """
    Generates a basis vector for a Hilbert space of size 'dim'.
    
    Parameters
    ----------  
    dim : int
        Size of the Hilbert space.

    n : int, default: 0
        Integer label of the basis vector.

    Returns
    -------
    basis_vec : np.ndarray
        The n^th basis vector for the Hilbert space of size dim.
    """ 
    basis_vec = np.zeros(dim, dtype=complex)
    basis_vec[n] = 1
    return basis_vec


def input_state_generator(d_t_total:list[int], input_bins:list[np.ndarray]=None, bond0:int=1, default_state=None) -> Iterator[np.ndarray]:
    """
    Creates an iterator (generator) for the input field states of the waveguide.

    Parameters
    ----------
    d_t_total : list[int]
        List of sizes of the photonic Hilbert spaces.

    input_bins : list[np.ndarray], default: None
        List of time bins describing the input field state.
        
    bond0 : int, default: 1
        Size of the initial bond dimension.

    default_state : ndarray, default: None
        Default time bin state yielded as an input state after all input_bins are exhusted. 
        If None then vacuum states are yielded.
    
    Returns
    -------
    gen : Generator
        A generator for the input field time bins.
    """ 
    d_t = np.prod(d_t_total)
    if input_bins is None:
        input_bins = []
        
    for i in range(len(input_bins)):
        yield input_bins[i]
    
    # After all specified input bins are yielded, start inputting vacuum bins
    if default_state is None:
        while True:
            yield wg_ground(d_t, bond0)
    else:
        while True:
            yield default_state

def coupling(coupl:str='symmetrical', gamma:float=1,gamma_r=None,gamma_l=None) -> tuple[float,float]:
    """ 
    Return (gamma_l, gamma_r) given a coupling specification.

    It can be 'symmetrical', 'chiral_r', 'chiral_l', 'other'
    For 'other', provide gamma_l and gamma_r explicitly.
    
    Parameters
    ----------
    coupl : {'symmetrical', 'chiral_r', 'chiral_l', 'other'}, default: 'symmetrical'
       Coupling option.
    
    gamma : float, default:1
        Total coupling. Code in units of coupling, hence, the default is 1.
    
    gamma_r : None/float, default: None
        Left coupling. If coupl = 'other' define explicitly.
        
    gamma_l : None/float, default: None
        Right coupling. If coupl = 'other' define explicitly.
    
    Returns
    -------
    gamma_l,gamma_r : tuple[float,float]
        Values of the left and right coupling
    """   
    if coupl == 'chiral_r': 
        gamma_r=gamma
        gamma_l=gamma - gamma_r
    elif coupl == 'chiral_l': 
        gamma_l=gamma
        gamma_r=gamma - gamma_l
    elif coupl == 'symmetrical':
        gamma_r=gamma/2.
        gamma_l=gamma - gamma_r
    elif coupl == 'other':
        gamma_r=gamma_r
        gamma_l=gamma_l

    else:
        raise ValueError("Coupling for the function must be 'chiral_r', 'chiral_l', or 'symmetrical'")

    return gamma_l,gamma_r       

#----------------------
#Pulse envelope helpers
#----------------------

def tophat_envelope(pulse_time:float, params:InputParams)->np.ndarray:
    """
    Create an unnormalized top hat pulse envelope given by the time length of the pulse.

    Parameters
    ----------
    pulse_time : float
        Duration time of the pulse (units of inverse coupling).

    params : InputParams
        Class containing the input parameters.
        
    Returns
    -------
    pulse_envelope : list[float]
        List of amplitude values of the pulse envelope.
    """ 
    delta_t=params.delta_t
    m = int(round(pulse_time/delta_t))
    return np.ones(m)

def gaussian_envelope(pulse_time:float, params:InputParams, gaussian_width:float, gaussian_center:float)->np.ndarray:
    """
    Create a gaussian pulse envelope given by the time length of the pulse 
    and the mean and standard deviation parameters.

    Parameters
    ----------
    pulse_time : float
        Duration time of the pulse (units of inverse coupling).

    params:InputParams
        Class containing the input parameters
        
    gaussian_width : float
        Variance of the gaussian (units of inverse coupling).
    
    gaussian_center : float
        Mean of the gaussian (units of inverse coupling).

    Returns
    -------
    pulse_envelope : np.ndarray[float]
        List of amplitude values of the pulse envelope.
    """ 
    
    delta_t=params.delta_t
    
    m = int(round(pulse_time/delta_t,0))
    times = np.arange(0, m) * delta_t
    diffs = times - gaussian_center
    exponent = - (diffs ** 2) / (2 * gaussian_width ** 2) 

    pulse_envelope = np.exp(exponent) / (gaussian_width * np.sqrt(2*np.pi))     
    return pulse_envelope

def exp_decay_envelope(pulse_time:float, params:InputParams, decay_rate:float, decay_center:float=0)->np.ndarray:
    """
    Create a exponential decay pulse envelope (unnormalized) given by the time length of the pulse 
    and the decay rate and decay center parameters.

    Parameters
    ----------
    pulse_time : float
        Duration time of the pulse (units of inverse coupling).

    params:InputParams
        Class containing the input parameters

    decay_rate : float
        Decay rate of the exponential.
    
    decay_center : float
        Time center/offset of the exponential decay function.
        
    Returns
    -------
    pulse_envelope : list[float]
        List of amplitude values of the pulse envelope.

    """ 
    delta_t=params.delta_t
    m = int(round(pulse_time/delta_t, 0))
    times = np.arange(0, m*delta_t, delta_t)
        
    time_diffs = times - decay_center
    
    pulse_envelope = np.exp(-time_diffs * decay_rate)
    return pulse_envelope

def normalize_pulse_envelope_integral(delta_t:float, pulse_env:np.ndarray)->np.ndarray:
    """
    Normalizes a given pulse envelope so that the integral of the square magnitude is 1.

    Parameters
    ----------
    delta_t : float
        Time step size for the simulation.
        
    pulse_env : np.ndarray[float]
        Time dependent pulse envelope that is being normalized.

    Returns
    -------
    pulse_env : np.ndarray[float]
        The normalized time dependent pulse envelope.

    """ 
    norm_factor = np.sum(np.abs(np.array(pulse_env))**2) * delta_t
    pulse_env /= np.sqrt(norm_factor)
    return pulse_env

def normalize_pulse_envelope(pulse_env:np.ndarray)->np.ndarray:
    """
    Normalizes a given pulse envelope so that the sum of the square elements is 1.

    Parameters
    ----------
    delta_t : float
        Time step size for the simulation.
        
    pulse_env : np.ndarray[float]
        Time dependent pulse envelope that is being normalized.

    Returns
    -------
    pulse_env : np.ndarray[float]
        The normalized time dependent pulse envelope.

    """ 
    norm_factor = np.sum(np.abs(np.array(pulse_env))**2)
    pulse_env /= np.sqrt(norm_factor)
    return pulse_env

#-------------------------
# Pulse Helper Functions
#-------------------------
def left_normalize_bins(bins:list[np.ndarray], bond_max:int) -> list[np.ndarray]:        
    """
    Given an MPS converts it to a left normalized form (OC in bin 0)

    Parameters
    ----------
    bins : list[np.ndarray]
        List of ordered bins in the MPS.

    bond_max : int
        The maximum bond dimension of the MPS.

    Returns
    -------
    bins_l_normed : list[np.ndarray]
        The same MPS in a left normalized form.
        
    Examples
    -------- 

    """ 
    m = len(bins)
    bin_dim = bins[0].shape[1]

    for k in range(m-1,0,-1):
        curr_bin = ncon([bins[k-1], bins[k]],[[-1,-2,1],[1,-3,-4]])
        curr_bin, stemp, i_n_r = sim._svd_tensors(curr_bin, bond_max, bin_dim, bin_dim)
        curr_bin = curr_bin * stemp[None,None,:]
        bins[k] = i_n_r
        bins[k-1] = curr_bin
        
    return bins

def _matrix_text_print(ap1:np.ndarray, apm:np.ndarray, ak:np.ndarray, time_bin_dim:int):
    """
    Prints out the initial, end, and a middle matrix for a given MPS factorization.

    Parameters
    ----------
    ap1 : np.ndarray
        First tensor in the factorization
    
    apm : np.ndarray
        The final tensor in the MPS factorization.

    ak : np.ndarray
        Some middle tensor in the MPS factorization.
    
    time_bin_dim : int
        The size of the physical index for the tensors in the waveguide. This is the product of all channel dimensions.
        
    Examples
    -------- 

    """ 
    
    print('A1:', ap1.shape)
    for i in range(time_bin_dim):
        print(np.real(ap1[:,i,:]))
    print('='*50)
    print('ApM:', apm.shape)
    for i in range(time_bin_dim):
        print(np.real(apm[:,i,:]))
    print('='*50)
    print('Ak:', ak.shape)
    for i in range(time_bin_dim):
        print(np.real(ak[:,i,:]))

def _single_pulse_env_preparation(pulse_envs:list[list[complex]], channel_num:int, pulse_bin_num:int) -> list[list[complex]]:
    """
    Prepares the pulse envelope so that they are tail truncated or padded (with 0's) to be of 
    length of the given pulse time. Also normalizes each pulse envelope to ensure it they are
    properly normalized
    Only works for single envelope (per channel) states, such as indistinguishable photon Fock states (i.e. f(t), not f(t_1,t_2)).

    Parameters
    ----------
    pulse_env : list[list[complex]]
        Time dependent pulse envelope for the incident pulse (can be unnormalized). 
        If None, uses a tophat pulse for the duration of the pulse_time.

    channel_num : int
        Number of channels in the waveguide. Given by the length of d_t_total.
    
    pulse_bin_num : int
        Number of bins in the pulse. Determined by the pulse_time and delta_t.

    Returns
    -------
    pulse_envs_normed : list[list[complex]]
        A list of normalized pulse envelopes of proper lengths.
        
    Examples
    -------- 

    """ 
    # Normalize the pulse envelopes
    for i in range(channel_num):
        # Default to single top hat pulse
        if pulse_envs[i] is None:
            pulse_envs[i] = np.ones(pulse_bin_num)
        else:
            pulse_envs[i] = np.array(pulse_envs[i])
        pulse_envs[i] = normalize_pulse_envelope(pulse_envs[i])
    
    # Pad envelopes as necessary to be of length m
    for i in range(channel_num):
        pulse_envs[i] = np.append(pulse_envs[i], [0] * (pulse_bin_num-len(pulse_envs[i])))
    return pulse_envs

def _tensor_outer_rank3(tensor_list:list[np.ndarray])->np.ndarray:
    """
    For a list of rank 3 tensors, takes the 3-dimensional outer/kronecker product of all of the tensors,
    working leftwise with the list in terms of the ordering of the tensor spaces.

    Parameters
    ----------
    tensor_list : list[np.ndarray]
        List of rank 3 tensors.

    Returns
    -------
    pulse_envs_normed : list[list[complex]]
        A list of normalized pulse envelopes of proper lengths.
        
    Examples
    -------- 

    """ 
    result = tensor_list[0]
    for t in tensor_list[1:]:
        result = np.einsum('ijk,lmn->iljmkn', result, t).reshape(
            result.shape[0]*t.shape[0], result.shape[1]*t.shape[1], result.shape[2]*t.shape[2]
        )
    return result

# Contractions placed directly into create_pulse function
def _contract_alpha(alpha:np.ndarray, ak:np.ndarray) -> np.ndarray:
    """
    Matrix multiplies each matrix ak[:,i,:] by alpha from the left to get the first MPS tensor.
    
    Parameters
    ----------
    alpha : np.ndarray
        Weight of initial bins values for the MPS. Associated with the initial states of the finite automata.

    ak : np.ndarray
        MPS factorized matrix for the first bin, to be contracted with alpha.

    Returns
    -------
    a1 : np.ndarray
        The first bin of the MPS factorization, with first dimension of 1.
        
    Examples
    -------- 

    """ 
    return np.einsum('iq,qjk->ijk', alpha, ak)

def _contract_omega(omega:np.ndarray, ak:np.ndarray) -> np.ndarray:
    """
    Matrix multiplies each matrix ak[:,i,:] by omega from the right to get the last MPS tensor.
    
    Parameters
    ----------
    omega : np.ndarray
        Weight of final dimensions values for the MPS. Associated with the final states of the finite automata.

    ak : np.ndarray
        MPS factorized matrix for the last bin, to be contracted with omega.

    Returns
    -------
    am : np.ndarray
        The final bin of the MPS factorization, with last dimension of 1.
        
    Examples
    -------- 

    """ 
    return np.einsum('ijk,kq->ijq', ak, omega)

#-------------------------
# Create the Pulse based on given matrix scheme
#-------------------------
#TODO Needs optimization work. Lot of room for performance improvements here.
def create_pulse(pulse_envs:list[list[complex]],pulse_time:float,params:InputParams, pulse_alphaOmega:Callable, alphaOmega_args:list[tuple], pulse_ak:Callable, ak_args:list[tuple], bond0:int=1)->list[np.ndarray]:    
    """
    Creates a pulse input field MPS with a pulse envelope. 
    Created quantum pulse is dependent on the matrix_args, and the alpha, omega, and ak matrices passed.

    Parameters
    ----------
    pulse_envs : list[list[complex]]
        List of time dependent pulse envelopes for each channel/tensorspace in the waveguide (d_t_total).
        If None uses a top-hat pulse (constant amplitude).

    pulse_time : float
        Time length of the pulse (units of inverse coupling). 
        If the pulse envelope is of greater length it will be truncated from the tail.

    params : InputParams
        Class containing the input parameters
    
    pulse_alphaOmega : Callable
        Function to yield the alpha/omega vectors used to contract with the first/last tensors.
        These are dependent on the weights of the input state(s) and final state(s).

    alphaOmega_args : list[tuple]
        List of arguments associated to the callable pulse_alphaOmega() function corresponding to each channel/tensorspace in the waveguide.
        
    pulse_ak : Callable
        Function that yields the tensors, at each bin k, for the MPS factorization.
        These can be described by the transition structure and weights of an automata.
        Expected implicitly that the first two arguments of this function will be the tensor number (k) followed by the dimension of the tensorspace

    ak_args : list[tuple]
        List of arguments associated to the callable pulse_ak() function corresponding to each channel/tensorspace in the waveguide.

    bond0 : int, default: 1
        Default bond dimension of bins.
    
    Returns
    -------
    fock_pulse : list[ndarray]
        A list of the incident time bins of the Fock pulse, with the first bin in index 0.
    
    """ 
    delta_t = params.delta_t
    d_t_total = params.d_t_total
    bond_max = params.bond_max
    
    m = int(round(pulse_time/delta_t,0))
    time_bin_dim = np.prod(d_t_total)
    channel_num = len(d_t_total)
    alphaOmega_args = np.array(alphaOmega_args)

    alphas = []; omegas = []
    for i in range(channel_num):
        alpha, omega = pulse_alphaOmega(*alphaOmega_args[i])
        alphas.append(alpha)
        omegas.append(omega)
    
    # Create inner function to calculate aks with appropriate outer product
    def calc_ak(k):
        aks = []
        for i in range(channel_num):
            aks.append(pulse_ak(k, d_t_total[i], *ak_args[i]))

        # If 0 or m-1, contract
        if k == 0:
            for i in range(channel_num):
                aks[i] = np.einsum('iq,qjk->ijk', alphas[i], aks[i]) #_contract_alpha(alphas[i], aks[i])
        if k == m-1:
            for i in range(channel_num):
                aks[i] = np.einsum('ijk,kq->ijq', aks[i], omegas[i]) #_contract_omega(omegas[i], aks[i])

        # Get the outer product for the channels
        ak = _tensor_outer_rank3(aks)
        return ak

    bins = [calc_ak(k) for k in range(m)]    
    
    # Test print of the bins
    #_matrix_text_print(bins[0], bins[-1], bins[3], time_bin_dim)

    bins_l_normed = left_normalize_bins(bins, bond_max)        
    return bins_l_normed


#-------------------------
# Fock pulse MPS generator
#-------------------------
def _fock_alphaOmega(dim:int,photon_num:int, bond0:int=1) -> tuple[np.ndarray,np.ndarray]:
    """
    Generates the alpha and omega vectors used to calculate the first/last tensors in the MPS factorization of a Fock state.
    
    Parameters
    ----------
    dim : int
        The dimension of the physical index of the MPS.
    
    photon_num : int
        Number of photons in the Fock State.

    Returns
    -------
    a1 : np.ndarray
        The alpha vector for the Fock State.
    am : np.ndarray
        The omega vector for the Fock State.

        
    Examples
    -------- 

    """ 
    a1 = np.zeros([bond0,dim], dtype=complex)
    am = np.zeros([dim,bond0], dtype=complex)
    a1[:,0] = 1
    am[photon_num, 0] = 1
    return a1, am

def _fock_pulse_ak(k:int, dt:int, photon_num:int, pulse_env:list[complex]) -> np.ndarray:
    """
    Generates the tensors for the MPS factorization of a Fock State for a given bin/site, k.
    
    Parameters
    ----------
    k : int
        The index of the bin of of the pulse being generated.

    dt : int
        The dimension of the physical index.
    
    photon_num : int
        The number of photons in the Fock state.

    pulse_env : list[complex]
        The pulse envelope of the pulse.

    Returns
    -------
    ak : np.ndarray
        The rank 3 tensor representing the k^th tensor of the MPS factorization of the Fock state.
            
    Examples
    -------- 

    """ 
    ak=np.zeros([dt,dt,dt],dtype=complex)
    photon_dim = photon_num+1
    indices = np.arange(0,photon_dim,1)
    # Vectorize this...
    for i in range(photon_dim):
        ak[indices[:photon_dim-i], i, indices[i:]] = pulse_env[k]**i / np.sqrt(sci.special.factorial(i))    
    return ak

# Test case with new matrix/normalization scheme
def fock_pulse(pulse_envs:list[list[complex]],pulse_time:float,params:InputParams, photon_nums:list[int], bond0:int=1)->list[np.ndarray]:    
    """
    Creates a Fock pulse input field MPS with a pulse envelope. 
    Can create a state of fock pulses in multiple channels, e.g. of the form |N,M,L>

    Parameters
    ----------
    pulse_envs : list[list[complex]]
        List of time dependent pulse envelopes for each channel/tensorspace in the waveguide (d_t_total).
        If None uses a top-hat pulse (constant amplitude).

    pulse_time : float
        Time length of the pulse (units of inverse coupling). 
        If the pulse envelope is of greater length it will be truncated from the tail.

    params : InputParams
        Class containing the input parameters
    
    photon_nums : list[int]
        List of photon numbers in the Fock pulse for each channel/tensorspace in the waveguide.
            
    bond0 : int, default: 1
        Default bond dimension of bins.
    
    Returns
    -------
    fock_pulse : list[ndarray]
        A list of the incident time bins of the Fock pulse, with the first bin in index 0.
    
    """ 
    # Checks that photon_nums and pulseEnvs are wrapped as lists, even in case of single channel
    if np.isscalar(photon_nums):
        photon_nums = [photon_nums]
    if np.isscalar(pulse_envs[0]):
        pulse_envs = [pulse_envs]

    # Normalize the pulse envelopes and pad as necessary with 0's
    m = int(round(pulse_time/params.delta_t,0))
    channel_num = len(params.d_t_total)
    pulse_envs = _single_pulse_env_preparation(pulse_envs, channel_num, m)

    # pack function arguments
    alphaOmega_args = []
    ak_args = []
    for i in range(channel_num):
        alphaOmega_args.append((params.d_t_total[i], photon_nums[i]))
        ak_args.append((photon_nums[i], pulse_envs[i]))

    return create_pulse(pulse_envs, pulse_time, params, _fock_alphaOmega, alphaOmega_args, _fock_pulse_ak, ak_args)

#-------------------------
# Coherent pulse MPS generator
#-------------------------
def calc_coherent_val(alpha:complex, pulse_env_val:complex, n:np.ndarray):
    """
    Calculates the coefficient for the projection of a coherent state, alpha, onto the number state, n.
    
    Parameters
    ----------
    alpha : complex
        Defines the displacement of the coherent state from the vacuum.
    
    pulse_env_val : complex
        The time dependent modulation of alpha at this point.

    n : np.ndarray
        The number state being projected onto.

    Returns
    -------
    coeff : complex
        Coefficient for the projection of the coherent state, onto the number state, n
        
    Examples
    -------- 

    """ 
    return np.exp(-np.abs(np.conj(pulse_env_val)*alpha)**2 / 2) * (np.conj(pulse_env_val)*alpha)**(n)/(np.sqrt(sci.special.factorial(n)))

def _tensor_prod_alphaOmega(_, __) -> tuple[np.ndarray,np.ndarray]:
    """
    Generates the alpha and omega vectors used to calculate the first/last tensors in the MPS factorization of a tensor product state.
    This is used when the bond dimension will be 1 for all bins, such as coherent states and single-time point squeezed states).
    
    Returns
    -------
    a1 : np.ndarray
        The alpha vector for the Fock State.
    am : np.ndarray
        The omega vector for the Fock State.

        
    Examples
    -------- 

    """ 
    a1 = np.ones((1,1), dtype=complex)
    return a1, a1

def _coherent_ak(k:int, dim:int, alpha:complex, pulse_env:list[complex]) -> np.ndarray:
    """
    Generates the tensors for the MPS factorization of a Coherent State for a given bin/site, k.
    Note that a coherent state is a total outer product state, and as such has bond dimensions of 1.
    
    Parameters
    ----------
    k : int
        The index of the bin of of the pulse being generated.

    dt : int
        The dimension of the physical index.
    
    alpha : complex
        The displacement of the coherent state from the vacuum.

    pulse_env : list[complex]
        The pulse envelope of the pulse.

    Returns
    -------
    ak : np.ndarray
        The rank 3 tensor representing the k^th tensor of the MPS factorization of the Coherent state.
            
    Examples
    -------- 

    """ 
    indices = np.arange(0,dim,1)
    coherent_vals = calc_coherent_val(alpha, pulse_env[k], indices)
    ak = np.zeros((1,dim,1),dtype=complex)
    ak[0,indices,0] = coherent_vals
    return ak


def coherent_pulse(pulse_envs:list[list[complex]],pulse_time:float, params:InputParams, alphas:list[complex],bond0:int=1) -> list[np.ndarray]:
    """
    Creates a coherent pulse input field MPS with a pulse envelope. 
    Can create a state of fock pulses in multiple channels, e.g. of the form |alpha1,alpha2,alpha3>

    Parameters
    ----------
    pulse_envs : list[list[complex]]
        List of time dependent pulse envelopes for each channel/tensorspace in the waveguide (d_t_total).
        If None uses a top-hat pulse (constant amplitude).

    pulse_time : float
        Time length of the pulse (units of inverse coupling). 
        If the pulse envelope is of greater length it will be truncated from the tail.

    params : InputParams
        Class containing the input parameters
    
    alphas : list[complex]
        List of alphas for the coherent pulses for each channel/tensorspace in the waveguide.
            
    bond0 : int, default: 1
        Default bond dimension of bins.
    
    Returns
    -------
    coherent_pulse : list[ndarray]
        A list of the incident time bins of the coherent pulse, with the first bin in index 0.
    
    """ 
    # Checks that photon_nums and pulseEnvs are wrapped as lists, even in case of single channel
    if np.isscalar(alphas):
        alphas = [alphas]
    if np.isscalar(pulse_envs[0]):
        pulse_envs = [pulse_envs]

    # Normalize the pulse envelopes and pad as necessary with 0's
    m = int(round(pulse_time/params.delta_t,0))
    channel_num = len(params.d_t_total)
    pulse_envs = _single_pulse_env_preparation(pulse_envs, channel_num, m)

    # pack function arguments
    alphaOmega_args = []
    ak_args = []
    for i in range(channel_num):
        alphaOmega_args.append(())
        ak_args.append((alphas[i], pulse_envs[i]))

    return create_pulse(pulse_envs, pulse_time, params, _fock_alphaOmega, alphaOmega_args, _fock_pulse_ak, ak_args)

#-------------------------
# 
#-------------------------
def calc_smsv_coeffs(zeta:complex, pulse_env_val:complex, n:np.ndarray):
    """
    Calculates the coefficient for the projection of a coherent state, alpha, onto the number state, n.
    
    Parameters
    ----------
    alpha : complex
        Defines the displacement of the coherent state from the vacuum.
    
    pulse_env_val : complex
        The time dependent modulation of alpha at this point.

    n : np.ndarray
        The number state being projected onto.

    Returns
    -------
    coeff : complex
        Coefficient for the projection of the coherent state, onto the number state, n
        
    Examples
    -------- 

    """ 
    n = np.asarray(n)
    zeta_local = zeta*pulse_env_val
    r = np.abs(zeta_local)
    eiphi = np.exp(1j * np.angle(zeta_local))
    halfn = n/2
    result = np.where(n % 2 == 0,
        1/(np.sqrt(np.cosh(r))) * (-eiphi * np.tanh(r))**(halfn) * (np.sqrt(sci.special.factorial(n)))/(2**halfn * sci.special.factorial(halfn)),
        0)
    return result

def calc_tmsv_coeffs(zeta:complex, pulse_env_val:complex, n:np.ndarray):
    """
    Calculates the coefficient for the projection of a coherent state, alpha, onto the number state, n.
    
    Parameters
    ----------
    alpha : complex
        Defines the displacement of the coherent state from the vacuum.
    
    pulse_env_val : complex
        The time dependent modulation of alpha at this point.

    n : np.ndarray
        The number state being projected onto.

    Returns
    -------
    coeff : complex
        Coefficient for the projection of the coherent state, onto the number state, n
        
    Examples
    -------- 

    """ 
    n = np.asarray(n)
    zeta_local = zeta*pulse_env_val
    r = np.abs(zeta_local)
    eiphi = np.exp(1j * np.angle(zeta_local))
    result = np.where(n % 2 == 0,
        1/np.cosh(r) * (-eiphi * np.tanh(r))**(n),
        0)
    return result


#-------------------------
# Inefficient operations on MPS's (will be large, and ideally should be simplified  afterwards)
#-------------------------
#TODO Doesn't seem to quite work (issue in final bin of pulse train, looks like maybe an OC issue?). Furthermore, goes crazy if trying to left normalize the MPS after adding two MPSs
def addMPSs(bins1:list[np.ndarray], bins2:list[np.ndarray], norm_coeff:float=1/np.sqrt(2)) -> list[np.ndarray]:
    new_bins = [None] * len(bins1)
    
    for i in range(1,len(bins1)-1):
        a_k = bins1[i]
        b_k = bins2[i]
        a_shape = a_k.shape
        b_shape = b_k.shape
        # output array
        c_k = np.zeros((a_shape[0]+b_shape[0], a_shape[1], a_shape[2]+b_shape[2]), dtype=complex)
        
        # top-left block: A_k
        c_k[:b_shape[0], :, :b_shape[2]] = a_k
        # bottom-right block: B_k
        c_k[a_shape[0]:, :, a_shape[2]:] = b_k
        new_bins[i] = c_k

    new_bins[0] = np.concatenate([bins1[0], bins2[0]], axis=2)
    new_bins[-1] = np.concatenate([bins1[-1],bins2[-1]], axis=0)

    new_bins[-1] *= norm_coeff
    return new_bins


# Below state is not implemented correctly yet. DO NOT USE.
#TODO Needs benchmarking, currently seems to only work with |N0> + |0N> states
# Requires with computational normalization, matrices are not defined currently such that state is normalized by default
def _noom_state(pulse_envs:list[list[float]],pulse_time:float,params:InputParams, photon_nums:list[int], bond0:int=1)->list[np.ndarray]:    
    """
    Creates a Fock pulse input field MPS with a pulse envelope.


    Parameters
    ----------
    pulse_env_r : list[float]
        Time dependent pulse envelope for a right incident pulse.
        If None, uses tophat pulse.

    pulse_time : float
        Time length of the pulse (units of inverse coupling). 
        If the pulse envelope is of greater length it will be truncated from the tail.

    params : InputParams
        Class containing the input parameters
    
    pulse_env_l : list[float]
        Time dependent pulse envelope for a left incident pulse.
        If None, uses tophat pulse.
    
    photon_num_l : int
        Left incident photon number. 
        (Interpretation may be different if photon_num_r is nonzero)

    photon_num_r : int
        Right incident photon number. 
        (Interpretation may be different if photon_num_l is nonzero)
    
    bond0 : int, default: 1
        Default bond dimension of bins.
    
    Returns
    -------
    fock_pulse : list[ndarray]
        A list of the incident time bins of the Fock pulse, with the first bin in index 0.
    
    """ 
    
    delta_t = params.delta_t
    d_t_total = params.d_t_total
    dt_max = max(d_t_total)
    bond = params.bond_max
    
    m = int(round(pulse_time/delta_t,0))
    time_bin_dim = np.prod(d_t_total)
    channel_num = len(d_t_total)
    
    photon_nums = np.array(photon_nums)
    
    # Normalize the pulse envelopes and pad as necessary with 0's
    pulse_envs = _single_pulse_env_preparation(pulse_envs, channel_num, m)


    ap1s = []
    apms = []
    #for i in range(channel_num):
    #    ap1s.append(_fock_pulse_ap1(d_t_total[i], dt_max, photon_nums[i], pulse_envs[i]))
    #    apms.append(_fock_pulse_apm(d_t_total[i], dt_max, photon_nums[i], pulse_envs[i], m))

    # Extend to outer space
    cum_dim = np.append(1, np.cumprod(d_t_total))
    cum_dim_rev = np.append(np.cumprod(d_t_total[1:][::-1])[::-1], 1)
    for i in range(channel_num):
        e0L = np.zeros(cum_dim[i]); e0R = np.zeros(cum_dim_rev[i])
        e0L[0] = 1; e0R[0] = 1
        ap1s[i] = np.einsum('ijk,a,b->iajbk', ap1s[i], e0L, e0R).reshape(bond0, time_bin_dim, dt_max)
        apms[i] = np.einsum('ijk,a,b->iajbk', apms[i], e0L, e0R).reshape(dt_max, time_bin_dim, bond0)

    ap1 = np.zeros([bond0,time_bin_dim,dt_max],dtype=complex)
    apm = np.zeros([dt_max,time_bin_dim,bond0],dtype=complex)
    for i in range(channel_num):
        ap1 += ap1s[i]
        apm += apms[i]

    # Accounting for addition of 1's in vacuum... way to avoid this with better ap1/apm/apk definitions?
    ap1[:,0,:] /= channel_num
    apm[:,0,:] /= channel_num
    # Normalization for addition of states
    apm /= np.sqrt(channel_num**max(photon_nums))

    # Create inner function to calculate aks with appropriate outer product
    def calc_ak(k):
        aks = []
        for i in range(channel_num):
            aks.append(_fock_pulse_ak(d_t_total[i], dt_max, photon_nums[i], pulse_envs[i], k))


        for i in range(channel_num):
            e0L = np.zeros(cum_dim[i]); e0R = np.zeros(cum_dim_rev[i])
            e0L[0] = 1; e0R[0] = 1
            #aks[i] = np.einsum('ijk,a,b->iajbk', ap1s[i], e0L, e0R).reshape(dt_max, time_bin_dim, dt_max)
            # With linear embedding matrix
            W = np.kron(np.kron(e0L[:,None], np.eye(d_t_total[i])), e0R[:,None])
            aks[i] = np.einsum('ijk,Jj->iJk', aks[i], W)

        ak = np.zeros([dt_max,time_bin_dim,dt_max],dtype=complex)
        for i in range(channel_num):
            ak += aks[i]

        ak[:,0,:] /= channel_num

        return ak

    # Test prints
    _matrix_text_print(ap1, apm, calc_ak, time_bin_dim)
    apk_can=[]    
    
    # Entanglement/normalization process
    apk_c=ncon([calc_ak(m-1), apm],[[-1,-2,1],[1,-3,-4]])            
    
    for k in range(m-2,1,-1):
        apk_c, stemp, i_n_r = sim._svd_tensors(apk_c, bond, time_bin_dim, time_bin_dim)
        apk_c = stemp[None,None,:] * apk_c
        apk_c = ncon([calc_ak(k),apk_c],[[-1,-2,1],[1,-3,-4]]) # k-1
        apk_can.append(i_n_r)        
    
    apk_c, stemp, i_n_r = sim._svd_tensors(apk_c, bond, time_bin_dim, time_bin_dim)
    apk_can.append(i_n_r)
    apk_c = apk_c * stemp[None,None,:]
    apk_c = ncon([ap1,apk_c],[[-1,-2,1],[1,-3,-4]])
    i_n_l, stemp, i_n_r = sim._svd_tensors(apk_c, bond, time_bin_dim, time_bin_dim)
    i_n_l = i_n_l * stemp[None,None,:]
    apk_can.append(i_n_r)
    apk_can.append(i_n_l)
    
    apk_can.reverse()
    return apk_can
