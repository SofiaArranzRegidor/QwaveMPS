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
from . import simulation as sim
from QwaveMPS.src.parameters import InputParams


#--------------------
#Initial basic states
#--------------------

def i_ng(d_t:int, bond0:int=1) -> np.ndarray:
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
        ndarray vacuum state.
    """ 
    i= np.zeros([bond0,d_t,bond0],dtype=complex) 
    i[:,0,:]=1.
    return i

def i_sg(d_sys:int=2, bond0:int=1) -> np.ndarray:
    """
    TLS ground state tensor.

    Parameters
    ----------
    d_sys : int, default: 2
        Size of the Hilbert space of the matter system.

    bond0 : int, default: 1
        Initial size of the bond dimension.
    
    Returns
    -------
    state : ndarray
        ndarray ground state atom.
    """ 
    i_s = np.zeros([bond0,d_sys,bond0],dtype=complex) 
    i_s[:,0,:]=1.
    return i_s
    
def i_se(d_sys:int=2, bond0:int=1) -> np.ndarray:
    """
    TLS excited state tensor.

    Parameters
    ----------
    d_sys : int, default: 2
        Size of the Hilbert space of the matter system.

    bond0 : int, default: 1
        Initial size of the bond dimension.
    
    Returns
    -------
    state : ndarray
        ndarray excited state atom.
    """ 
    i_s = np.zeros([bond0,d_sys,bond0],dtype=complex) 
    i_s[:,1,:]=1.
    return i_s

def vacuum(time_length:float, params:InputParams) -> np.ndarray:
    """
    Produces an array of vacuum time bins for a given time_length.

    Parameters
    ----------

    time_length : float
        Length of the vacuum pulse.

    params:InputParams
        Class containing the input parameters

    Returns
    -------
    state : ndarray
        ndarray vacuum state.
    """ 
    delta_t = params.delta_t
    d_t_total = params.d_t_total
    
    bond0=1
    l = int(round(time_length/delta_t, 0))
    d_t=np.prod(d_t_total)
    
    
    return [i_ng(d_t, bond0) for i in range(l)]

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
        Default time bin state yielded as an input state after all input_bins are exhusted. If None then vacuum states are yielded.
    
    Returns
    -------
    gen : Generator
        A generator for the input field time bins.
    """ 
    d_t = np.prod(d_t_total)
    if input_bins == None:
        input_bins = []
        
    for i in range(len(input_bins)):
        yield input_bins[i]
    
    # After all specified input bins are yielded, start inputting vacuum bins
    if default_state is None:
        while True:
            yield i_ng(d_t, bond0)
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
    coupl : str, default: 'symmetrical'
       Coupling options: 'symmetrical', 'chiral_r', 'chiral_l', 'other' 
    
    gamma : float, default:1
        Total coupling. Code in units of coupling, hence, the default is 1.
    
    gamma_r : None or float, default : None
        Left coupling. If coupl = 'other' define explicitly.
        
    gamma_l : None or float, default : None
        Right coupling. If coupl = 'other' define explicitly.
    Returns
    -------
    gamma_l,gamma_r : tuple[float,float]
        A tuple with the values of the left and right coupling
    """   
    if coupl == 'chiral_r': 
        gamma_r=gamma
        gamma_l=gamma - gamma_r
    if coupl == 'chiral_l': 
        gamma_l=gamma
        gamma_r=gamma - gamma_l
    if coupl == 'symmetrical':
        gamma_r=gamma/2.
        gamma_l=gamma - gamma_r
    if coupl == 'other':
        gamma_r=gamma_r
        gamma_l=gamma_l
    return gamma_l,gamma_r       

#----------------------
#Pulse envelope helpers
#----------------------

def tophat_envelope(pulse_time:float, params:InputParams)->np.ndarray:
    """
    Create a top hat pulse envelope given by the time length of the pulse.

    Parameters
    ----------
    pulse_time : float
        Duration time of the pulse.

    params:InputParams
        Class containing the input parameters
        
    Returns
    -------
    pulse_envelope : list[float]
        List of amplitude values of the pulse envelope.
    """ 
    delta_t=params.delta_t
    m = int(round(pulse_time/delta_t))
    return np.ones(m)

def gaussian_envelope(pulse_time:float, params:InputParams, gaussian_width:float, gaussian_center:float, initial_time:int=0)->np.ndarray:
    """
    Create a gaussian pulse envelope given by the time length of the pulse 
    and the mean and variance parameters.

    Parameters
    ----------
    pulse_time : float
        Duration time of the pulse.

    params:InputParams
        Class containing the input parameters
        
    gaussian_width : float
        Variance of the gaussian.
    
    gaussian_center : float
        Mean of the gaussian.

    initial_time : float, default: 0
        Initial time of the returned pulse envelope function.
    
    Returns
    -------
    pulse_envelope : list[float]
        List of amplitude values of the pulse envelope.
    """ 
    
    delta_t=params.delta_t
    
    m = int(round(pulse_time/delta_t,0))
    times = np.arange(initial_time, initial_time + m*delta_t, delta_t)
    diffs = times - gaussian_center
    exponent = - (diffs ** 2) / (2 * gaussian_width ** 2) 

    pulse_envelope = np.exp(exponent)         
    return pulse_envelope

def exp_decay_envelope(pulse_time:float, params:InputParams, decay_rate:float, decay_center:float=0)->np.ndarray:
    """
    Create a exponential decay pulse envelope given by the time length of the pulse 
    and the decay rate and decay center parameters.

    Parameters
    ----------
    pulse_time : float
        Duration time of the pulse.

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

def normalize_pulse_envelope(delta_t:float, pulse_env:list[float])->np.ndarray:
    """
    Normalized a given pulse envelope so that the integral of the square magnitude is 1.

    Parameters
    ----------
    delta_t : float
        Time step size for the simulation.
        
    pulse_env : list[float]
        Time dependent pulse envelope that is being normalized.

    Returns
    -------
    pulse_env : list[float]
        The normalized time dependent pulse envelope.

    """ 
    norm_factor = np.sum(np.abs(np.array(pulse_env))**2) * delta_t
    pulse_env /= np.sqrt(norm_factor)
    return pulse_env

#-------------------------
# Fock pulse MPS generator
#-------------------------
def fock_pulse(pulse_env:list[float],pulse_time:float,params:InputParams, photon_num:int, direction:str='R', bond0:int=1)->list[np.ndarray]:
    """
    Creates an Fock pulse input field state with a normalized pulse envelope

    Parameters
    ----------
    pulse_time : float
        Time length of the pulse. If the pulse envelope is of greater length it will be truncated.
    
    delta_t : float
        Time step size for the simulation.

    d_t_total : list[int]
        List of sizes of the photonic Hilbert spaces.

    bond : int
        Truncation for maximum bond dimension. 

    pulse_env : list[float], default: None
        Time dependent pulse envelope for the incident pulse. Default uses a tophat pulse for the duration of the pulse_time.

    photon_num : int
        Incident photon number.

    direction : str, default: 'R'
        Flag to dictate the direction of the propagating pulse.

    bond0 : int, default: 1
        Default bond dimension of bins.

    Returns
    -------
    apk_can : list[ndarray]
        A list of the incident time bins of the Fock pulse, with the first bin in index 0.

    Examples
    -------- 

    """ 


    if direction.upper() == 'L' or direction == 1 or len(params.d_t_total)==1:
        photon_num_l = photon_num
        photon_num_r = 0

    else:
        photon_num_r = photon_num
        photon_num_l = 0

    return _fock_pulse(pulse_env, pulse_time, params, pulse_env, photon_num_l, photon_num_r, bond0)


def _fock_pulse(pulse_env_r:list[float],pulse_time:float,params:InputParams, pulse_env_l:list[float]=None, photon_num_l:int=0, photon_num_r:int=1, bond0:int=1)->list[np.ndarray]:    
    """
    Creates an Fock pulse input field MPS with a normalized pulse envelope

    Parameters
    ----------
    pulse_time : float
        Time length of the pulse. If the pulse envelope is of greater length it will be truncated.

    params:InputParams
        Class containing the input parameters
        
    pulse_env_l : list[float], default: None
        Time dependent pulse envelope for a left incident pulse. Default uses a tophat pulse for the duration of the pulse_time.
    
    photon_num_l : int, default: 0
        Left incident photon (figure out how to interpret)

    pulse_env_r : list[float], default: None
        Time dependent pulse envelope for a right incident pulse. Default uses a tophat pulse for the duration of the pulse_time.
        
    photon_num_r : int, default: 1
        Right indident photon (same as above...)
    
    bond0 : int, default: 1
        Default bond dimension of bins.
    
    Returns
    -------
    apk_can : list[ndarray]
        A list of the incident time bins of the Fock pulse, with the first bin in index 0.
    
    """ 
    
    delta_t = params.delta_t
    d_t_total = params.d_t_total
    bond = params.bond
    
    m = int(round(pulse_time/delta_t,0))
    time_bin_dim = np.prod(d_t_total)
    dt = d_t_total[0]    
    channel_num=min(len(d_t_total), 2)
    
    # Lists created to track parameters for the L and R Hilbert spaces respectively
    indices_untruncated = [np.arange(0,time_bin_dim, dt), np.arange(0,dt,1)]    #IndicesL and IndicesR
    indices_untruncated = indices_untruncated[-channel_num:] # Correct the size positions in single hilbert space case
    photon_nums = [photon_num_l, photon_num_r]
    photon_num_dims = [photon_num_l+1, photon_num_r+1]
    
    indices = []
    indices2 = []
    dt_indices = []
    for i in range(len(indices_untruncated)):
        indices.append(indices_untruncated[i][:photon_num_dims[i]]) # Truncate if necessary (fewer photon pulse than size of Hilbert space)
        indices2.append(indices[i][::-1])
        dt_indices.append(np.arange(0,dt)[:photon_num_dims[i]]) # Should be truncated or not?

    # Normalize the pulse envelopes
    pulse_envs = [pulse_env_l, pulse_env_r]
    for i in range(channel_num):
        # Default to single top hat pulse
        if pulse_envs[i] is None:
            pulse_envs[i] = np.ones(m)
        else:
            pulse_envs[i] = np.array(pulse_envs[i])
        pulse_envs[i] = normalize_pulse_envelope(delta_t, pulse_envs[i])
    
    # Pad envelopes as necessary to be of length m
    pulse_envs[0] = np.append(pulse_envs[0], [0] * (m-len(pulse_envs[0])))
    pulse_envs[1] = np.append(pulse_envs[1], [0] * (m-len(pulse_envs[1])))

    pulse_envs = list(zip(pulse_envs[0], pulse_envs[1]))

    ap1=np.zeros([bond0,time_bin_dim,dt],dtype=complex)
    apm=np.zeros([dt,time_bin_dim,bond0],dtype=complex)

    # Evaluate the first and last matrices (each iteration for L and R respectively)
    for i in range(channel_num):
        ap1[:,indices[i],dt_indices[i]] = np.sqrt(photon_nums[i]) * pulse_envs[0][i]**np.arange(photon_num_dims[i])
        ap1[:,indices[i][0],dt_indices[i][0]] = 1

        combinatorialFactors = sci.special.comb(photon_nums[i],np.arange(photon_num_dims[i]))
        apmVals = np.sqrt(combinatorialFactors)* pulse_envs[-1][i]**np.arange(photon_num_dims[i])
        apm[dt_indices[i][::-1], indices[i],:] = apmVals[:,None]
        apm[dt_indices[i][0], indices[i][-1],:] = np.sqrt(photon_nums[i]) * pulse_envs[-1][i]**photon_nums[i]


    # Internal function to evaluate the k^th matrix
    def calc_ak(dt, d_total, pulse_envs_k, k):
        ak=np.zeros([dt,d_total,dt],dtype=complex)
        for j in range(channel_num):
            for i in range(photon_num_dims[j]):
                ak[dt_indices[j][:photon_num_dims[j]-i], indices[j][i], dt_indices[j][i:]] = np.sqrt(sci.special.comb(dt_indices[j][i:],i)) * pulse_envs_k[j]**i
            ak[0, indices[j], dt_indices[j]] = np.sqrt(photon_nums[j]) * pulse_envs_k[j]**np.arange(photon_num_dims[j])
            ak[dt_indices[j],0,dt_indices[j]] = 1
        return ak

    apk_can=[]    
    
    # Entanglement/normalization process
    apk_c=ncon([calc_ak(dt, time_bin_dim, pulse_envs[m-2], m-1), apm],[[-1,-2,1],[1,-3,-4]])            
    
    for k in range(m-2,1,-1):
        apk_c, stemp, i_n_r = sim._svd_tensors(apk_c, bond, time_bin_dim, time_bin_dim)
        apk_c = stemp[None,None,:] * apk_c
        apk_c = ncon([calc_ak(dt, time_bin_dim, pulse_envs[k-1], k),apk_c],[[-1,-2,1],[1,-3,-4]]) # k-1
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




