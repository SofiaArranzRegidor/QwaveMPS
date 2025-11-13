#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the functions to generate quantum pulses and some simple pulse envelopes.

"""


import numpy as np
from ncon import ncon
import scipy as sci
from . import simulation as sim
from . import states as states

#%% Pulse Envelopes
def tophat_envelope(pulse_time:float, delta_t:float)->np.ndarray:
    """
    Create a pulse envelope given by the sum of gaussians with given means and variances.

    Parameters
    ----------
    pulse_time : float
        Duration time of the pulse.

    delta_t : float
        Time evolution step of the simulation.
        
    Returns
    -------
    pulse_envelope : list[float]
        List of amplitude values of the pulse envelopes at indexed times (separated by delta_t).
    
    Examples
    -------- 
    """ 

    m = int(round(pulse_time/delta_t),0)
    return np.ones(m)

def gaussian_envelope(pulse_time:float, delta_t:float, gaussian_width:float, gaussian_center:float, initial_time:int=0)->np.ndarray:
    """
    Create a pulse envelope given by the sum of gaussians with given means and variances.

    Parameters
    ----------
    pulse_time : float
        Duration time of the pulse.

    delta_t : float
        Time evolution step of the simulation.
        
    gaussian_width : float
        Variance of the gaussian.
    
    gaussian_center : float
        Mean of the gaussian.

    initial_time : float, default: 0
        Initial time of the returned pulse envelope function.
    
    Returns
    -------
    pulse_envelope : list[float]
        List of amplitude values of the pulse envelopes at indexed times (separated by delta_t).
    
    Examples
    -------- 
    """ 

    m = int(round(pulse_time/delta_t,0))
    times = np.arange(initial_time, initial_time + m*delta_t, delta_t)
    diffs = times - gaussian_center
    exponent = - (diffs ** 2) / (2 * gaussian_width ** 2) 

    pulse_envelope = np.exp(exponent)         
    return pulse_envelope


def multiple_gaussian_envelope(pulse_time:float, delta_t:float, gaussian_widths:list[float], gaussian_centers:list[float], initial_time:int=0)->np.ndarray:
    """
    Create a pulse envelope given by the sum of gaussians with given means and variances.

    Parameters
    ----------
    pulse_time : float
        Duration time of the pulse.

    delta_t : float
        Time evolution step of the simulation.
        
    gaussian_widths : list[float]
        List of variances of gaussians.
    
    gaussian_centers : list[float]
        List of means of gaussians.

    initial_time : float, default: 0
        Initial time of the returned pulse envelope function.
    
    Returns
    -------
    pulse_envelope : list[float]
        List of amplitude values of the pulse envelopes at indexed times (separated by delta_t).
    
    Examples
    -------- 
    """ 

    m = int(round(pulse_time/delta_t,0))
    indices = np.arange(initial_time, initial_time + m*delta_t, delta_t)
    
    def gaussian_values(times, gaussian_widths, gaussian_centers):
        times = np.array(times)[:,None]
        gaussian_widths_np = np.array(gaussian_widths)[None,:]
        gaussian_centers_np = np.array(gaussian_centers)[None,:]
        
        diffs = times - gaussian_centers_np
        exponent = - (diffs ** 2) / (2 * gaussian_widths_np ** 2) 
        return (np.exp(exponent)).sum(axis=1)
    
    pulse_envelope = gaussian_values(indices, gaussian_widths, gaussian_centers)         
    return pulse_envelope

def exp_decay_envelope(pulse_time:float, delta_t:float, decay_rate:float, decay_center:float=0)->np.ndarray:
    """
    Create a pulse envelope given by the sum of gaussians with given means and variances.

    Parameters
    ----------
    pulse_time : float
        Duration time of the pulse.

    delta_t : float
        Time evolution step of the simulation.

    decay_rate : float
        Decay rate of the exponential.
    
    decay_center : float
        Time center/offset of the exponential decay function.
        
    Returns
    -------
    pulse_envelope : list[float]
        List of amplitude values of the pulse envelopes at indexed times (separated by delta_t).
    
    Examples
    -------- 
    """ 
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
    
    Examples
    -------- 
    """ 
    norm_factor = np.sum(np.abs(np.array(pulse_env))**2) * delta_t
    pulse_env /= np.sqrt(norm_factor)
    return pulse_env

#%% Pulses

# Could probably be generalized to N photonic channels by just generalizing the indices list
# Also note that not sure it makes sense if dont's have one of photon_num_l or photon_num_r = 0, or both equal each other (in which case seem to have (|N>|0> + |0>|N>) state) 
# Currently designed to only work for dt_l = dt_r
# May try to generalize it later so can use same function for one time bin space...
# Will have to do a few tests to be certain things are right, this is extension of function I usually use that does same thing but ONLY for L OR R channel
# Can always just make a wrapper function for this that only allows for one or the other if we want.
def fock_pulse(pulse_time:float, delta_t:float, d_t_total:int, bond:int, pulse_env_l:list[float]=None, photon_num_l:int=0, pulse_env_r:list[float]=None, photon_num_r:int=1, bond0:int=1)->list[np.ndarray]:
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
    
    Examples
    -------- 
    """ 
    m = int(round(pulse_time/delta_t,0))
    time_bin_dim = np.prod(d_t_total)
    dt = d_t_total[0]
    channel_num=2
    
    # Lists created to track parameters for the L and R Hilbert spaces respectively
    indices = [np.arange(0,time_bin_dim, dt), np.arange(0,dt,1)]    #IndicesL and IndicesR
    photon_nums = [photon_num_l, photon_num_r]
    photon_num_dims = [photon_num_l+1, photon_num_r+1]
    
    indices = [indices[0][:photon_num_dims[0]], indices[1][:photon_num_dims[1]]] # Truncate if necessary (fewer photon pulse than size of Hilbert space)
    indices2 = [indices[0][::-1], indices[1][::-1]]
    dt_indices = [np.arange(0,dt)[:photon_num_dims[0]], np.arange(0,dt)[:photon_num_dims[1]]] # Should be truncated or not?

    # Normalize the pulse envelopes
    pulse_envs = [pulse_env_l, pulse_env_r]
    for i in range(channel_num):
        # Default to single top hat pulse
        if pulse_envs[i] is None:
            pulse_envs[i] = np.ones(m)
        else:
            pulse_envs[i] = np.array(pulse_envs[i])
        pulse_envs[i] = normalize_pulse_envelope(delta_t, pulse_envs[i])
    
    #m = max(len(pulse_envs[0]), len(pulse_envs[1]))  # To make the pulse duration dependent on given envelope

    # Pad envelopes as necessary to be of length m
    pulse_envs[0] = np.append(pulse_envs[0], [0] * (m-len(pulse_envs[0])))
    pulse_envs[1] = np.append(pulse_envs[1], [0] * (m-len(pulse_envs[1])))

    pulse_envs = list(zip(pulse_envs[0], pulse_envs[1]))
    print(pulse_envs)
    
    ap1=np.zeros([bond0,time_bin_dim,dt],dtype=complex)
    apm=np.zeros([dt,time_bin_dim,bond0],dtype=complex)

    # Evaluate the first and last matrices (each iteration for L and R respectively)
    for i in range(channel_num):
        ap1[:,indices[i],dt_indices[i]] = np.sqrt(photon_nums[i]) * pulse_envs[0][i]**np.arange(photon_num_dims[i])
        ap1[:,indices[i][0],dt_indices[i][0]] = 1

        combinatorialFactors = sci.special.comb(photon_nums[i],np.arange(photon_num_dims[i]))
        apmVals = np.sqrt(combinatorialFactors)* pulse_envs[-1][i]**np.arange(photon_num_dims[i])
        apm[dt_indices[i][::-1], indices[i],:] = apmVals[:,None]
        #ApM[dTimeIndices[-1], indices[0],:] = 1
        apm[dt_indices[i][0], indices[i][-1],:] = np.sqrt(photon_nums[i]) * pulse_envs[-1][i]**photon_nums[i]


    # Internal function to evaluate the k^th matrix
    def calc_ak(dt, d_total, pulse_envs_k, k):
        ak=np.zeros([dt,d_total,dt],dtype=complex)
        for j in range(channel_num):
            for i in range(photon_num_dims[j]):
                #Ak[i, indices[: photonNumDims-i], dTimeIndices[i:]] = fkVal
                ak[dt_indices[j][:photon_num_dims[j]-i], indices[j][i], dt_indices[j][i:]] = np.sqrt(sci.special.comb(dt_indices[j][i:],i)) * pulse_envs_k[j]**i
            ak[0, indices[j], dt_indices[j]] = np.sqrt(photon_nums[j]) * pulse_envs_k[j]**np.arange(photon_num_dims[j])
            ak[dt_indices[j],0,dt_indices[j]] = 1
        return ak

    '''
    print('A1:')
    for i in range(timeBinDim):
        print(np.real(Ap1[:,i,:]))
    print('='*50)
    print('ApM:')
    print('M=', m)
    for i in range(timeBinDim):
        print(np.real(ApM[:,i,:]))
    print('='*50)
    print('Ak:')
    for k in range(2,3):
        print('='*50)
        print('k=', k)
        Ak = calcAk(dTime, timeBinDim, fk[k-1], k)
        for i in range(timeBinDim):
            print(np.real(Ak[:,i,:]))
    exit(-1)
    '''

    apk_can=[]    
    
    # Entanglement/normalization process
    apk_c=ncon([calc_ak(dt, time_bin_dim, pulse_envs[m-2], m-1), apm],[[-1,-2,1],[1,-3,-4]])            
    
    for k in range(m-1,1,-1):
        apk_c, stemp, i_n_r = sim._svd_tensors(apk_c, apk_c.shape[0]*time_bin_dim, apk_c.shape[-1]*time_bin_dim, bond, time_bin_dim, time_bin_dim)
        apk_c = stemp[None,None,:] * apk_c
        apk_c = ncon([calc_ak(dt, time_bin_dim, pulse_envs[k-1], k),apk_c],[[-1,-2,1],[1,-3,-4]]) # k-1
        apk_can.append(i_n_r)        
    
    apk_c, stemp, i_n_r = sim._svd_tensors(apk_c, apk_c.shape[0]*time_bin_dim, apk_c.shape[-1]*time_bin_dim, bond, time_bin_dim, time_bin_dim)
    apk_c = stemp[None,None,:] * apk_c
    apk_c = ncon([ap1,apk_c],[[-1,-2,1],[1,-3,-4]])
    i_n_l, stemp, i_n_r = sim._svd_tensors(apk_c, apk_c.shape[0]*time_bin_dim, apk_c.shape[-1]*time_bin_dim, bond, time_bin_dim, time_bin_dim)
    i_n_l = stemp[None,None,:] * i_n_l
    apk_can.append(i_n_r)
    apk_can.append(i_n_l)
    
    apk_can.reverse()
    return apk_can


def vacuum_pulse(time_length:float, delta_t:float, d_t:int, bond0:int=1) -> np.ndarray:
    """
    Produces a pulse of vacuum time bins for an interval of length time_length.

    Parameters
    ----------
    
    time_length : float
        Length of the vacuum pulse.

    delta_t : float
        Length of the simulation time step.
        
    d_t : int
        Size of the truncated Hilbert space of the light field.

    bond0 : int, default: 1
        Size of the bond dimension.
    
    Returns
    -------
    state : ndarray
        ndarray vacuum state.
    
    Examples
    -------- 
    """ 
    l = int(round(time_length/delta_t, 0))

    return [states.i_ng(d_t, bond0) for i in range(l)]

