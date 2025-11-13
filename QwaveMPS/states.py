#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains initial states for the waveguide and the TLSs

"""

import numpy as np
from collections.abc import Iterator

def i_ng(d_t:int, bond0:int=1) -> np.ndarray:
    """
    Waveguide vacuum state.

    Parameters
    ----------
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
    i= np.zeros([bond0,d_t,bond0],dtype=complex) 
    i[:,0,:]=1.
    return i


def i_sg(d_sys1:int=2, bond0:int=1) -> np.ndarray:
    """
    TLS ground state.

    Parameters
    ----------
    d_sys1 : int, default: 2
        Size of the Hilbert space of the matter system.

    bond0 : int, default: 1
        Size of the bond dimension.
    
    Returns
    -------
    state : ndarray
        ndarray ground state atom.
    
    Examples
    -------- 
    """ 
    i_s = np.zeros([bond0,d_sys1,bond0],dtype=complex) 
    i_s[:,0,:]=1.
    return i_s
    
def i_se(d_sys1:int=2, bond0:int=1) -> np.ndarray:
    """
    TLS excited state.

    Parameters
    ----------
    d_sys1 : int, default: 2
        Size of the Hilbert space of the matter system.

    bond0 : int, default: 1
        Size of the bond dimension.
    
    Returns
    -------
    state : ndarray
        ndarray excited state atom.
    
    Examples
    -------- 
    """ 
    i_s = np.zeros([bond0,d_sys1,bond0],dtype=complex) 
    i_s[:,1,:]=1.
    return i_s

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
    
    Examples
    -------- 
    """ 
    d_t = np.prod(d_t_total)
    if input_bins == None:
        input_bins = []
        
    for i in range(len(input_bins)):
        yield input_bins[i]
    
    # After all specified input bins are yielded, start inputting vacuum bins
    # Is there any reason we may want the default bin to be something else (an input parameter)?
    if default_state is None:
        while True:
            yield i_ng(d_t, bond0)
    else:
        while True:
            yield default_state


def coupling(coupl:str='symmetrical', gamma:float=1,gamma_r=None,gamma_l=None) -> tuple[float,float]:
    """ Coupling can be chiral or symmetrical.
    Symmetrical by default."""   
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