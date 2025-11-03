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

def input_state_generator(d_t:int, input_bins:list[np.ndarray]=None, bond0:int=1) -> Iterator[np.ndarray]:
    """
    Creates an iterator (generator) for the input field states of the waveguide.

    Parameters
    ----------
    d_t1 : int
        Size of the Hilbert space of the field.

    input_bins : list[np.ndarray], default: None
        List of time bins describing the input field state.
        
    bond0 : int, default: 1
        Size of the initial bond dimension.
    
    Returns
    -------
    gen : Generator
        A generator for the input field time bins.
    
    Examples
    -------- 
    """ 
    if input_bins == None:
        input_bins = []
        
    for i in range(len(input_bins)):
        yield input_bins[i]
    
    # After all specified input bins are yielded, start inputting vacuum bins
    # Is there any reason we may want the default bin to be something else (an input parameter)?
    while True:
        vacState = i_ng(d_t, bond0)
        yield vacState


def coupling(coupl:str='symmetrical', gamma:float=1) -> tuple[float,float]:
    """ Coupling can be chiral or symmetrical.
    Symmetrical by default."""   
    if coupl == 'chiral': 
        gammaR=gamma
        gammaL=gamma - gammaR
    if coupl == 'symmetrical':
        gammaR=gamma/2.
        gammaL=gamma - gammaR
    return gammaL,gammaR       