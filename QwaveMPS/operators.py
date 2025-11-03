#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the main quantum operators

It requires the module ncon (pip install --user ncon)

"""

import numpy as np
from scipy.linalg import expm
from ncon import ncon

#%%

# class basic_operators:
"""
This includes the basic boson and TLS operators used to build the Hamiltonian 
"""
#def sigmaplus(d_sys: int=2) -> npt.NDArray[np.complex128]:  
#def sigmaplus(d_sys: int=2) -> np.ndarray[Annotated[tuple[int,int],('n', 'n')], np.dtype[complex]]:  
def sigmaplus(d_sys:int=2) -> np.ndarray:  
    """
    Raising operator for the Pauli spins.

    Returns
    -------
    oper : ndarray
        ndarray for the Pauli spin raising operator.
    
    Examples
    -------- 
    """
    a = np.zeros((d_sys,d_sys),dtype=complex)
    a[1,0]=1.       
    return a

def sigmaminus(d_sys:int=2) -> np.ndarray:  
    """
    Lowering operator for the Pauli spins.

    Returns
    -------
    oper : ndarray
        ndarray for the Pauli spin lowering operator.
    
    Examples
    -------- 
    """
    a = np.zeros((d_sys,d_sys),dtype=complex)
    a[0,1]=1.       
    return a

def DeltaBdag(Deltat:float, d_t:int=2) -> np.ndarray:  
    """
    Time bin noise creation (raising) operator.

    Parameters
    ----------
    Deltat : float
        Time step for system evolution.

    d_t : int, default: 2
        Size of the truncated field Hilbert space

    Returns
    -------
    oper : ndarray
        ndarray time bin noise creation operator.
    
    Examples
    -------- 
    """
    return np.sqrt(Deltat) * np.diag(np.sqrt(np.arange(1, d_t, dtype=complex)), -1) 

def DeltaB(Deltat:float, d_t:int=2) -> np.ndarray:  
    """
    Time bin noise annihilation (lowering) operator.

    Parameters
    ----------
    Deltat : float
        Time step for system evolution.
    
    d_t : int, default: 2
        Size of the truncated field Hilbert space

    Returns
    -------
    oper : ndarray
        ndarray time bin noise creation operator.
    
    Examples
    -------- 
    """      
    return np.sqrt(Deltat) * np.diag(np.sqrt(np.arange(1, d_t, dtype=complex)), 1)    

def DeltaBdagL(Deltat:float, d_t:int=2) -> np.ndarray:  
    """
    Left time bin noise creation (raising) operator for a system with two channels of light, left and right moving.

    Parameters
    ----------
    Deltat : float
        Time step for system evolution.
    
    d_t : int, default: 2
        Size of the truncated field Hilbert space

    Returns
    -------
    oper : ndarray
        ndarray left time bin noise creation operator.
    
    Examples
    -------- 
    """ 
    return np.kron(DeltaBdag(Deltat, d_t),np.eye(d_t))     

def DeltaBdagR(Deltat:float, d_t:int=2) -> np.ndarray: 
    """
    Right time bin noise creation (raising) operator for a system with two channels of light, left and right moving.

    Parameters
    ----------
    Deltat : float
        Time step for system evolution.
    
    d_t : int, default: 2
        Size of the truncated field Hilbert space

    Returns
    -------
    oper : ndarray
        ndarray right time bin noise creation operator.
    
    Examples
    -------- 
    """ 
    return np.kron(np.eye(d_t), DeltaBdag(Deltat, d_t))     

def DeltaBL(Deltat:float, d_t:int=2) -> np.ndarray:  
    """
    Left time bin noise annihilation (lowering) operator for a system with two channels of light, left and right moving.

    Parameters
    ----------
    Deltat : float
        Time step for system evolution.
    
    d_t : int, default: 2
        Size of the truncated field Hilbert space

    Returns
    -------
    oper : ndarray
        ndarray left time bin noise annihilation operator.
    
    Examples
    -------- 
    """ 
    return np.kron(DeltaB(Deltat, d_t),np.eye(d_t))     

def DeltaBR(Deltat:float, d_t:int=2) -> np.ndarray:  
    """
    Right time bin noise annihilation (lowering) operator for a system with two channels of light, left and right moving.

    Parameters
    ----------
    Deltat : float
        Time step for system evolution.

    Returns
    -------
    oper : ndarray
        ndarray right time bin noise annihilation operator.
    
    Examples
    -------- 
    """ 
    return np.kron(np.eye(d_t),DeltaB(Deltat, d_t))       

def e(d_sys:int=2) -> np.ndarray:
    """
    |e⟩⟨e| operator for the TLS.

    Parameters
    ----------
    d_sys : float
        Size of the Hilbert space of the matter system.

    Returns
    -------
    oper : ndarray
        ndarray <What it is>.
    
    Examples
    -------- 
    """ 
    exc = np.zeros((d_sys,d_sys),dtype=complex)
    exc[1,1]=1.      
    return exc

def U(Hm:np.ndarray, d_sys:int=2, d_t:int=2, interacting_timebins_num:int=1) -> np.ndarray:
    """
    Creates a time evolution operator for a given Hamiltonian.

    Parameters
    ----------
    Hm : ndarray
        Hamiltonian of the system.

    d_sys : int, default: 2
        Size of the Hilbert space of the matter system.
    
    d_t : int, default: 2
        Size of the truncated Hilbert space of the light field.

    interacting_timebins_num : int, default: 1
        Number of light channels/feedback loops involved in the Hamiltonian.

    Returns
    -------
    oper : ndarray
        ndarray time evolution operator of shape ((d_sys,) + (d_t,)*interacting_timebins_num)*2.
    
    Examples
    -------- 
    """ 
    sol= expm(-1j*Hm)
    #shape = (((d_t,)*interacting_timebins_num) + (d_sys,) ) * 2
    shape = ((d_sys,) + ((d_t,)*interacting_timebins_num)) * 2
    return sol.reshape(shape)

def swap(dim1:int, dim2:int) -> np.ndarray:
    """
    Swap tensor to swap the contents of adjacent MPS bins.

    Parameters
    ----------
    dim1 : int
        Size of the first Hilbert space.
            
    dim2 : int, default: 2
        Size of the second Hilbert space.

    Returns
    -------
    oper : ndarray
        ndarray of shape (dim1,dim2,dim1,dim2) swap operator.
    
    Examples
    -------- 
    """ 
    size = dim1 * dim2
    swap = np.zeros([size,size],dtype=complex)
    for i in range(dim1):
        for j in range(dim2):
            swap[i + j*dim1,(i*dim2)+j]=1
    return swap.reshape(dim1,dim2,dim1,dim2)   
 
# I think slightly less performant for small dims, but slightly faster for large dims, could just remove
# Reduces to single explicit python loop
def vectorizedswap(dim1:int, dim2:int) -> np.ndarray:
    """
    Swap tensor to swap the contents of adjacent MPS bins.

    Parameters
    ----------
    dim1 : int
        Size of the first Hilbert space.
            
    dim2 : int, default: 2
        Size of the second Hilbert space.

    Returns
    -------
    oper : ndarray
        ndarray of shape (dim1,dim2,dim1,dim2) swap operator.
    
    Examples
    -------- 
    """ 
    size = dim1*dim2
    swap = np.zeros((size, size), dtype=complex)
    indices = np.array([(i%dim2)*dim1 + int(i/dim2) for i in range(size)], dtype=int)
    swap[indices, np.arange(swap.shape[0])] = 1
    return swap.reshape(dim1,dim2,dim1,dim2)


def expectation(AList:np.ndarray, MPO:np.ndarray) -> complex:
    """
    The expectation value of a MPS bin with a given operator.

    Parameters
    ----------
    AList : ndarray
        MPS bin defining the state having an expectation taken with respect to some operator.

    MPO : ndarray
        Operator whose expectation value is being taken.
    
    Returns
    -------
    expectation value : complex
        The expectation value of the operator for the given state.
    
    Examples
    -------- 
    """ 
    sol = ncon([np.conj(AList),MPO,AList],[[1,2,4],[2,3],[1,3,4]])
    return sol


# class observables:

# def __init__(self, operators=None):
#     if operators is None:
#         operators = basic_operators()  
#     self.op = operators

def TLS_pop(d_sys:int=2) -> np.ndarray:    
    return (sigmaplus() @ sigmaminus())
    
def a_R_pop(Deltat:float, d_t:int=2) -> np.ndarray:
    return (DeltaBdagR(Deltat) @ DeltaBR(Deltat))/Deltat

def a_L_pop(Deltat:float, d_t:int=2) -> np.ndarray:  
    return (DeltaBdagL(Deltat) @ DeltaBL(Deltat))/Deltat

def a_pop(Deltat:float, d_t:int=2) -> np.ndarray:  
    return (DeltaBdag(Deltat) @ DeltaB(Deltat))/Deltat