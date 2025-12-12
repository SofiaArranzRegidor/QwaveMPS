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

def delta_b_dag(delta_t:float, d_t:int=2) -> np.ndarray:  
    """
    Time bin noise creation (raising) operator.

    Parameters
    ----------
    delta_t : float
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
    return np.sqrt(delta_t) * np.diag(np.sqrt(np.arange(1, d_t, dtype=complex)), -1) 

def delta_b(delta_t:float, d_t:int=2) -> np.ndarray:  
    """
    Time bin noise annihilation (lowering) operator.

    Parameters
    ----------
    delta_t : float
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
    return np.sqrt(delta_t) * np.diag(np.sqrt(np.arange(1, d_t, dtype=complex)), 1)    

def delta_b_dag_l(delta_t:float, d_t_total:np.array) -> np.ndarray:  
    """
    Left time bin noise creation (raising) operator for a system with two channels of light, left and right moving.

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
    
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

    Returns
    -------
    oper : ndarray
        ndarray left time bin noise creation operator.
    
    Examples
    -------- 
    """ 
    d_t_l=d_t_total[0]
    d_t_r=d_t_total[1]
    return np.kron(delta_b_dag(delta_t, d_t_l),np.eye(d_t_r))     

def delta_b_dag_r(delta_t:float, d_t_total:np.array) -> np.ndarray: 
    """
    Right time bin noise creation (raising) operator for a system with two channels of light, left and right moving.

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
    
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

    Returns
    -------
    oper : ndarray
        ndarray right time bin noise creation operator.
    
    Examples
    -------- 
    """ 
    d_t_l=d_t_total[0]
    d_t_r=d_t_total[1]
    return np.kron(np.eye(d_t_l), delta_b_dag(delta_t, d_t_r))     

def delta_b_l(delta_t:float, d_t_total:np.array) -> np.ndarray:  
    """
    Left time bin noise annihilation (lowering) operator for a system with two channels of light, left and right moving.

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
    
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

    Returns
    -------
    oper : ndarray
        ndarray left time bin noise annihilation operator.
    
    Examples
    -------- 
    """ 
    d_t_l=d_t_total[0]
    d_t_r=d_t_total[1]
    return np.kron(delta_b(delta_t, d_t_l),np.eye(d_t_r))     

def delta_b_r(delta_t:float, d_t_total:np.array) -> np.ndarray:  
    """
    Right time bin noise annihilation (lowering) operator for a system with two channels of light, left and right moving.

    Parameters
    ----------
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

    Returns
    -------
    oper : ndarray
        ndarray right time bin noise annihilation operator.
    
    Examples
    -------- 
    """ 
    d_t_l=d_t_total[0]
    d_t_r=d_t_total[1]
    return np.kron(np.eye(d_t_l),delta_b(delta_t, d_t_r))       

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


def u_evol(Hm:np.ndarray|list, d_sys_total:np.array, d_t_total:np.array, interacting_timebins_num:int=1) -> np.ndarray|list:
    """
    Creates a time evolution operator for a given Hamiltonian.

    Parameters
    ----------
    Hm : ndarray
        Hamiltonian of the system.

    d_sys : int, default: 2
        Size of the Hilbert space of the matter system.
    
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

    interacting_timebins_num : int, default: 1
        Number of light channels/feedback loops involved in the Hamiltonian.

    Returns
    -------
    oper : ndarray
        ndarray time evolution operator of shape ((d_sys,) + (d_t,)*interacting_timebins_num)*2.
    
    Examples
    -------- 
    """ 
    d_t=np.prod(d_t_total)
    d_sys=np.prod(d_sys_total)
    shape = ((d_sys,) + ((d_t,)*interacting_timebins_num)) * 2
    #For the time dependent hamiltonian
    if isinstance(Hm, list):
        sol=[]
        for h_i in Hm:
           sol.append(expm(-1j*h_i).reshape(shape)) 
    else:
        sol= expm(-1j*Hm).reshape(shape)
    return sol

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
def vectorized_swap(dim1:int, dim2:int) -> np.ndarray:
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


def expectation(a_list:np.ndarray, mpo:np.ndarray) -> complex:
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
    sol = ncon([np.conj(a_list),mpo,a_list],[[1,2,4],[2,3],[1,3,4]])
    return sol

def expectation_2(a_list:np.ndarray, mpo:np.ndarray) -> complex:
    """
    The expectation value of a 2-bin MPS with a given operator.

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
    sol = ncon([np.conj(a_list),mpo,a_list],[[1,2,5,4],[2,3,5,6],[1,3,6,4]])
    return sol

def tls_pop(d_sys:int=2) -> np.ndarray:    
    return np.real((sigmaplus() @ sigmaminus()))
    
def a_r_pop(delta_t:float, d_t_total:np.array) -> np.ndarray:
    return np.real((delta_b_dag_r(delta_t,d_t_total) @ delta_b_r(delta_t,d_t_total))/delta_t)

def a_l_pop(delta_t:float, d_t_total:np.array) -> np.ndarray:  
    return np.real((delta_b_dag_l(delta_t,d_t_total) @ delta_b_l(delta_t,d_t_total))/delta_t)

def a_pop(delta_t:float, d_t:int=2) -> np.ndarray:  
    return np.real((delta_b_dag(delta_t,d_t) @ delta_b(delta_t,d_t))/delta_t)

def g1_rr(delta_t:float,d_t_total:np.array):
    b = np.kron(delta_b_dag_r(delta_t,d_t_total),delta_b_r(delta_t,d_t_total))
    d_t=np.prod(d_t_total)
    b=b.reshape(d_t,d_t,d_t,d_t)
    return b

def g1_rl(delta_t:float,d_t_total:np.array):
    b = np.kron(delta_b_dag_r(delta_t,d_t_total),delta_b_l(delta_t,d_t_total))
    d_t=np.prod(d_t_total)
    b=b.reshape(d_t,d_t,d_t,d_t)
    return b

def g1_lr(delta_t:float,d_t_total:np.array):
    b = np.kron(delta_b_dag_l(delta_t,d_t_total),delta_b_r(delta_t,d_t_total))
    d_t=np.prod(d_t_total)
    b=b.reshape(d_t,d_t,d_t,d_t)
    return b

def g1_ll(delta_t:float,d_t_total:np.array):
    b = np.kron(delta_b_dag_l(delta_t,d_t_total),delta_b_l(delta_t,d_t_total))
    d_t=np.prod(d_t_total)
    b=b.reshape(d_t,d_t,d_t,d_t)
    return b

def g2_rr(delta_t:float,d_t_total:np.array):
    b = np.kron(delta_b_dag_r(delta_t,d_t_total) @ delta_b_r(delta_t,d_t_total) ,delta_b_dag_r(delta_t,d_t_total) @ delta_b_r(delta_t,d_t_total)) 
    d_t=np.prod(d_t_total)
    b=b.reshape(d_t,d_t,d_t,d_t)
    return b

def g2_ll(delta_t:float,d_t_total:np.array):
    b = np.kron(delta_b_dag_l(delta_t,d_t_total) @ delta_b_l(delta_t,d_t_total) ,delta_b_dag_l(delta_t,d_t_total) @ delta_b_l(delta_t,d_t_total)) 
    d_t=np.prod(d_t_total)
    b=b.reshape(d_t,d_t,d_t,d_t)
    return b

def g2_rl(delta_t:float,d_t_total:np.array):
    b = np.kron(delta_b_dag_r(delta_t,d_t_total) @ delta_b_r(delta_t,d_t_total) ,delta_b_dag_l(delta_t,d_t_total) @ delta_b_l(delta_t,d_t_total)) 
    d_t=np.prod(d_t_total)
    b=b.reshape(d_t,d_t,d_t,d_t)
    return b

def g2_lr(delta_t:float,d_t_total:np.array):
    b = np.kron(delta_b_dag_l(delta_t,d_t_total) @ delta_b_l(delta_t,d_t_total) ,delta_b_dag_r(delta_t,d_t_total) @ delta_b_r(delta_t,d_t_total)) 
    d_t=np.prod(d_t_total)
    b=b.reshape(d_t,d_t,d_t,d_t)
    return b

def g1(delta_t:float,d_t_total:np.array):
    b = np.kron(delta_b_dag(delta_t,d_t_total),delta_b(delta_t,d_t_total))
    d_t=np.prod(d_t_total)
    b=b.reshape(d_t,d_t,d_t,d_t)
    return b

def g2(delta_t:float,d_t_total:np.array):
    b = np.kron(delta_b_dag(delta_t,d_t_total) @ delta_b(delta_t,d_t_total) ,delta_b_dag(delta_t,d_t_total) @ delta_b(delta_t,d_t_total)) 
    d_t=np.prod(d_t_total)
    b=b.reshape(d_t,d_t,d_t,d_t)
    return b


def steady_state_index(pop,window=10, tol=1e-5):
    """
    pop : list or array of population values
    window : number of recent points to analyze
    tol : maximum deviation allowed in the final window
    """
    # import warnings

    pop = np.asarray(pop)
    for i in range(window, len(pop)):
        tail = pop[i-window:i]
        if tail.max() - tail.min() > tol:
            continue
        if np.max(np.abs(np.diff(tail))) > tol:
            continue
        return i - window
    # warnings.warn("tmax not long enough for steady state")
    return None