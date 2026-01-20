#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the main quantum operators written as MPOs and simple utilities used by QwaveMPS

It provides the following functions:
    
    - Pauli raising/lowering operators
    - Bosonic creation/annihilation operators for the time bins
    - Swap bin operators
    - Expectation operators for 1 and 2 bins
    - Simpel correlator operator constructors
    - Steady state index helper 
    
Note 
----    
It requires the module ncon (pip install --user ncon)
        
"""


import numpy as np
from scipy.linalg import expm
from ncon import ncon

#-----------------------------
# 
#-----------------------------
def _op_list_check(op_list):
    '''
    Checks if given variable is a list of operators (ndarrays), either [] or numpy list.
    '''
    return isinstance(op_list, (list, tuple)) \
        or (isinstance(op_list, np.ndarray) and op_list.ndim > 2)

#-----------------------------
#Basic TLS and boson operators
#-----------------------------

def sigmaplus() -> np.ndarray:  
    """
    Raising operator for the Pauli spins (|e><g|).

    Returns
    -------
    oper : ndarray
        ndarray for the Pauli spin raising operator.
    """
    a = np.zeros((2,2),dtype=complex)
    a[1,0]=1.       
    return a

def sigmaminus() -> np.ndarray:  
    """
    Lowering operator for the Pauli spins  (|g><e|). 
    
    Returns
    -------
    oper : ndarray
        ndarray for the Pauli spin lowering operator.
    """
    a = np.zeros((2,2),dtype=complex)
    a[0,1]=1.       
    return a

def e(d_sys:int=2) -> np.ndarray:
    """
    Projector onto the excited TLS state (|e><e|).

    Parameters
    ----------
    d_sys : int, default: 2 (for a TLS)
        Size of the Hilbert space of the matter system.

    Returns
    -------
    oper : ndarray
        ndarray for the excited state projector
    """ 
    exc = np.zeros((d_sys,d_sys),dtype=complex)
    exc[1,1]=1.      
    return exc

def delta_b_dag(delta_t:float, d_t:int=2) -> np.ndarray:  
    """
    Time bin noise creation operator scaled by sqrt(delta_t) in the truncated Fock
    basis.
    
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
    """
    return np.sqrt(delta_t) * np.diag(np.sqrt(np.arange(1, d_t, dtype=complex)), -1) 

def delta_b(delta_t:float, d_t:int=2) -> np.ndarray:  
    """
    Time bin noise annihilation operator scaled by sqrt(delta_t) in the truncated Fock
    basis.

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
    """      
    return np.sqrt(delta_t) * np.diag(np.sqrt(np.arange(1, d_t, dtype=complex)), 1)    

def delta_b_dag_l(delta_t:float, d_t_total:np.array) -> np.ndarray:  
    """
    Left time bin noise creation operator for a system with two field channels,
    scaled by sqrt(delta_t) in the truncated Fock basis.

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
    
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces (left and right channels).

    Returns
    -------
    oper : ndarray
        ndarray left time bin noise creation operator.
    """ 
    d_t_l=d_t_total[0]
    d_t_r=d_t_total[1]
    return np.kron(delta_b_dag(delta_t, d_t_l),np.eye(d_t_r))     

def delta_b_dag_r(delta_t:float, d_t_total:np.array) -> np.ndarray: 
    """
    Right time bin noise creation operator for a system with two field channels,
    scaled by sqrt(delta_t) in the truncated Fock basis.

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
    
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces (left and right channels).

    Returns
    -------
    oper : ndarray
        ndarray right time bin noise creation operator.
    """ 
    d_t_l=d_t_total[0]
    d_t_r=d_t_total[1]
    return np.kron(np.eye(d_t_l), delta_b_dag(delta_t, d_t_r))     

def delta_b_l(delta_t:float, d_t_total:np.array) -> np.ndarray:  
    """
    Left time bin noise annihilation operator for a system with two field channels,
    scaled by sqrt(delta_t) in the truncated Fock basis.

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
    
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces (left and right channels).

    Returns
    -------
    oper : ndarray
        ndarray left time bin noise annihilation operator.
    """ 
    d_t_l=d_t_total[0]
    d_t_r=d_t_total[1]
    return np.kron(delta_b(delta_t, d_t_l),np.eye(d_t_r))     

def delta_b_r(delta_t:float, d_t_total:np.array) -> np.ndarray:  
    """
    Right time bin noise annihilation operator for a system with two field channels,
    scaled by sqrt(delta_t) in the truncated Fock basis.

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
        
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

    Returns
    -------
    oper : ndarray
        ndarray right time bin noise annihilation operator (left and right channels).
    """ 
    d_t_l=d_t_total[0]
    d_t_r=d_t_total[1]
    return np.kron(np.eye(d_t_l),delta_b(delta_t, d_t_r))       

#------------------------------
# Normalized Bosonic Observable Operators
#------------------------------
def a_dag(delta_t:float, d_t:int=2) -> np.ndarray:  
    """
    Creation operator for observables in the truncated Fock basis.
    
    Parameters
    ----------
    delta_t : float
        Time step for system evolution.

    d_t : int, default: 2
        Size of the truncated field Hilbert space

    Returns
    -------
    oper : ndarray
        ndarray creation operator observable.
    """
    return delta_b_dag(delta_t, d_t) / delta_t

def a(delta_t:float, d_t:int=2) -> np.ndarray:  
    """
    Annihilation operator for observables in the truncated Fock basis.

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
    
    d_t : int, default: 2
        Size of the truncated field Hilbert space

    Returns
    -------
    oper : ndarray
        ndarray annihilation operator observable.
    """      
    return delta_b(delta_t, d_t) / delta_t

def a_dag_l(delta_t:float, d_t_total:np.array) -> np.ndarray:  
    """
    Left creation operator for a system with two field channels in the truncated Fock basis.

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
    
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces (left and right channels).

    Returns
    -------
    oper : ndarray
        ndarray left creation operator observable.
    """ 
    return delta_b_dag_l(delta_t, d_t_total) / delta_t   

def a_dag_r(delta_t:float, d_t_total:np.array) -> np.ndarray: 
    """
    Right creation operator for a system with two field channels, in the truncated Fock basis.

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
    
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces (left and right channels).

    Returns
    -------
    oper : ndarray
        ndarray right creation operator observable.
    """ 
    return delta_b_dag_r(delta_t, d_t_total) / delta_t    

def a_l(delta_t:float, d_t_total:np.array) -> np.ndarray:  
    """
    Left annihilation operator for a system with two field channels in the truncated Fock basis.

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
    
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces (left and right channels).

    Returns
    -------
    oper : ndarray
        ndarray left annihilation operator observable.
    """ 
    return delta_b_l(delta_t, d_t_total) / delta_t

def a_r(delta_t:float, d_t_total:np.array) -> np.ndarray:  
    """
    Right annihilation operator for a system with two field channels, in the truncated Fock basis.

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
        
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces (left and right channels).

    Returns
    -------
    oper : ndarray
        ndarray right annihilation operator observable.
    """ 
    return delta_b_r(delta_t, d_t_total) / delta_t

#-------------------
#Time evolution MPO
#-------------------

def u_evol(Hm:np.ndarray|list, d_sys_total:np.array, d_t_total:np.array, interacting_timebins_num:int=1) -> np.ndarray|list:
    """
    Creates a time evolution operator exp(-1j H) for a given Hamiltonian,
    and reshape to expected tensor shape.

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

#----------
#Swap MPOs
#----------

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
    """ 
    size = dim1 * dim2
    swap = np.zeros([size,size],dtype=complex)
    for i in range(dim1):
        for j in range(dim2):
            swap[i + j*dim1,(i*dim2)+j]=1
    return swap.reshape(dim1,dim2,dim1,dim2)   
 
def vectorized_swap(dim1:int, dim2:int) -> np.ndarray:
    """
    Vectorized version of swap tensor to swap the contents of adjacent MPS bins.
    Less performant for small dims, but slightly faster for large dims

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
    """ 
    size = dim1*dim2
    swap = np.zeros((size, size), dtype=complex)
    indices = np.array([(i%dim2)*dim1 + int(i/dim2) for i in range(size)], dtype=int)
    swap[indices, np.arange(swap.shape[0])] = 1
    return swap.reshape(dim1,dim2,dim1,dim2)

#-----------------
#Expectation MPOs
#-----------------

def expectation_1bin(a_list:np.ndarray, mpo:np.ndarray) -> complex:
    """
    The expectation value <A|MPO|A> of a single MPS bin with a given operator.

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
    """ 
    sol = ncon([np.conj(a_list),mpo,a_list],[[1,2,4],[2,3],[1,3,4]])
    return sol

def expectation_2bins(a_list:np.ndarray, mpo:np.ndarray) -> complex:
    """
    The expectation value <A|MPO|A> of a 2-bin MPS with a given operator.

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
    """ 
    sol = ncon([np.conj(a_list),mpo,a_list],[[1,2,5,4],[2,3,5,6],[1,3,6,4]])
    return sol

def expectation_nbins(ket:np.ndarray, mpo:np.ndarray) -> complex:
    """ 
    General expectation utility: expectation operation ket for larger/arbitrary tensor spaces.
    Take the expectation value of an nth rank tensor ket with an nth rank MPO.
    
    This helper caches index ordering logic depending on the operator rank to avoid
    recomputing index lists repeatedly for identical operator ranks.
    
    Parameters
    ----------
    ket : ndarray
        Ket for taking the expectation value
    
    mpo : ndarray
        Matrix product operator for the expectation value.

    Returns
    -------
    result : complex
        The expectation value of the operator for the given ket.
        <ket| mpo |ket>
    """

    curr_rank_op = len(mpo.shape)+2 #Adjusted for indices numbering
    if expectation_nbins.prev_rank != curr_rank_op:
        expectation_nbins.prev_rank = curr_rank_op
        half_rank_op = int(curr_rank_op/2)+1
        expectation_nbins.ket_indices = np.concatenate((np.arange(1,half_rank_op, dtype=int), [curr_rank_op])).tolist()
        expectation_nbins.op_indices = np.concatenate((np.arange(half_rank_op, curr_rank_op, dtype=int), 
                                                       np.arange(2,half_rank_op, dtype=int))).tolist()
        expectation_nbins.bra_indices = np.concatenate(([1], np.arange(half_rank_op,curr_rank_op+1, dtype=int))).tolist()

    return ncon([np.conj(ket), mpo, ket], [expectation_nbins.ket_indices, expectation_nbins.op_indices, expectation_nbins.bra_indices])

# initialize cache attributes for expectation_n
expectation_nbins.prev_rank = None

def single_time_expectation(normalized_bins:list[np.ndarray], ops_list:list[np.ndarray]) -> np.ndarray:
    """
    Compute expectation values of a list of operators on a list of OC normalized bins.

    Parameters
    ----------
    normalized_bins : list[ndarray]
        List of OC normalized bins in order of time to have localized expectation values taken.

    ops_list : list[ndarray]
        List of operators to take expectation values.
        Each operator must be compatible with the bin physical space.
    Returns
    -------
    np.ndarray
        2D array shaped (len(ops_list), len(normalized_bins)) with expectation
        values for each operator at each time.
    """
    # Check if the operator is a list of operators, if  so return only the 0th element of the list
    is_list_flag = _op_list_check(ops_list)
    if not is_list_flag:
        ops_list = [ops_list]

    result = np.array([[expectation_1bin(bin, op) for bin in normalized_bins] for op in ops_list])

    if not is_list_flag:
        result = result[0]

    return result

#-----------------
#Population MPOs
#-----------------

def tls_pop(d_sys:int=2) -> np.ndarray:    
    """
    Single TLS population operator sigma^+ sigma^-. 

    Parameters
    ----------
    d_sys : int, default: 2 (for a TLS)
        Size of the Hilbert space of the matter system.
    
    Returns
    -------
    Population operator for a TLS: np.ndarray
    """ 
    return np.real((sigmaplus() @ sigmaminus()))
    
def a_r_pop(delta_t:float, d_t_total:np.array) -> np.ndarray:
    """
    Right-channel photonic population operator (normalized by delta_t).

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
        
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.
    
    Returns
    -------
    Population of the right-channel photons: np.ndarray
    """ 
    return np.real((delta_b_dag_r(delta_t,d_t_total) @ delta_b_r(delta_t,d_t_total))/delta_t)

def a_l_pop(delta_t:float, d_t_total:np.array) -> np.ndarray:  
    """
    Left-channel photonic population operator (normalized by delta_t).

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
        
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.
    
    Returns
    -------
    Population of the left-channel photons: np.ndarray
    """ 
    return np.real((delta_b_dag_l(delta_t,d_t_total) @ delta_b_l(delta_t,d_t_total))/delta_t)

def a_pop(delta_t:float, d_t:int=2) -> np.ndarray:  
    """
    Single-channel photonic population operator (normalized by delta_t).

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
        
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.
    
    Returns
    -------
    Photonic population fora single channel solution: np.ndarray
    """ 
    return np.real((delta_b_dag(delta_t,d_t) @ delta_b(delta_t,d_t))/delta_t)
