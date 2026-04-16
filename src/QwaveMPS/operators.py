#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the main quantum operators written as MPOs and simple utilities used by QwaveMPS

It provides the following functions:
    
    - Pauli raising/lowering operators
    - Bosonic creation/annihilation operators for the time bins
    - Bosonic creation/annihilation operators for the user for flux calculations
    - Swap bin operators
    - Expectation operators for 1, 2, and N bins
    - Entropy calculation
    - Single time expectation function
    - Helper function to integrate over feedback windows    
Note 
----    
It requires the module ncon (pip install --user ncon)
        
"""


import numpy as np
from scipy.linalg import expm
from ncon import ncon
from QwaveMPS.parameters import InputParams
from QwaveMPS.states import basis

__all__ = ['sigmaplus', 'sigmaminus', 'e', 'tls_pop', 'transition_op', 'create', 'destroy',
           'extend_op', 'extend_ket',
           'b_dag', 'b', 'b_dag_l', 'b_dag_r', 'b_l', 'b_r', 'b_pop', 'b_pop_r', 'b_pop_l',
           'single_time_expectation', 'loop_integrated_statistics', 'entanglement',
           'delta_b_dag','delta_b','delta_b_dag_l','delta_b_dag_r','delta_b_l', 'delta_b_r']

#-----------------------------
# Helper function to test for list of operators
#-----------------------------
def op_list_check(op_list:object) -> bool:
    '''
    Checks if given variable is a list of operators (ndarrays), either [] or numpy list.
    
    Parameters
    ----------
    op_list : object
        Possible list of operators to be tested.

    Returns
    -------
    is_list : bool
        Truth value of if the passed variable is a list/np.ndarray that may contain operators

    '''
    return isinstance(op_list, (list, tuple)) \
        or (isinstance(op_list, np.ndarray) and op_list.ndim > 2)

#-----------------------------
# Helper functions to extend states/operators to higher tensor spaces
#-----------------------------
def extend_op(op:np.ndarray, dim_list:list[int], i:int) -> np.ndarray:
    """
    Extends an operator to a higher tensor space, with identities used for the extension to the other tensor spaces.
    
    Parameters
    ----------
    op : np.ndarray
        Operator acting in the single Hilbert space.
    
    dim_list : list[int]
        Ordered list of Hilbert space sizes.

    i : int
        Index of the tensor space in which the unextended operators exists.

    Returns
    -------
    op_ext : ndarray
        The extended version of the operator to act over the total tensor space produced by dim_list.
    """ 
    l_dim = int(np.prod(dim_list[:i]))
    r_dim = int(np.prod(dim_list[i+1:]))

    return np.kron(np.kron(np.eye(l_dim), op), np.eye(r_dim))

def extend_ket(ket:np.ndarray, dim_list:list[int], i:int) -> np.ndarray:
    """
    Extends a ket to a higher tensor space, with identities used for the extension to the other tensor spaces.
    
    Parameters
    ----------
    ket : np.ndarray
        Ket existing in the single Hilbert space.
    
    dim_list : list[int]
        Ordered list of Hilbert space sizes.

    i : int
        Index of the tensor space in which the unextended operators exists.

    Returns
    -------
    ket_ext : ndarray
        The extended version of the ket to exist over the total tensor space produced by dim_list.
    """ 
    l_dim = int(np.prod(dim_list[:i]))
    r_dim = int(np.prod(dim_list[i+1:]))

    return np.outer(np.outer(basis(l_dim,0), ket), basis(r_dim,0))

#-----------------------------
# TLS operators
#-----------------------------

def sigmaplus() -> np.ndarray:  
    """
    Raising operator for the Pauli spins, :math:`|e\\rangle \\langle g|`.

    Returns
    -------
    sigma_plus : ndarray
        The Pauli spin raising operator.
    """
    op = np.zeros((2,2),dtype=complex)
    op[1,0]=1.       
    return op

def sigmaminus() -> np.ndarray:  
    """
    Lowering operator for the Pauli spins, :math:`|g\\rangle\\langle e|`. 
    
    Returns
    -------
    sigma_minus : ndarray
        The Pauli spin lowering operator.
    """
    op = np.zeros((2,2),dtype=complex)
    op[0,1]=1.       
    return op

def e(d_sys:int=2) -> np.ndarray:
    """
    Projector onto the excited TLS state, :math:`|e\\rangle\\langle e|`.
    
    Parameters
    ----------
    d_sys : int, default: 2
        Size of the Hilbert space of the matter system.
        For a two level system have d_sys=2.

    Returns
    -------
    oper : ndarray
        The excited state projector
    """ 
    exc = np.zeros((d_sys,d_sys),dtype=complex)
    exc[1,1]=1.      
    return exc

def tls_pop(d_sys:int=2) -> np.ndarray:    
    """
    Single TLS population operator, :math:`\\sigma^+ \\sigma^-`. 

    Parameters
    ----------
    d_sys : int, default: 2
        Size of the Hilbert space of the matter system.
        For a two level system have d_sys=2.
    
    Returns
    -------
    pop_operator : ndarray
        Population operator for a TLS
    """ 
    return np.real((sigmaplus() @ sigmaminus()))

#-----------------------------
# General transition and bosonic operators
#-----------------------------
def transition_op(dim:int, i:int, j:int) -> np.ndarray:
    """
    Creates a transition operator of size 'dim' from basis state 'i' to basis state 'j'.
    
    Parameters
    ----------
    dim : int
        Size of the Hilbert space.
    
    i : int
        First basis vector in the outer product defining the transition operator.

    j : int
        Second basis vector in the outer product defining the transition operator.

    Returns
    -------
    creation_op : ndarray
        The annihilation operator.
    """ 
    return np.outer(basis(dim,i), basis(dim,j)) 

def create(dim:int) -> np.ndarray:
    """
    Creates a creation operator for a Hilbert space of size 'dim'.
    
    Parameters
    ----------
    dim : int
        Size of the Hilbert space.

    Returns
    -------
    creation_op : ndarray
        The creation operator.
    """ 
    return np.diag(np.sqrt(np.arange(1, dim, dtype=complex)), -1) 

def destroy(dim:int) -> np.ndarray:
    """
    Creates a annihilation operator for a Hilbert space of size 'dim'.
    
    Parameters
    ----------
    dim : int
        Size of the Hilbert space.

    Returns
    -------
    creation_op : ndarray
        The annihilation operator.
    """ 
    return np.diag(np.sqrt(np.arange(1, dim, dtype=complex)), 1)    

#-----------------------------
# Time bin noise operators
#-----------------------------

def delta_b_dag(delta_t:float, d_t:int=2) -> np.ndarray:  
    """
    Time bin noise creation operator scaled by :math:`\\sqrt{\\Delta t}` in the truncated Fock
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
        Time bin noise creation operator.
    """
    return np.sqrt(delta_t) * np.diag(np.sqrt(np.arange(1, d_t, dtype=complex)), -1) 

def delta_b(delta_t:float, d_t:int=2) -> np.ndarray:  
    """
    Time bin noise annihilation operator scaled by :math:`\\sqrt{\\Delta t}` in the truncated Fock
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
        Time bin noise creation operator.
    """      
    return np.sqrt(delta_t) * np.diag(np.sqrt(np.arange(1, d_t, dtype=complex)), 1)    

def delta_b_dag_l(delta_t:float, d_t_total:np.ndarray) -> np.ndarray:  
    """
    Left time bin noise creation operator for a system with two field channels,
    scaled by :math:`\\sqrt{\\Delta t}` in the truncated Fock basis.

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
    
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces (left and right channels).

    Returns
    -------
    oper : ndarray
        Left time bin noise creation operator.
    """ 
    d_t_l=d_t_total[0]
    d_t_r=d_t_total[1]
    return np.kron(delta_b_dag(delta_t, d_t_l),np.eye(d_t_r))     

def delta_b_dag_r(delta_t:float, d_t_total:np.ndarray) -> np.ndarray: 
    """
    Right time bin noise creation operator for a system with two field channels,
    scaled by :math:`\\sqrt{\\Delta t}` in the truncated Fock basis.

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
    
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces (left and right channels).

    Returns
    -------
    oper : ndarray
        Right time bin noise creation operator.
    """ 
    d_t_l=d_t_total[0]
    d_t_r=d_t_total[1]
    return np.kron(np.eye(d_t_l), delta_b_dag(delta_t, d_t_r))     

def delta_b_l(delta_t:float, d_t_total:np.ndarray) -> np.ndarray:  
    """
    Left time bin noise annihilation operator for a system with two field channels,
    scaled by :math:`\\sqrt{\\Delta t}` in the truncated Fock basis.

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
    
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces (left and right channels).

    Returns
    -------
    oper : ndarray
        Left time bin noise annihilation operator.
    """ 
    d_t_l=d_t_total[0]
    d_t_r=d_t_total[1]
    return np.kron(delta_b(delta_t, d_t_l),np.eye(d_t_r))     

def delta_b_r(delta_t:float, d_t_total:np.ndarray) -> np.ndarray:  
    """
    Right time bin noise annihilation operator for a system with two field channels,
    scaled by :math:`\\sqrt{\\Delta t}` in the truncated Fock basis.

    Parameters
    ----------
    delta_t : float
        Time step for system evolution.
        
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

    Returns
    -------
    oper : ndarray
        Right time bin noise annihilation operator (left and right channels).
    """ 
    d_t_l=d_t_total[0]
    d_t_r=d_t_total[1]
    return np.kron(np.eye(d_t_l),delta_b(delta_t, d_t_r))       

#------------------------------
# Normalized Bosonic Observable Operators
#------------------------------
def b_dag(params:InputParams) -> np.ndarray:  
    """
    Creation operator for observables in the truncated Fock basis.
    Normalized by :math:`\\frac{1}{\\sqrt{\\Delta t}}`.
    
    Parameters
    ----------
    params : InputParams
        Object that contains the simulation parameters (time step size and photonic tensor space sizes)

    Returns
    -------
    oper : ndarray
        Creation operator observable.
    """
    return delta_b_dag(params.delta_t, np.prod(params.d_t_total)) / params.delta_t

def b(params:InputParams) -> np.ndarray:  
    """
    Annihilation operator for observables in the truncated Fock basis.
    Normalized by :math:`\\frac{1}{\\sqrt{\\Delta t}}`.

    Parameters
    ----------
    params : InputParams
        Object that contains the simulation parameters (time step size and photonic tensor space sizes)

    Returns
    -------
    oper : ndarray
        Annihilation operator observable.
    """
    return delta_b(params.delta_t, np.prod(params.d_t_total)) / params.delta_t

def b_dag_l(params:InputParams) -> np.ndarray:  
    """
    Left creation operator for a system with two field channels in the truncated Fock basis.
    Normalized by :math:`\\frac{1}{\\sqrt{\\Delta t}}`.

    Parameters
    ----------
    params : InputParams
        Object that contains the simulation parameters (time step size and photonic tensor space sizes)

    Returns
    -------
    oper : ndarray
        Creation operator observable for left channel.
    """
    return delta_b_dag_l(params.delta_t, params.d_t_total) / params.delta_t   

def b_dag_r(params:InputParams) -> np.ndarray: 
    """
    Right creation operator for a system with two field channels, in the truncated Fock basis.
    Normalized by :math:`\\frac{1}{\\sqrt{\\Delta t}}`.

    Parameters
    ----------
    params : InputParams
        Object that contains the simulation parameters (time step size and photonic tensor space sizes)

    Returns
    -------
    oper : ndarray
        Creation operator observable for right channel.
    """
    return delta_b_dag_r(params.delta_t, params.d_t_total) / params.delta_t    

def b_l(params:InputParams) -> np.ndarray:  
    """
    Left annihilation operator for a system with two field channels in the truncated Fock basis.
    Normalized by :math:`\\frac{1}{\\sqrt{\\Delta t}}`.

    Parameters
    ----------
    params : InputParams
        Object that contains the simulation parameters (time step size and photonic tensor space sizes)

    Returns
    -------
    oper : ndarray
        Annihilation operator observable for left channel.
    """
    return delta_b_l(params.delta_t, params.d_t_total) / params.delta_t

def b_r(params:InputParams) -> np.ndarray:  
    """
    Right annihilation operator for a system with two field channels, in the truncated Fock basis.
    Normalized by :math:`\\frac{1}{\\sqrt{\\Delta t}}`.

    Parameters
    ----------
    params : InputParams
        Object that contains the simulation parameters (time step size and photonic tensor space sizes)

    Returns
    -------
    oper : ndarray
        Annihlation operator observable for right channel.
    """
    return delta_b_r(params.delta_t, params.d_t_total) / params.delta_t

def b_pop(params:InputParams) -> np.ndarray:  
    """
    Single-channel photonic population operator (normalized by :math:`\\frac{1}{\\Delta t}`).

    Parameters
    ----------
    params : InputParams
        Object that contains the simulation parameters (time step size and photonic tensor space sizes)

    Returns
    -------
    oper : ndarray
        Population operator observable.
    """
    return np.real((delta_b_dag(params.delta_t, np.prod(params.d_t_total)) 
                    @ delta_b(params.delta_t, np.prod(params.d_t_total)))/params.delta_t**2)

def b_pop_r(params:InputParams) -> np.ndarray:
    """
    Right-channel photonic population operator (normalized by :math:`\\frac{1}{\\Delta t}`).

    Parameters
    ----------
    params : InputParams
        Object that contains the simulation parameters (time step size and photonic tensor space sizes)

    Returns
    -------
    oper : ndarray
        Population operator for the right channel
    """
    return np.real((delta_b_dag_r(params.delta_t, params.d_t_total)
                     @ delta_b_r(params.delta_t, params.d_t_total))/params.delta_t**2)

def b_pop_l(params:InputParams) -> np.ndarray:  
    """
    Left-channel photonic population operator (normalized by :math:`\\frac{1}{\\Delta t}`).

    Parameters
    ----------
    params : InputParams
        Object that contains the simulation parameters (time step size and photonic tensor space sizes)

    Returns
    -------
    oper : ndarray
        Population operator observable for the left channel.
    """
    return np.real((delta_b_dag_l(params.delta_t, params.d_t_total)
                     @ delta_b_l(params.delta_t, params.d_t_total))/params.delta_t**2)


#-------------------
# Time evolution MPO
#-------------------

def u_evol(Hm:np.ndarray|list, d_sys_total:np.ndarray, d_t_total:np.ndarray, interacting_timebins_num:int=1) -> np.ndarray:
    """
    Creates a time evolution operator :math:`\\exp{-1j H}` for a given Hamiltonian,
    and reshape to expected tensor shape.

    Parameters
    ----------
    Hm : ndarray
        Hamiltonian of the system.

    d_sys_total : ndarray
        Sizes of the Hilbert spaces of the matter system.
    
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

    interacting_timebins_num : int, default: 1
        Number of light channels/feedback loops involved in the Hamiltonian.

    Returns
    -------
    oper : ndarray
        Time evolution operator of shape ((d_sys,) + (d_t,)*interacting_timebins_num)*2.
    """ 
    d_t=np.prod(d_t_total)
    d_sys=np.prod(d_sys_total)
    shape = (((d_t,)*interacting_timebins_num) + (d_sys,)) * 2
    sol= expm(-1j*Hm).reshape(shape)
    return sol

def u_evol_multi_emitter_bins(Hm:np.ndarray|list, d_sys_list:list[int], d_t_total:np.ndarray, interacting_timebins_num:int=1) -> np.ndarray:
    """
    Creates a time evolution operator :math:`\\exp{-1j H}` for a given Hamiltonian,
    and reshape to expected tensor shape, with the time bins to the left and the ordered
    list of system dimensions to the right.

    Parameters
    ----------
    Hm : ndarray
        Hamiltonian of the system.

    d_sys_list : list[int]
        Sizes of the emitter Hilbert spaces .
    
    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

    interacting_timebins_num : int, default: 1
        Number of light channels/feedback loops involved in the Hamiltonian.

    Returns
    -------
    oper : ndarray
        Time evolution operator of shape ((d_sys,) + (d_t,)*interacting_timebins_num)*2.
    """ 
    d_t=np.prod(d_t_total)
    shape = (((d_t,)*interacting_timebins_num) + tuple(d_sys_list)) * 2
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
            
    dim2 : int
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
            
    dim2 : int
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

def expectation_1bin(bin_state:np.ndarray, mpo:np.ndarray) -> complex:
    """
    The expectation value :math:`\\langle A|\\mathrm{MPO}|A\\rangle` of a single MPS bin with a given operator.

    Parameters
    ----------
    bin_state : ndarray
        MPS bin defining the state having an expectation taken with respect to some operator.

    mpo : ndarray
        Operator whose expectation value is being taken.
    
    Returns
    -------
    expectation value : complex
        The expectation value of the operator for the given state.
    """ 
    sol = ncon([np.conj(bin_state),mpo,bin_state],[[1,2,4],[2,3],[1,3,4]])
    return sol

def expectation_2bins(bin_state:np.ndarray, mpo:np.ndarray) -> complex:
    """
    The expectation value :math:`\\langle A|\\mathrm{MPO}|A\\rangle` of a 2-bin MPS with a given operator.

    Parameters
    ----------
    bin_state : ndarray
        MPS bin defining the state having an expectation taken with respect to some operator.

    MPO : ndarray
        Operator whose expectation value is being taken.
    
    Returns
    -------
    expectation value : complex
        The expectation value of the operator for the given state.
    """ 
    sol = ncon([np.conj(bin_state),mpo,bin_state],[[1,2,3,6],[4,5,2,3],[1,4,5,6]])
    return sol

def expectation_nbins(ket:np.ndarray, mpo:np.ndarray) -> complex:
    """ 
    General expectation utility: expectation operation ket for larger/arbitrary tensor spaces.
    Take the expectation value of an nth rank tensor ket with an nth rank MPO, :math:`\\langle A|\\mathrm{MPO}|A\\rangle`.
    
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

#-----------------
# Single Time Observables
#-----------------

def single_time_expectation(normalized_bins:list[np.ndarray], ops_list:np.ndarray|list[np.ndarray]) -> np.ndarray:
    """
    Compute expectation values of a list of operators on a list of OC normalized bins.

    Parameters
    ----------
    normalized_bins : list[ndarray]
        List of OC normalized bins in order of time to have localized expectation values taken.

    ops_list : ndarray/list[ndarray]
        List of operators or a single operator to take expectation values.
        Each operator must be compatible with the bin physical space.
    
    Returns
    -------
    expectation_values: np.ndarray[complex] | np.ndarray[np.ndarray[complex]]
        In the case of a single operator returns a list of expectation values for each time point.
        For a list of operators returns a 2D array shaped (len(ops_list), len(normalized_bins)) with expectation
        values for each operator at each time.
    """
    # Check if the operator is a list of operators, if  so return only the 0th element of the list
    is_list_flag = op_list_check(ops_list)
    if not is_list_flag:
        ops_list = [ops_list]

    result = np.array([[expectation_1bin(bin, op) for bin in normalized_bins] for op in ops_list])

    if not is_list_flag:
        result = result[0]

    return result

def loop_integrated_statistics(time_dependent_func:np.ndarray[complex], params:InputParams) -> np.ndarray:
    """
    Calculates the time dependent integral of the function over all points in the feedback loop at each time point of the system evolution.
    This is a moving windowed integral over the function, with the window size of length tau (the feedback time)
    For t<tau assumes time points yet to be reached in the loop will initially be 0.

    Parameters
    ----------
    time_dependent_func : np.ndarray[complex]
        List of values for the time dependent function to be integrated over the loop at each time point.
    
    params : InputParams
        Simulation parameters

    Returns
    -------
    observable_integrated_in_loop : np.ndarray
        List of values for the integration of time_dependent_func over the feedback loop at each time point.
        This is a moving integral over the function, for a window of length tau.
    """

    tau = params.tau
    delta_t = params.delta_t

    n=len(time_dependent_func) 
    observable_integrated_in_loop = np.zeros(n,dtype=complex)
    
    l=int(round(tau/delta_t,0))

    cumulative_sum = np.cumsum(time_dependent_func)
    observable_integrated_in_loop[:l+1] = cumulative_sum[:l+1]
    observable_integrated_in_loop[l:] = cumulative_sum[l:] - cumulative_sum[:-l]
    
    return observable_integrated_in_loop * delta_t

def entanglement(sch:list[np.ndarray]) -> list[float]:
    """
    Compute von Neumann entanglement entropy across a list of Schmidt coefficient arrays.

    Parameters
    ----------
    sch : list[np.ndarray]
        List of Schmidt coefficient arrays (s) for each bipartition.

    Returns
    -------
    time_dependent_entanglement : list[float]
        Entanglement entropies computed as :math:`-\\sum(p\\log_2 p)` where :math:`p = s^2`.
    """
    ent_list=[]
    for s in sch:
        sqrd_sch=s**2   
        sqrd_sch=np.trim_zeros(sqrd_sch) 
        log_sqrd_sch=np.log2(sqrd_sch)
        prod=sqrd_sch*log_sqrd_sch
        ent=-sum(prod)
        ent_list.append(ent)
    return ent_list
