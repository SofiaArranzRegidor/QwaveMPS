"""
This module contains the functions used to calculate two time point correlation functions
and supporting functions that might be used on such calculated correlation functions

It provides full two time point correlation calculation, calculation of a single cross sections
of a two time correlation function, steady state correlation functions, and spectra.

It requires the module ncon (pip install --user ncon)

"""

import numpy as np
import copy
from ncon import ncon

from QwaveMPS.src.simulation import _svd_tensors
from QwaveMPS.src.operators import op_list_check,expectation_1bin,expectation_nbins,swap,single_time_expectation
from QwaveMPS.src.parameters import InputParams

# ----------------------
# Functions acting on correlation results
# ----------------------


def spectrum_w(delta_t:float, g1_list: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the (discrete) spectrum in the long-time limit via Fourier transform 
    of the two-time first-order correlation (steady-state solution).

    Parameters
    ----------
    delta_t : float
        Time step used in the simulation; used to set frequency sampling.
   
    g1_list : np.ndarray
        Steady-state first order correlation.

    Returns
    -------
    s_w : np.ndarray
        Spectrum in the long-time limit (steady state solution)
    wlist : np.ndarray
        Corresponding frequency list.
    """
    s_w = np.fft.fftshift(np.fft.fft(g1_list))
    n=s_w.size
    wlist = np.fft.fftshift(np.fft.fftfreq(n,d=delta_t))*2*np.pi   
    return s_w,wlist

def transform_t_tau_to_t1_t2(positive_tau_results:np.ndarray, negative_tau_results:np.ndarray=None) -> np.ndarray:
    """
    Transforms two time correlations from a (t,tau) representation to a (t1,t2) representation.
    Takes the computed correlation function with operators ordered for the positive and negative tau (t>t+tau) cases.
    Note that this truncates the result to maintain the same overall shape by having t2 have the same domain as t1 (truncates cases where t+tau>t1_max).
    If only given one matrix assumes symmetry over the tau axis.

    Parameters
    ----------
    positive_tau_results : np.ndarray
        Computed two time correlation function in the case of operators ordered for positive tau data.
   
    negative_tau_results : np.ndarray, default: None
        Computed two time correlation function in the case of operators ordered for negative tau data.
        If None, uses the positive_tau_results, treating the observable as symmetric over the tau axis.

    Returns
    -------
    transformed_t1_t2_data : np.ndarray
        Truncated data with (t1,t2) axes.
    """
    if negative_tau_results is None:
        negative_tau_results = positive_tau_results

    transformed_t1_t2_data = np.zeros(negative_tau_results.shape, dtype=complex)
    t_size, tau_size = negative_tau_results.shape 
    
    # Shape is square, this indexing requires equal number of values for t/tau
    # TODO: If adding ability to measure subsections of the two time correlation have to update this
    i, j = np.triu_indices(t_size)
    
    # Add contributions from both t>= tau and t<= tau (diagonal is equal)
    transformed_t1_t2_data[i, j] = positive_tau_results[i, j - i]
    transformed_t1_t2_data[j, i] = negative_tau_results[i, j - i]

    return transformed_t1_t2_data

def spectral_intensity(correlation_matrix:np.ndarray, input_params:InputParams, padding:int=0, hanning_filter:bool=False, taper_length:int=16) -> tuple[np.ndarray,np.ndarray]:
    """
    Calculate the time dependent spectral intensity from a given two time correlation function.
    
    Parameters
    ----------
    correlation_matrix : np.ndarray
        Computed two time correlation matrix used for the calculation of the spectral intensity. 
   
    input_params : InputParams
        Input parameters of the simulation

    padding : int, default=0
        Number of 0's added to the Fourier transform as padding for smoother results.

    hanning_filter : bool, default=False
        Determines whether or not a Hanning filter is used to smooth the decay at the end of the function for a smoother result.

    taper_length : int, default=16
        Determines the number of time points from the end of the data on which the Hanning filter is applied.
        Only relevant if hanning_filter is True.

    Returns
    -------
    spectral_intensity : np.ndarray
        The computed time dependent spectral intensity of the given correlation function.
    w_list : np.ndarray
        List of frequencies associated with the calculated spectral intensity.
    """
    delta_t = input_params.delta_t

    correlation_matrix_copy = copy.deepcopy(correlation_matrix)
    # Taper end of signal if using filter
    if hanning_filter:
        taper_window = np.hanning(2*taper_length)[taper_length:]
        correlation_matrix_copy[:,-taper_length:] *= taper_window
    
    spectral_intensity = np.fft.fftshift(np.fft.fft(correlation_matrix_copy, axis=1, n=correlation_matrix_copy.shape[1]+padding), axes=1)
    
    # Multiply by factor of delta_t for correct scaling to mimic the continuous FT
    spectral_intensity *= delta_t

    w_list = np.fft.fftshift(np.fft.fftfreq(spectral_intensity.shape[1], d=delta_t))
    w_list = w_list * 2.0 * np.pi

    return np.real(spectral_intensity), w_list

def time_dependent_spectrum(correlation_matrix:np.ndarray, input_params:InputParams, w_list:np.ndarray=None, padding:int=0) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the time dependent spectra from a given two time correlation function.
    
    Parameters
    ----------
    correlation_matrix : np.ndarray
        Computed two time correlation matrix used for the calculation of the spectral intensity. 
   
    input_params : InputParams
        Input parameters of the simulation

    w_list : np.ndarray, default=None
        Frequency points at which the time dependent spectrum is calculated.
        If None, generates the frequency list using np.fft.fftfreq() on the length of the correlation_matrix.

    padding : int, default: 0
        Padding added to the frequency domain.
        
    Returns
    -------
    spectrum : np.ndarray
        The computed time dependent spectrum of the given correlation function.
    w_list : np.ndarray
        List of frequencies associated with the calculated time dependent spectrum.
    """
    delta_t = input_params.delta_t
    correlation_matrix_copy = copy.deepcopy(correlation_matrix)
    size = correlation_matrix_copy.shape[0]

    if w_list is None:
        w_list = np.fft.fftshift(np.fft.fftfreq(size+padding, d=delta_t)) * 2.0 * np.pi
    spectrum = np.zeros((size, len(w_list)), dtype=np.complex128)
          
    # Try to use numpy numerical integration over rank 3 tensor created: [t'][t''][omega]
    integration_elements = np.broadcast_to(correlation_matrix_copy, (len(w_list),*correlation_matrix_copy.shape)).astype(np.complex128).copy() # Puts omega index first

    # Calculate phase factors, store result as [omega, t', tau]
    omegas = w_list[:, np.newaxis]
    tau = np.arange(size)[np.newaxis,:]
    phase_factors = np.exp(1j * omegas * tau * delta_t)
    integration_elements *= phase_factors[:,np.newaxis,:]
    
    # Have integration elements in form [omega, t', tau]

    # Direct summation (works correctly... coud it be faster)
    # Remove tPrime summation somehow (then can avoid += with just assignment and slice)?
    # Switch order of loops somehow? (substitution of summation indices?)
    
    # Cummulative sum for the function of t, outer most integral
    cummulative_sums = np.cumsum(integration_elements, axis=2)
    for t in range(size):
        for t_prime in range(t+1):
            spectrum[t,:] += cummulative_sums[:,t_prime,t-t_prime]
    
    spectrum *= delta_t**2
    return np.real(spectrum), w_list

# ----------------------
# Two time point Correlation functions
# Wrapper functions designed in qutip sytle (these are less general)
# ----------------------

def correlation_2op_2t(correlation_bins:list[np.ndarray], a_op_list:np.ndarray|list[np.ndarray], b_op_list:np.ndarray|list[np.ndarray], params:InputParams, completion_print_flag:bool=True) -> tuple[list[np.ndarray]|np.ndarray, np.ndarray]:
    """ 
    Calculates the two time correlation function <A(t)B(t+t')> for either single operators A and B, or each A/B in a_op_list/b_op_list.
    Provides list functionality as a single function call with a list of operators is much faster than individual function calls
    for each operator.
            
    Parameters
    ----------
    correlation_bins : list[ndarray]
        List of time bins with the OC in either the initial or final bin in the list.
    
    a_op_list : ndarray/list[ndarray]
        Single operator, A, or a list of operators.
    
    b_op_list : ndarray/list[ndarray]
        Single operator, B, or a list of operators.

    params : InputParams
        Simulation parameters 
    
    completion_print_flag : bool, default: True
        Prints the percent completion of the of the outer loop over t values for the calculation.
        Note that each loop is shorter, resulting in the percents being weighted more heavily to the start of the calculation.

    Returns
    -------
    correlations : list[np.ndarray]
        In the case of single A and B operators a 2D array. In the case of a list of operators returns a
        list of 2D arrays, each a two time correlation function corresponding by index to the operators in the two operator lists.
        The two time correlation function is stored as f[t,t'], with non-negative t' and time increments between points given by the simulation.
    
    t_list : np.ndarray
        List of time points for the t and t' axes.
    """
    list_flag = op_list_check(a_op_list)

    if list_flag and len(a_op_list) != len(b_op_list):
        raise ValueError("Lengths of operators lists are not equals")
    
    ops_same_time = []; ops_two_time = []

    if list_flag:
        for i in range(len(a_op_list)):
            ops_same_time.append(a_op_list[i] @ b_op_list[i])
            ops_two_time.append(np.kron(a_op_list[i], b_op_list[i]))
    else:
        ops_same_time.append(a_op_list @ b_op_list)
        ops_two_time.append(np.kron(a_op_list, b_op_list))
    
    results, t_list =  correlations_2t(correlation_bins, ops_same_time, ops_two_time, params, completion_print_flag=completion_print_flag)

    if not list_flag:
        results = results[0]

    return results, t_list

def correlation_4op_2t(correlation_bins:list[np.ndarray], a_op_list:np.ndarray|list[np.ndarray], b_op_list:np.ndarray|list[np.ndarray], c_op_list:np.ndarray|list[np.ndarray], d_op_list:np.ndarray|list[np.ndarray], params:InputParams, completion_print_flag:bool=True) -> tuple[list[np.ndarray]|np.ndarray, np.ndarray]:
    """ 
    Calculates the two time correlation function <A(t)B(t+t')C(t+t')D(t)> for either single operators A/B/C/D, or each operator in the four lists.
    Provides list functionality as a single function call with a list of operators is much faster than individual function calls
    for each operator.
            
    Parameters
    ----------
    correlation_bins : list[ndarray]
        List of time bins with the OC in either the initial or final bin in the list.
    
    a_op_list : ndarray/list[ndarray]
        Single operator, A, or a list of operators.
    
    b_op_list : ndarray/list[ndarray]
        Single operator, B, or a list of operators.
    
    c_op_list : ndarray/list[ndarray]
        Single operator, C, or a list of operators.

    d_op_list : ndarray/list[ndarray]
        Single operator, D, or a list of operators.

    params : InputParams
        Simulation parameters 

    completion_print_flag : bool, default: True
        Prints the percent completion of the of the outer loop over t values for the calculation.
        Note that each loop is shorter, resulting in the percents being weighted more heavily to the start of the calculation.

    Returns
    -------
    correlations : list[np.ndarray]
        In the case of single operators a 2D array. In the case of a list of operators returns a
        list of 2D arrays, each a two time correlation function corresponding by index to the operators in the two operator lists.
        The two time correlation function is stored as f[t,t'], with non-negative t' and time increments between points given by the simulation.
    
    t_list : np.ndarray
        List of time points for the t and t' axes.
    """
    list_flag = op_list_check(a_op_list)

    if list_flag and not (len(a_op_list) == len(b_op_list) == len(c_op_list) == len(d_op_list)):
        raise ValueError("Lengths of operators lists are not equal")


    ops_same_time = []; ops_two_time = []

    if list_flag:
        for i in range(len(a_op_list)):
            ops_same_time.append(a_op_list[i] @ b_op_list[i] @ c_op_list[i] @ d_op_list[i])
            ops_two_time.append(np.kron(a_op_list[i] @ d_op_list[i], b_op_list[i] @ c_op_list[i]))
    else:
        ops_same_time.append(a_op_list @ b_op_list @ c_op_list @ d_op_list)
        ops_two_time.append(np.kron(a_op_list @ d_op_list, b_op_list @ c_op_list))
    
    results, t_list = correlations_2t(correlation_bins, ops_same_time, ops_two_time, params, completion_print_flag=completion_print_flag)

    # Don't return as list
    if not list_flag:
        results = results[0]
    return results, t_list

def correlation_2op_1t(correlation_bins:list[np.ndarray], a_op_list:np.ndarray|list[np.ndarray], b_op_list:np.ndarray|list[np.ndarray], t:float, params:InputParams) -> tuple[list[np.ndarray]|np.ndarray, np.ndarray]:
    """ 
    Calculates the two time correlation function <A(t)B(t+t')> at a fixed t for either single operators A/B, or each operator in the lists.
    Provides list functionality as a single function call with a list of operators is much faster than individual function calls
    for each operator.
            
    Parameters
    ----------
    correlation_bins : list[ndarray]
        List of time bins with the OC in either the initial or final bin in the list.
    
    a_op_list : ndarray/list[ndarray]
        Single operator, A, or a list of operators.
    
    b_op_list : ndarray/list[ndarray]
        Single operator, B, or a list of operators.
    
    params : InputParams
        Simulation parameters 

    Returns
    -------
    correlations : list[np.ndarray]
        In the case of single operators a 1D array. In the case of a list of operators returns a
        list of 1D arrays, each a two time correlation function of fixed t, corresponding by index to the operators in the two operator lists.
        The two time correlation function is stored as f[t'], with time increments between points given by the simulation.
    
    t_list : np.ndarray
        List of time points for the t' axis.
    """
    list_flag = op_list_check(a_op_list)

    if list_flag and len(a_op_list) != len(b_op_list):
        raise ValueError("Lengths of operators lists are not equals")
    
    ops_same_time = []; ops_two_time = []

    if list_flag:
        for i in range(len(a_op_list)):
            ops_same_time.append(a_op_list[i] @ b_op_list[i])
            ops_two_time.append(np.kron(a_op_list[i], b_op_list[i]))
    else:
        ops_same_time.append(a_op_list @ b_op_list)
        ops_two_time.append(np.kron(a_op_list, b_op_list))
    
    results, t_list =  correlations_1t(correlation_bins, ops_same_time, ops_two_time, t, params)

    if not list_flag:
        results = results[0]

    return results, t_list

def correlation_4op_1t(correlation_bins:list[np.ndarray], a_op_list:np.ndarray|list[np.ndarray], b_op_list:np.ndarray|list[np.ndarray], c_op_list:np.ndarray|list[np.ndarray], d_op_list:np.ndarray|list[np.ndarray], t:float, params:InputParams) -> tuple[list[np.ndarray]|np.ndarray, np.ndarray]:
    """ 
    Calculates the two time correlation function <A(t)B(t+t')C(t+t')D(t)> at a fixed t for either single operators, or each operator in the lists.
    Provides list functionality as a single function call with a list of operators is much faster than individual function calls
    for each operator.
            
    Parameters
    ----------
    correlation_bins : list[ndarray]
        List of time bins with the OC in either the initial or final bin in the list.
    
    a_op_list : ndarray/list[ndarray]
        Single operator, A, or a list of operators.
    
    b_op_list : ndarray/list[ndarray]
        Single operator, B, or a list of operators.
    
    c_op_list : ndarray/list[ndarray]
        Single operator, C, or a list of operators.

    d_op_list : ndarray/list[ndarray]
        Single operator, D, or a list of operators.
    
    params : InputParams
        Simulation parameters 

    Returns
    -------
    correlations : list[np.ndarray]
        In the case of single operators a 1D array. In the case of a list of operators returns a
        list of 1D arrays, each a two time correlation function of fixed t, corresponding by index to the operators in the two operator lists.
        The two time correlation function is stored as f[t'], with time increments between points given by the simulation.
    
    t_list : np.ndarray
        List of time points for the t' axis.
    """

    list_flag = op_list_check(a_op_list)

    if list_flag and not (len(a_op_list) == len(b_op_list) == len(c_op_list) == len(d_op_list)):
        raise ValueError("Lengths of operators lists are not equal")


    ops_same_time = []; ops_two_time = []

    if list_flag:
        for i in range(len(a_op_list)):
            ops_same_time.append(a_op_list[i] @ b_op_list[i] @ c_op_list[i] @ d_op_list[i])
            ops_two_time.append(np.kron(a_op_list[i] @ d_op_list[i], b_op_list[i] @ c_op_list[i]))
    else:
        ops_same_time.append(a_op_list @ b_op_list @ c_op_list @ d_op_list)
        ops_two_time.append(np.kron(a_op_list @ d_op_list, b_op_list @ c_op_list))
    
    results, t_list = correlations_1t(correlation_bins, ops_same_time, ops_two_time, t, params)

    # Don't return as list
    if not list_flag:
        results = results[0]
    return results, t_list

def correlation_ss_2op(correlation_bins:list[np.ndarray], output_field_states:list[np.ndarray], a_op_list:np.ndarray|list[np.ndarray], b_op_list:np.ndarray|list[np.ndarray], params:InputParams, tol:float=1e-5, window:int = 20, t_steady:float=None) -> tuple[list[np.ndarray]|np.ndarray, np.ndarray, float]:
    """ 
    Calculates the two time correlation function <A(t)B(t+t')> at a steady state value of t for either single operators, or each operator in the lists.
    Provides list functionality as a single function call with a list of operators is much faster than individual function calls
    for each operator. In that case calculates the steady states correlation from the greatest steady state time of the operators.
            
    Parameters
    ----------
    correlation_bins : list[ndarray]
        List of time bins with the OC in either the initial or final bin in the list.
    
    a_op_list : ndarray/list[ndarray]
        Single operator, A, or a list of operators.
    
    b_op_list : ndarray/list[ndarray]
        Single operator, B, or a list of operators.
        
    params : InputParams
        Simulation parameters 

    tol : float, default: 1e-5
        The tolerance for which convergence of the operators is determined. Used to find the steady state time.    
    
    window : int, default: 20
        Number of recent points to analyze when determining the steady state time.
        
    t_steady : float, default: None
        User defined steady state time. If not provided, steady state is determined by convergence
        of the same time expectation values of the observables.

    Returns
    -------
    correlations : list[ndarray]
        In the case of single operators a 1D array. In the case of a list of operators returns a
        list of 1D arrays, each a two time correlation function of fixed t at steady state, corresponding by index to the operators in the two operator lists.
        The two time correlation function is stored as f[t'], with time increments between points given by the simulation.
    
    t_list : ndarray
        List of time points for the t' axis.

    t_ss : float
        Time that steady state is reached.
    """
    list_flag = op_list_check(a_op_list)

    if list_flag and len(a_op_list) != len(b_op_list):
        raise ValueError("Lengths of operators lists are not equals")
    
    ops_same_time = []; ops_two_time = []

    if list_flag:
        for i in range(len(a_op_list)):
            ops_same_time.append(a_op_list[i] @ b_op_list[i])
            ops_two_time.append(np.kron(a_op_list[i], b_op_list[i]))
    else:
        ops_same_time.append(a_op_list @ b_op_list)
        ops_two_time.append(np.kron(a_op_list, b_op_list))
    
    results, tau_list, t_ss =  correlation_ss_1t(correlation_bins, output_field_states, ops_same_time, ops_two_time, params, window=window, tol=tol, t_steady=t_steady)

    if not list_flag:
        results = results[0]

    return results, tau_list, t_ss

def correlation_ss_4op(correlation_bins:list[np.ndarray], output_field_states:list[np.ndarray], a_op_list:np.ndarray|list[np.ndarray], b_op_list:np.ndarray|list[np.ndarray], c_op_list:np.ndarray|list[np.ndarray], d_op_list:np.ndarray|list[np.ndarray], params:InputParams, tol:float=1e-5, window:int=20, t_steady:float=None) -> tuple[list[np.ndarray]|np.ndarray, np.ndarray, float]:
    """ 
    Calculates the two time correlation function <A(t)B(t+t')C(t+t')D(t)> at a steady state value of t for either single operators, or each operator in the lists.
    Provides list functionality as a single function call with a list of operators is much faster than individual function calls
    for each operator. In that case calculates the steady states correlation from the greatest steady state time of the operators.
            
    Parameters
    ----------
    correlation_bins : list[ndarray]
        List of time bins with the OC in either the initial or final bin in the list.
    
    a_op_list : ndarray/list[ndarray]
        Single operator, A, or a list of operators.
    
    b_op_list : ndarray/list[ndarray]
        Single operator, B, or a list of operators.
    
    c_op_list : ndarray/list[ndarray]
        Single operator, C, or a list of operators.

    d_op_list : ndarray/list[ndarray]
        Single operator, D, or a list of operators.
    
    params : InputParams
        Simulation parameters 

    tol : float, default: 1e-5
        The tolerance for which convergence of the operators is determined. Used to find the steady state time.    
    
    window : int, default: 20
        Number of recent points to analyze when determining the steady state time.
        
    t_steady : float, default: None
        User defined steady state time. If not provided, steady state is determined by convergence
        of the same time expectation values of the observables.

    Returns
    -------
    correlations : list[ndarray]
        In the case of single operators a 1D array. In the case of a list of operators returns a
        list of 1D arrays, each a two time correlation function of fixed t at steady state, corresponding by index to the operators in the two operator lists.
        The two time correlation function is stored as f[t'], with time increments between points given by the simulation.
    
    t_list : ndarray
        List of time points for the t' axis.

    t_ss : float
        Time that steady state is reached.

    """
    list_flag = op_list_check(a_op_list)

    if list_flag and not (len(a_op_list) == len(b_op_list) == len(c_op_list) == len(d_op_list)):
        raise ValueError("Lengths of operators lists are not equal")


    ops_same_time = []; ops_two_time = []

    if list_flag:
        for i in range(len(a_op_list)):
            ops_same_time.append(a_op_list[i] @ b_op_list[i] @ c_op_list[i] @ d_op_list[i])
            ops_two_time.append(np.kron(a_op_list[i] @ d_op_list[i], b_op_list[i] @ c_op_list[i]))
    else:
        ops_same_time.append(a_op_list @ b_op_list @ c_op_list @ d_op_list)
        ops_two_time.append(np.kron(a_op_list @ d_op_list, b_op_list @ c_op_list))
    
    results, tau_list, t_ss = correlation_ss_1t(correlation_bins, output_field_states, ops_same_time, ops_two_time, params, tol=tol, window=window, t_steady=t_steady)

    # Don't return as list
    if not list_flag:
        results = results[0]

    return results, tau_list, t_ss


#-------------------------------------------
# The functional code used for correlation calculations.
# This is the logic, and are also more general functions 
# (can calculate ANY arbitrary two time correlation functions on the output field)
#-------------------------------------------
def correlations_2t(correlation_bins:list[np.ndarray], ops_same_time:list[np.ndarray], ops_two_time:list[np.ndarray], params:InputParams, completion_print_flag:bool=False) -> tuple[list[np.ndarray], np.ndarray]:
    """ 
    General two-time correlation calculator.
    Take in list of time ordered normalized (with OC) time bins at position of relevance.
    Calculate a list of arbitrary two time point correlation functions at t and t+t' for nonnegative t'. 
        
    Parameters
    ----------
    time_bin_list : [ndarray]
        List of time bins with the OC in either the initial or final bin in the list.
    
    ops_same_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that t'=0 (same time). These should exist in a single time-bin tensor space.

    ops_two_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that t' > 0. These should be ordered in a corresponding order to
        ops_same_time and should exist in a tensor space that is the outer product of two time bin tensor spaces, with the right space corresponding to the greater time.

    params : InputParams
        Simulation parameters 
       
    completion_print_flag : bool, default=True
        Flag to print completion loop number percent of the calculation (note this is not the percent completion, and later loops complete faster than earlier ones). 

    Returns
    -------
    result : list[np.ndarray]
        List of 2D arrays, each a two time correlation function corresponding by index to the operators in ops_same_time and ops_two_time.
        The two time correlation function is stored as f[t,t'], with non-negative t' and time increments between points given by the simulation.

    correaltion_times : np.ndarray[float]
        List of time points for the t and t' axes for the calculated correlation functions.
    """
    d_t_total=params.d_t_total
    bond=params.bond_max
    d_t=np.prod(d_t_total)
    
    time_bin_list_copy = copy.deepcopy(correlation_bins)
    swap_matrix = swap(d_t, d_t)
    
    # Resize two_time_ops if needed
    for i in range(len(ops_two_time)):
        ops_two_time[i] = ops_two_time[i].reshape((d_t,)*(2*2))
    
    correlations = np.array([np.zeros((len(time_bin_list_copy), len(time_bin_list_copy)), dtype=complex) for i in ops_two_time])
    
    # If the OC is at end of the time bin list, move it to the start (shifts OC from one end to other, index 0)
    for i in range(len(time_bin_list_copy)-1,0,-1):
        bin_contraction = ncon([time_bin_list_copy[i-1],time_bin_list_copy[i]],[[-1,-2,1],[1,-3,-4]])
        left_bin, stemp, right_bin = _svd_tensors(bin_contraction, bond, d_t, d_t)
        time_bin_list_copy[i] = right_bin #right normalized system bin    
        time_bin_list_copy[i-1]= left_bin * stemp[None,None,:] #OC on left bin
    
    # Loop over to fill in correlation matrices values
    if completion_print_flag:
        print('Correlation Calculation Completion:')
    loop_num = len(time_bin_list_copy) - 1
    print_rate = max(round(loop_num / 20.0), 1) # Print every 5%, 20/100
    for i in range(len(time_bin_list_copy)-1):

        i_1=time_bin_list_copy[0]
        i_2=time_bin_list_copy[1] 
        
        #for the first row (tau=0)
        for k in range(len(correlations)):
            correlations[k][i,0] = expectation_1bin(i_1, ops_same_time[k]) #this means I'm storing [t,tau] 
        
        #for the rest of the rows (column by column)
        for j in range(len(time_bin_list_copy)-1):       
            state=ncon([i_1,i_2],[[-1,-2,1],[1,-3,-4]]) 
            for k in range(len(correlations)):
                correlations[k][i,j+1] = expectation_nbins(state, ops_two_time[k]) #this means I'm storing [t,tau] 

            
            swapped_tensor=ncon([i_1,i_2,swap_matrix],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #swapping the time bin down the line
            i_t2, stemp, i_t1 = _svd_tensors(swapped_tensor, bond, d_t, d_t)

            i_1 = stemp[:,None,None] * i_t1 #OC tau bin            

            if j < (len(time_bin_list_copy)-2):                
                i_2=time_bin_list_copy[j+2] #next time bin for the next correlation
                time_bin_list_copy[j]=i_t2 #update of the increasing bin
            if j == len(time_bin_list_copy)-2:
                time_bin_list_copy[j]=i_t2
                time_bin_list_copy[j+1]= i_1
        
        #after the last value of the column we bring back the first time
        for j in range(len(time_bin_list_copy)-1,0,-1):            
            swapped_tensor=ncon([time_bin_list_copy[j-1],time_bin_list_copy[j],swap_matrix],[[-1,5,2],[2,6,-4],[-2,-3,5,6]])
            returning_bin, stemp, right_bin = _svd_tensors(swapped_tensor, bond, d_t, d_t)
            if j>1:
                #timeBinListCopy[j] = vt[range(chi),:].reshape(chi,dTime,timeBinListCopy[i].shape[-1]) #right normalized system bin    
                time_bin_list_copy[j] = right_bin #right normalized system bin    
                time_bin_list_copy[j-1]= returning_bin * stemp[None,None,:] #OC on left bin
            # Final iteration drop the returning bin
            if j == 1:
               time_bin_list_copy[j] = stemp[:,None,None] * right_bin
        time_bin_list_copy=time_bin_list_copy[1:]    #Truncating the start of the list now that are done with that bin (t=i)
        
        if i % print_rate == 0 and completion_print_flag:
            print(round((float(i)/loop_num)*100,1), '%')
    
    t_list = np.arange(len(correlation_bins)) * params.delta_t
    return correlations, t_list

def correlations_1t(correlation_bins:list[np.ndarray], ops_same_time:list[np.ndarray], ops_two_time:list[np.ndarray], t:float, params:InputParams) -> tuple[list[np.ndarray], np.ndarray]:
    """ 
    General two-time correlation calculator along a single axis.
    Take in list of time ordered normalized (with OC) time bins at position of relevance.
    Calculate a list of arbitrary two time point correlation functions at t and t+tau for nonnegative t'. 
        
    Parameters
    ----------
    time_bin_list : list[ndarray]
        List of time bins with the OC in either the initial or final bin in the list.
    
    ops_same_time : list[ndarray]
        List of operators of which correlation functions should be calculated in the case that t'=0 (same time). These should exist in a single time-bin tensor space.

    ops_two_time : list[ndarray]
        List of operators of which correlation functions should be calculated in the case that t' > 0. These should be ordered in a corresponding order to
        ops_same_time and should exist in a tensor space that is the outer product of two time bin tensor spaces, with the right space corresponding to the greater time.

    t : float
        Time point for fixed t at which to take the two time point correlation.
        
    params : InputParams
        Simulation parameters 

    Returns
    -------
    correlations : ndarray[ndarray[complex]]
        List of 1D arrays, each a two time correlation function corresponding by index to the operators in ops_same_time and ops_two_time.
        The two time correlation function is stored as f[t,t'], with non-negative t' and time increments between points given by the simulation.

    ts_correlation : ndarray[float]
        List of time points for the t' axis at which the two time point correlation functions are taken.
    """
    d_t_total=params.d_t_total
    bond=params.bond_max
    delta_t = params.delta_t
    d_t=np.prod(d_t_total)

    t_index = int(round(t / delta_t, 0))
    
    time_bin_list_copy = copy.deepcopy(correlation_bins) # Work on deep copy to not risk altering initial
    swap_matrix = swap(d_t, d_t)

    # Resize two_time_ops if needed
    for i in range(len(ops_two_time)):
        ops_two_time[i] = ops_two_time[i].reshape((d_t,)*(2*2)) # One for factor for bin number, 2 point
    
    # Truncate to steady state index and create appropriate 
    size = len(time_bin_list_copy)
    correlations = np.array([np.zeros(size, dtype=complex) for i in ops_two_time])
    
    # Move OC back to t_index, then swap that bin to start of list
    for i in range(size-1,t_index,-1):
        bin_contraction = ncon([time_bin_list_copy[i-1],time_bin_list_copy[i]],[[-1,-2,1],[1,-3,-4]])
        left_bin, stemp, right_bin = _svd_tensors(bin_contraction, bond, d_t, d_t)
        time_bin_list_copy[i] = right_bin #right normalized system bin    
        time_bin_list_copy[i-1]= left_bin * stemp[None,None,:] #OC on left bin

    # Swap bin the t_index bin backwards from t_index -> 0, with the OC
    for i in range(t_index, 0, -1):
        bin_contraction = ncon([time_bin_list_copy[i-1],time_bin_list_copy[i], swap_matrix],
                               [[-1,5,1],[1,6,-4], [-2,-3,5,6]])
        left_bin, stemp, right_bin = _svd_tensors(bin_contraction, bond, d_t, d_t)
        time_bin_list_copy[i] = right_bin #right normalized system bin    
        time_bin_list_copy[i-1]= left_bin * stemp[None,None,:] #OC on left bin

    # Calculate the rest of the points
    for i in range(0, size-1):
        i_1 = time_bin_list_copy[i]
        i_2 = time_bin_list_copy[i+1]
        state=ncon([i_1,i_2],[[-1,-2,1],[1,-3,-4]]) 

        # Calculate each two time point op
        for j in range(len(ops_two_time)):
            correlations[j][i] = expectation_nbins(state, ops_two_time[j])

        swaps=ncon([i_1,i_2,swap_matrix],[[-1,5,2],[2,6,-4],[-2,-3,5,6]])
        i_t1,stemp,i_t2=_svd_tensors(swaps,bond,d_t,d_t)

        # Now put OC in the right bin, i_t2, to move it up the chain
        time_bin_list_copy[i+1]= stemp[:,None,None] * i_t2 #OC on right bin

    # Calculate the single same time point
    # Shift values above t_index to the right to prepare for insertion
    
    for j in range(len(ops_same_time)):
        correlations[j][t_index+1:] = correlations[j][t_index:-1]
        correlations[j][t_index] = expectation_1bin(time_bin_list_copy[-1], ops_same_time[j])
    
    tau_list = (np.arange(size) - t_index) * delta_t
    return correlations, tau_list
    
#-------------------------------------------
#Steady-state index helper, and correlations
#-------------------------------------------

def operator_steady_state_index(output_field_states:list[np.ndarray], operator_list:list[np.ndarray], tol: float=1e-5, window: int=10) -> np.ndarray[int]:
    """
    Steady-state index helper function to find the time step 
    when the steady state is reached in the single time dynamics of each operator.
    
    Parameters
    ----------
    output_field_states : list[np.ndarray]
        List of OC normalized output_field_states/bins.

    operator_list : list[np.ndarray]
        List of single time point operators to test convergence of their expectation values.
    
    tol : float, default: 1e-5
        Maximum deviation allowed in the final window
    
    window : int, default: 10
        Number of recent points to analyze
        
    Returns
    -------
    steady_state_indices : ndarray[int]
        The index of the start of the steady window for each operator.
        For each operator that a steady state is not found, the array contains np.nan at that index.
    """
    op_num = len(operator_list)
    expectation_vals_list = single_time_expectation(output_field_states, operator_list)
    steady_state_indices = np.zeros(op_num, dtype=int)
    op_tracker = np.arange(op_num)
    
    for i in range(window, len(output_field_states)):
        for j in op_tracker:
            tail = expectation_vals_list[j][i-window:i]
            if tail.max() - tail.min() > tol:
                continue
            if np.max(np.abs(np.diff(tail))) > tol:
                continue
            
            # Steady state index has been found for the operator
            steady_state_indices[j] = i - window
            op_tracker = np.delete(op_tracker, np.where(op_tracker == j))     
        
        # Check if all steady state times have been found
        if len(op_tracker) == 0:
            break
    
    # Replace places without steady state with None
    for j in op_tracker:
        steady_state_indices[j] = np.nan
    return steady_state_indices

def steady_state_index(output_field_states:list[np.ndarray], tol: float=1e-5, window: int=10) -> np.ndarray[int]:
    """
    Steady-state index helper function to find the time step 
    when the steady state is reached in the single time dynamics of each operator.
    
    Parameters
    ----------
    output_field_states : list[np.ndarray]
        List of OC normalized output_field_states/bins.
    
    tol : float, default: 1e-5
        Maximum deviation allowed in the final window
    
    window : int, default: 10
        Number of recent points to analyze
        
    Returns
    -------
    steady_state_index : int
        The index of the start of the steady window for the output field.
    """
    bin_num = len(output_field_states)
    bin_dim = output_field_states[0].shape[1]
    contracted_bins = np.empty((bin_num, bin_dim, bin_dim), dtype=complex)
    #TODO Maybe in future have density matrix function
    contracted_bins[:window-1] = np.stack([ncon([output_field_states[i], np.conj(output_field_states[i])], 
                                              [[1,-1,2],[1,-2,2]]) for i in range(window-1)]) 
    for i in range(window, bin_num):
        contracted_bins[i] = ncon([output_field_states[i], np.conj(output_field_states[i])], 
                                              [[1,-1,2],[1,-2,2]])
        
        # Check if [i-window:i] bins are close in contents
        tail = contracted_bins[i-window:i]
        if np.allclose(tail, tail[0], atol=tol):
            return i - window        
            
    return None


def correlation_ss_1t(correlation_bins:list[np.ndarray], output_field_states:list[np.ndarray], ops_same_time:list[np.ndarray], ops_two_time:list[np.ndarray], params:InputParams, tol: float=1e-5, window: int=10, t_steady:float=None) -> tuple[list[np.ndarray], np.ndarray, float]:
    """
    Efficient steady-state correlation calculation.
    This computes time differences starting from a convergence index (steady-state
    index). It returns a list of the 1D correlation arrays corresponding to the operator list, 
    a list of tau points, and the initial t point at which steady state is considered.
    
    Parameters
    ----------
    correlation_bins : list[np.ndarray]
        Correlation bins of the outputfield states used to determine multi-timepoint correlation functions of the output field.
    
    output_field_states : list[np.ndarray]
        OC normalized output field states.

    ops_same_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that tau=0 (same time). These should exist in a single time-bin tensor space.

    ops_two_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that tau > 0. These should be ordered in a corresponding order to
        ops_same_time and should exist in a tensor space that is the outer product of two time bin tensor spaces, with the right space corresponding to the greater time.
    
    params : InputParams
        Simulation parameters 
    
    window : int, default: 10
        Number of recent points to analyze when determining the steady state time.
    
    tol : float, default: 1e-5
        Maximum deviation allowed in the final window for the steady state time.

    t_steady : float, default: None
        User defined steady state time. If not provided, steady state is determined by convergence
        of the same time expectation values of the observables.
        
    Returns
    -------
    correlations : list[ndarray]
        A list of 1D arrays, each a two time correlation function of fixed t at steady state, corresponding by index to the operators in the two operator lists.
        The two time correlation function is stored as f[t'], with time increments between points given by the simulation.
    
    t_list : ndarray
        List of time points for the t' axis.

    t_ss : float
        Time that steady state is reached.
    """
    d_t_total=params.d_t_total
    bond=params.bond_max
    delta_t = params.delta_t
    d_t=np.prod(d_t_total)
    
    time_bin_list_copy = copy.deepcopy(correlation_bins) # Work on deep copy to not risk altering initial
    swap_matrix = swap(d_t, d_t)
    
    # Resize two_time_ops if needed
    for i in range(len(ops_two_time)):
        ops_two_time[i] = ops_two_time[i].reshape((d_t,)*(2*2)) # One for factor for bin number, 2 point
    
    #First check convergence of all correlations if not given a time:
    if t_steady is None:
        conv_index =  steady_state_index(output_field_states,window=window, tol=tol)  
        if conv_index is None:
            raise ValueError("tmax not long enough for steady state to be reached")
        
        t_steady = conv_index * delta_t
    else:
        conv_index = int(round(t_steady / delta_t))
    
    # Truncate to steady state index and create appropriate 
    time_bin_list_copy=time_bin_list_copy[conv_index:]
    size = len(time_bin_list_copy)
    correlations = np.array([np.zeros(size, dtype=complex) for i in ops_two_time])
    
    # Move OC back to t, then forward for positive taus
    for i in range(size-1,0,-1):
        bin_contraction = ncon([time_bin_list_copy[i-1],time_bin_list_copy[i]],[[-1,-2,1],[1,-3,-4]])
        left_bin, stemp, right_bin = _svd_tensors(bin_contraction, bond, d_t, d_t)
        time_bin_list_copy[i] = right_bin #right normalized system bin    
        time_bin_list_copy[i-1]= left_bin * stemp[None,None,:] #OC on left bin

    # Calculate the single same time point
    for j in range(len(ops_same_time)):
        correlations[j][0] = expectation_1bin(time_bin_list_copy[0], ops_same_time[j])
    
    # Calculate the rest of the points
    for i in range(1, size):
        i_1 = time_bin_list_copy[i-1]
        i_2 = time_bin_list_copy[i]
        state=ncon([i_1,i_2],[[-1,-2,1],[1,-3,-4]]) 

        # Calculate each two time point op
        for j in range(len(ops_two_time)):
            correlations[j][i] = expectation_nbins(state, ops_two_time[j])

        swaps=ncon([i_1,i_2,swap_matrix],[[-1,5,2],[2,6,-4],[-2,-3,5,6]])
        i_t1,stemp,i_t2=_svd_tensors(swaps,bond,d_t,d_t)

        # Now put OC in the right bin, i_t2, to move it up the chain
        time_bin_list_copy[i]= stemp[:,None,None] * i_t2 #OC on right bin

    tau_list = np.arange(size) * delta_t
    return correlations, tau_list, t_steady
    
