#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module contains data classes for variables that
are often repeated throughout the QwaveMPS repository.

It collects lightweight data containers that hold
parameters, binning information and 
precomputed correlation/population arrays.

"""

from dataclasses import dataclass
import numpy as np

@dataclass
class InputParams:
    """Input / simulation parameters:
        
    - delta_t: float
        Time step used for time propagation.
    - tmax: float
        Maximum simulation time.
    - d_sys_total: np.ndarray
        Array describing the local physical dimensions of the system bins.
    - d_t_total: np.ndarray
        Array with dimensions for the time bins.
    - bond_max: int
        Maximum MPS bond dimension (chi) to use for truncation.
    - gamma_l, gamma_r: float, (is deprecated and will be removed in future versions)
        Coupling (decay) rates to the left and right channels respectively.
    - gamma_l2, gamma_r2: float, optional, (is deprecated and will be removed in future versions)
        Optional second set of coupling rates (e.g. for a second TLS).
        Default 0 means only one system.
    - tau: float, optional
        Delay time if modelling non-Markovian dynamics.
        Default 0.
    - phase: float, optional, (is deprecated and will be removed in future versions)
        Relative delayed phase. Default 0.
    """
    delta_t: float
    tmax: float
    d_sys_total: np.ndarray
    d_t_total: np.ndarray
    bond_max: int
    gamma_l: float
    gamma_r: float
    gamma_l2:float= 0
    gamma_r2:float= 0
    tau: float = 0
    phase: float = 0
    
    
@dataclass
class Bins:
    """Bin metadata used for analysing time-dependent quantities.
   - system_states: list
       List of system bins used when calculating single time system observables.
   - output_field_states: list
       Time bins used when calculating single time field observables.
    - input_field_states: list
        List of input time bins used for calculating single time field observables incident the system resulting from
        defined initial field state.
   - correlation_bins: list
       Correlation bins used when computing output field photon correlation functions.
   - schmidt: list
       Schmidt decomposition system bins usen when calculating entanglement entropy.
   - loop_field_states: list, optional
       Tau (delay) bins used when calculating delayed field observables. This is the list of 
       field states entering the feedback loop at each time point.
   - schmidt_tau: list, optional
       Schmidt decomposition tau bins usen when calculating delayed entanglement entropy.
   """
    system_states: list     
    output_field_states: list
    input_field_states: list
    correlation_bins: list
    schmidt: list
    loop_field_states: list = None
    schmidt_tau: list = None
