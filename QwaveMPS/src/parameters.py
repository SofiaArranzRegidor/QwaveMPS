#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module contains data classes for the input parameters (which are used to specify the overall simulation environment)
and the Bins class (which includes the results of a simulation).

"""

from dataclasses import dataclass
import numpy as np

@dataclass
class InputParams:
    """Input / simulation parameters:
    
    Parameters
    ----------
    delta_t: float
        Time step used for time propagation.

    tmax: float
        Maximum simulation time.

    d_sys_total: np.ndarray
        Array describing the local physical dimensions of the system bins.
        Each index is associated with the size of a tensor space.

    d_t_total: np.ndarray
        Array with dimensions for the time bins.
        Each index is associated with the size of a tensor space.
        In the case of a two directional light channel will be a list of two values.

    bond_max: int
        Maximum MPS bond dimension (chi) to use for truncation.

    gamma_l, gamma_r: float
        Coupling (decay) rates to the left and right channels respectively.
        (may be removed from this class in future versions)

    gamma_l2, gamma_r2: float, default: 0
        Optional second set of coupling rates (e.g. for a second TLS).
        Default 0 means only one system.
        (may be removed from this class in future versions)

    tau: float, default: 0
        Delay time if modelling non-Markovian dynamics.

    phase: float, defulat: 0
        Relative delayed phase.
        (may be removed from this class in future versions)
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
    """Bin data used for analysing time-dependent quantities.
    
    Parameters
    ----------
    system_states: list
       List of system bins used when calculating single time system observables.
    
    output_field_states: list
       Time bins used when calculating single time field observables.
    
    input_field_states: list
        List of input time bins used for calculating single time field observables incident the system resulting from
        the defined initial field state.

    correlation_bins: list   
       Correlation bins used when computing output field photon correlation functions at two time points.

    schmidt: list
       Schmidt decomposition system bins usen when calculating entanglement entropy.
 
    loop_field_states: list, default: None
       Tau (delay) bins used when calculating delayed field observables. This is the list of 
       field states entering the feedback loop at each time point.
   
    schmidt_tau: list, default: None
       Schmidt decomposition tau bins usen when calculating delayed entanglement entropy.
   """
    system_states: list     
    output_field_states: list
    input_field_states: list
    correlation_bins: list
    schmidt: list
    loop_field_states: list = None
    schmidt_tau: list = None
