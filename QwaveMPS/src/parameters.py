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
    - bond: int
        Maximum MPS bond dimension (chi) to use for truncation.
    - gamma_l, gamma_r: float
        Coupling (decay) rates to the left and right channels respectively.
    - gamma_l2, gamma_r2: float, optional
        Optional second set of coupling rates (e.g. for a second TLS).
        Default 0 means only one system.
    - tau: float, optional
        Delay time if modelling non-Markovian dynamics.
        Default 0.
    - phase: float, optional
        Relative delayed phase. Default 0.
    """
    delta_t: float
    tmax: float
    d_sys_total: np.ndarray
    d_t_total: np.ndarray
    bond: int
    gamma_l: float
    gamma_r: float
    gamma_l2:float= 0
    gamma_r2:float= 0
    tau: float = 0
    phase: float = 0
    
    
@dataclass
class Bins:
    """Bin metadata used for analysing time-dependent quantities.
   - sys_b: list
       List of system bins used when calculating system observables.
   - time_b: list
       Time bins used when calculating field observables.
   - cor_b: list
       Correlation bins used when computing photon correlation functions.
   - schmidt: list
       Schmidt decomposition system bins usen when calculating entanglement entropy.
   - tau_b: list, optional
       Tau (delay) bins used when calculating delayed field observables.
   - schmidt_tau: list, optional
       Schmidt decomposition tau bins usen when calculating delayed entanglement entropy.
   """
    sys_b: list     
    time_b: list
    cor_b: list
    schmidt: list
    tau_b: list = None
    schmidt_tau: list = None
    
@dataclass
class Pop1TLS:
    """Population data for a single TLS with two channels.

    - pop: np.ndarray
        Population array for the TLS.
    - tbins_r, tbins_l: np.ndarray
        Time bin arrays for right/left channels.
    - int_n_r, int_n_l: np.ndarray
        Time-dependent integrated photon flux in the right/left channels.
    - total: np.ndarray
        Total population count used for normalization or checks.
    """
    pop: np.ndarray
    tbins_r: np.ndarray
    tbins_l: np.ndarray
    int_n_r: np.ndarray
    int_n_l: np.ndarray
    total: np.ndarray
    
@dataclass
class Pop1Channel:
    """Population data for a single TLS with one output channel.

    - pop: np.ndarray
        Population array for the TLS.
    - tbins: np.ndarray
        Time bin array. 
    - trans: np.ndarray
        Integrated photon flux going out of the loop  
    - loop: np.ndarray
        Integrated count of photons in the loop
    - total: np.ndarray
        Total population count used for normalization or checks.
    """
    pop: np.ndarray
    tbins: np.ndarray
    trans: np.ndarray
    loop: np.ndarray
    total: np.ndarray
    
@dataclass
class Pop2TLS:
    """Population data for 2 TLSs with two channels.

    - pop1, pop2: np.ndarray
        Populations for TLS 1 and TLS 2.
    - tbins_r1, tbins_l1, tbins_r2, tbins_l2: np.ndarray
        Time bin arrays for right/left channels going out of TLS 1 and TLS 2 respectively.
    - int_n_r, int_n_l: np.ndarray
        Time-dependent integrated photon flux in the right/left channels.
    - in_r, in_l: np.ndarray
        Integrated count of photons between the TLSs in the right/left channels.
    - total: np.ndarray
        Total population count used for normalization or checks.
    """
    pop1: np.ndarray
    pop2: np.ndarray
    tbins_r1: np.ndarray
    tbins_l1: np.ndarray
    tbins_r2: np.ndarray
    tbins_l2: np.ndarray
    int_n_r: np.ndarray
    int_n_l: np.ndarray
    in_r: np.ndarray
    in_l: np.ndarray
    total: np.ndarray

@dataclass
class G1Correl:
    """First-order (G1) correlation matrices for different channel pairings.

    - g1_rr_matrix: np.ndarray    
        G1 correlation for right-right detections
    - g1_ll_matrix: np.ndarray
        G1 correlation for left-left detections
    - g1_rl_matrix: np.ndarray
        G1 correlation right-left
    - g1_lr_matrix: np.ndarray
        G1 correlation left-right
    """
    g1_rr_matrix: np.ndarray
    g1_ll_matrix: np.ndarray
    g1_rl_matrix: np.ndarray
    g1_lr_matrix: np.ndarray

@dataclass
class G2Correl:
    """Second-order (G2) correlation matrices for different channel pairings.

   - g2_rr_matrix: np.ndarray
       G2 correlation for right-right detections
   - g2_ll_matrix: np.ndarray
       G2 correlation for left-left detections
   - g2_rl_matrix: np.ndarray
       G2 correlation right-left
   - g2_lr_matrix: np.ndarray
       G2 correlation left-right  
   """
    g2_rr_matrix: np.ndarray
    g2_ll_matrix: np.ndarray
    g2_rl_matrix: np.ndarray
    g2_lr_matrix: np.ndarray

@dataclass
class SSCorrel:
    """Steady state correlation lists for two-channel solutions.

    - t_cor: list
        List of correlation times when steady state reached
    - g1_listl, g1_listr: list 
        Normalized g1 lists for left and right channel respectively
    - g2_listl, g2_listr: list  
        Normalized g2 lists for left and right channel respectively
    - c1_l, c1_r: list
        Not normalized G1 lists for left and right channel respectively
    - c2_l, c2_r: list  
        Not normalized G2 lists for left and right channel respectively
    """
    t_cor: list     
    g1_listl: list     
    g1_listr: list     
    g2_listl: list     
    g2_listr: list     
    c1_l: list     
    c1_r: list     
    c2_l: list     
    c2_r: list     
    
@dataclass
class SSCorrel1Channel:
    """Steady state correlation lists for a single channel solution.

    - t_cor: list
        List of correlation times when steady state reached
    - g1_list: list 
        Normalized g1 list
    - g2_list: list  
        Normalized g2 list
    - c1: list
        Not normalized G1 list
    - c2: list  
        Not normalized G2 list
    """
    t_cor: list     
    g1_list: list     
    g2_list: list      
    c1: list       
    c2: list         
    

    
    