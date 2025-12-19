#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 09:35:04 2025

@author: sofia
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class InputParams:
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
    sys_b: list     
    time_b: list
    cor_b: list
    schmidt: list
    tau_b: list = None
    schmidt_tau: list = None
    
@dataclass
class Pop1TLS:
    pop: np.ndarray
    tbins_r: np.ndarray
    tbins_l: np.ndarray
    int_n_r: np.ndarray
    int_n_l: np.ndarray
    total: np.ndarray
    
@dataclass
class Pop1Channel:
    pop: np.ndarray
    tbins: np.ndarray
    trans: np.ndarray
    loop: np.ndarray
    total: np.ndarray
    
@dataclass
class Pop2TLS:
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
    g1_rr_matrix: np.ndarray
    g1_ll_matrix: np.ndarray
    g1_rl_matrix: np.ndarray
    g1_lr_matrix: np.ndarray

@dataclass
class G2Correl:
    g2_rr_matrix: np.ndarray
    g2_ll_matrix: np.ndarray
    g2_rl_matrix: np.ndarray
    g2_lr_matrix: np.ndarray

@dataclass
class SSCorrel:
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
    t_cor: list     
    g1_list: list     
    g2_list: list      
    c1: list       
    c2: list         
    

    
    