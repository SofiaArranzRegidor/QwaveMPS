#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the simulations to evolve the systems 
and calculate the main observables.

Irovides time-evolution routines (Markovian and non-Markovian) for systems
coupled to a 1D field, together with observable
calculations (populations, correlations, spectra and entanglement).

It requires the module ncon (pip install --user ncon)

"""


import numpy as np
import copy
from ncon import ncon
from scipy.linalg import svd,norm
from .operators import * 
from . import states as states
from collections.abc import Iterator
from QwaveMPS.src.parameters import *
from typing import Callable, TypeAlias

Hamiltonian: TypeAlias = np.ndarray | Callable[[int], np.ndarray]

# -----------------------------------
# Singular Value Decomposition helper
# -----------------------------------

def _svd_tensors(tensor:np.ndarray, bond:int, d_1:int, d_2:int) -> np.ndarray:
    """
    Perform a SVD, reshape the tensors and return left tensor, 
    normalized Schmidt vector, and right tensor.

    Parameters
    ----------
    tensor : ndarray
        tensor to decompose
    
    bond : int
        max. bond dimension
    
    d_1 : int
        physical dimension of first tensor
    
    d_2 : int
        physical dimension of second tensor

    Returns
    -------
    u : ndarray
        left normalized tensor

    s_norm : ndarray
        smichdt coefficients normalized 
    
    vt : ndarray
        transposed right normalized tensor
    """
    u, s, vt = svd(tensor.reshape(tensor.shape[0]*d_1, tensor.shape[-1]*d_2), full_matrices=False)
    chi = min(bond, len(s))
    epsilon = 1e-12 #to avoid dividing by zero
    s_norm = s[:chi] / (norm(s[:chi])+ epsilon)
    u = u[:, :chi].reshape(tensor.shape[0],d_1,chi)
    vt = vt[:chi, :].reshape(chi,d_2,tensor.shape[-1])
    return u, s_norm, vt

# ------------------------------------------------------
# Time evolution: Markovian and non-Markovian evolutions
# ------------------------------------------------------

def t_evol_mar(ham:Hamiltonian, i_s0:np.ndarray, i_n0:np.ndarray, params:InputParams) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """ 
    Time evolution of the system without delay times (Markovian regime)
    
    Parameters
    ----------
    ham : ndarray or callable
        Either a fixed evolution operator/tensor or a callable returning the
        evolution operator for time-step k: ham(k).
    
    i_s0 : ndarray
        Initial system bin (tensor).
        
    i_n0: ndarray 
        Initial field bin.
        Seed for the input time-bin generator.
        
    params:InputParams
        Class containing the input parameters
        (contains delta_t, tmax, bond, d_t_total, d_sys_total).

    Returns
    -------
    Bins:  Dataclass (from parameters.py) 
        containing:
            - sys_b: list of system bins
            - time_b: list of time bins
            - cor_b: list of tensors used for correlations
            - schmidt: list of Schmidt coefficient arrays (for entanglement calculation)
    """
    
    delta_t = params.delta_t
    tmax=params.tmax
    bond=params.bond
    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total

    
    d_t=np.prod(d_t_total)
    d_sys=np.prod(d_sys_total)
    n=int(tmax/delta_t)
    t_k=0
    i_s=i_s0
    sbins=[] 
    sbins.append(i_s0)
    tbins=[]
    tbins.append(states.i_ng(d_t))
    schmidt=[]
    schmidt.append(np.zeros(1))
    if not callable(ham):
        evol=u_evol(ham,d_sys,d_t)
    swap_sys_t=swap(d_sys,d_t)
    input_field=states.input_state_generator(d_t_total, i_n0)
    cor_list=[]
    for k in range(n):   
        i_nk = next(input_field)   
        if callable(ham):
            evol=u_evol(ham(k),d_sys,d_t)
        phi1=ncon([i_s,i_nk,evol],[[-1,2,3],[3,4,-4],[-2,-3,2,4]]) #system bin, time bin + u operator contraction  
        i_s,stemp,i_n=_svd_tensors(phi1, bond,d_sys,d_t)
        i_s=i_s*stemp[None,None,:] #OC system bin
        sbins.append(i_s)
        tbins.append(stemp[:,None,None]*i_n)
                    
        phi2=ncon([i_s,i_n,swap_sys_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #system bin, time bin + swap contraction
        i_n,stemp,i_st=_svd_tensors(phi2, bond,d_t,d_sys)
        i_s=stemp[:,None,None]*i_st   #OC system bin
        t_k += delta_t
        
        schmidt.append(stemp)
        
        if k < (n-1):
            cor_list.append(i_n)
        if k == n-1:
            cor_list.append(ncon([i_n,np.diag(stemp)],[[-1,-2,1],[1,-3]]))
        
    return Bins(sys_b=sbins,time_b=tbins,cor_b=cor_list,schmidt=schmidt)

def t_evol_nmar(ham:Hamiltonian, i_s0:np.ndarray, i_n0:np.ndarray,params:InputParams) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """ 
    Time evolution of the system with finite delays/feedback (non-Markovian regime)
    
    Parameters
    ----------
    ham : ndarray or callable
        Either a fixed evolution operator/tensor or a callable returning the
        evolution operator for time-step k: ham(k).
        
     i_s0 : ndarray
         Initial system bin (tensor).
         
     i_n0: ndarray 
         Initial field bin.
         Seed for the input time-bin generator.

     params:InputParams
         Class containing the input parameters
         (contains delta_t, tmax, bond, d_t_total, d_sys_total, tau.).

    Returns
    -------
    Bins:  Dataclass (from parameters.py) 
        containing:
          - sys_b: list of system bins
          - time_b: list of time bins
          - tau_b: list of feedback bins 
          - cor_b: list of tensors used for correlations
          - schmidt, schmidt_tau: lists of Schmidt coefficient arrays
    """
    delta_t = params.delta_t
    tmax=params.tmax
    bond=params.bond
    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total
    tau=params.tau
    
    d_t=np.prod(d_t_total)
    d_sys=np.prod(d_sys_total)
    sbins=[] 
    tbins=[]
    taubins=[]
    nbins=[]
    cor_list=[]
    schmidt=[]
    schmidt_tau=[]
    sbins.append(i_s0)   
    tbins.append(states.i_ng(d_t))
    taubins.append(states.i_ng(d_t))
    schmidt.append(np.zeros(1))
    schmidt_tau.append(np.zeros(1))
    input_field=states.input_state_generator(d_t_total, i_n0)
    n=int(round(tmax/delta_t,0))
    t_k=0
    t_0=0
    if not callable(ham):
        evol=u_evol(ham,d_sys,d_t,2) #Feedback loop means time evolution involves an input and a feedback time bin. Can generalize this later, leaving 2 for now so it runs.
    swap_t_t=swap(d_t,d_t)
    swap_sys_t=swap(d_sys,d_t)
    l=int(round(tau/delta_t,0)) #time steps between system and feedback
    
    for i in range(l):
        nbins.append(states.i_ng(d_t))
        t_0+=delta_t
    
    i_stemp=i_s0      
    
    for k in range(n):   
        #swap of the feedback until being next to the system
        i_tau= nbins[k] #starting from the feedback bin
        for i in range(k,k+l-1): 
            i_n=nbins[i+1] 
            swaps=ncon([i_tau,i_n,swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) 
            i_n2,stemp,i_t=_svd_tensors(swaps,bond,d_t,d_t)
            i_tau = ncon([np.diag(stemp),i_t],[[-1,1],[1,-3,-4]]) 
            nbins[i]=i_n2 
            
        #Make the system bin the OC
        i_1=ncon([i_tau,i_stemp],[[-1,-2,1],[1,-3,-4]]) #feedback-system contraction
        i_t,stemp,i_stemp=_svd_tensors(i_1, bond,d_t,d_sys)
        i_s=stemp[:,None,None]*i_stemp #OC system bin
        
        #now contract the 3 bins and apply u, followed by 2 svd to recover the 3 bins 
        i_nk = next(input_field)                
        if callable(ham):
            evol=u_evol(ham(k),d_sys,d_t, 2)
        phi1=ncon([i_t,i_s,i_nk,evol],[[-1,3,1],[1,4,2],[2,5,-5],[-2,-3,-4,3,4,5]]) #tau bin, system bin, future time bin + u operator contraction
        i_t,stemp,i_2=_svd_tensors(phi1, bond,d_t,d_t*d_sys)
        i_2=stemp[:,None,None]*i_2
        i_stemp,stemp,i_n=_svd_tensors(i_2, bond,d_sys,d_t)
        i_s = i_stemp*stemp[None,None,:]
        sbins.append(i_s) 
        
        #swap system and i_n
        phi2=ncon([i_s,i_n,swap_sys_t],[[-1,3,2],[2,4,-4],[-2,-3,3,4]]) #system bin, time bin + swap contraction
        i_n,stemp,i_stemp=_svd_tensors(phi2, bond,d_sys,d_t)   
        i_n=i_n*stemp[None,None,:] #the OC in time bin     
        
        cont= ncon([i_t,i_n],[[-1,-2,1],[1,-3,-4]]) 
        i_t,stemp,i_n=_svd_tensors(cont, bond,d_t,d_t)   
        i_tau = i_t*stemp[None,None,:] #OC in feedback bin     
        tbins.append(stemp[:,None,None]*i_n)
        
        #feedback bin, time bin contraction
        taubins.append(i_tau) 
        nbins[k+l-1]=i_tau #update of the feedback bin
        nbins.append(i_n)         
        t_k += delta_t
        schmidt.append(stemp) #storing the Schmidt coeff for calculating the entanglement

        #swap back of the feedback bin      
        for i in range(k+l-1,k,-1): #goes from the last time bin to first one
            i_n=nbins[i-1] #time bin
            swaps=ncon([i_n,i_tau,swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #time bin, feedback bin + swap contraction
            i_t,stemp,i_n2=_svd_tensors(swaps, bond,d_t,d_t)   
            i_tau = i_t*stemp[None,None,:] #OC tau bin         
            nbins[i]=i_n2    #update nbins  
        schmidt_tau.append(stemp)  
        if k<(n-1):         
            nbins[k+1] = stemp[:,None,None]*i_n2 #new tau bin for the next time step       
            cor_list.append(i_t)
        if k == n-1:
            cor_list.append(i_t*stemp[None,None,:])   
            
    return Bins(sys_b=sbins,time_b=tbins,tau_b=taubins,cor_b=cor_list,schmidt=schmidt,schmidt_tau=schmidt_tau)

# ---------------------------------------------------------------
# Observables: populations, entanglement, spectrum
# ---------------------------------------------------------------


def pop_dynamics(bins:Bins, params:InputParams) -> Pop1TLS:
    """
    Calculates the main population dynamics for a single TLS in an infinite waveguide

    Parameters
    ----------
    bins : Bins
        Bins returned by t_evol_mar
    
    params : InputParams
        Simulation parameters (contains delta_t, d_t_total, d_sys_total).

    Returns
    -------
    Pop1TLS: Dataclass
         containing:
            - pop: TLS population
            - tbins_r: right-moving photon flux per time bin
            - tbins_l: left-moving photon flux per time bin
            - int_n_r/int_n_l: integrated right/left flux
            - total: total excitations (populations + integrated flux) at each time
    """
    delta_t = params.delta_t
    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total
    sbins=bins.sys_b
    tbins=bins.time_b
    
    d_sys=np.prod(d_sys_total)
    pop=np.array([expectation(s, tls_pop(d_sys)) for s in sbins])
    tbinsR=np.array([expectation(t, a_r_pop(delta_t,d_t_total)) for t in tbins])
    tbinsL=np.array([expectation(t, a_l_pop(delta_t,d_t_total)) for t in tbins])
   
    # Cumulative sums
    trans = np.cumsum(tbinsR)
    ref = np.cumsum(tbinsL)
    total = trans + ref + pop
        
    return Pop1TLS(pop=pop,tbins_r=tbinsR,tbins_l=tbinsL,int_n_r=trans,int_n_l=ref,total=total)

def pop_dynamics_1tls_nmar(bins:Bins, params:InputParams) -> Pop1Channel:
    """
    Calculates the main population dynamics for a single TLS in a semi-infinite waveguide,
    with non-Markovian feedback (one channel).

    Parameters
    ----------
    bins : Bins
        Bins returned by t_evol_nmar
    
    params : InputParams
        Simulation parameters

    Returns
    -------
    Pop1Channel: Dataclass
         containing:
            - pop: TLS population
            - tbins: photon flux per time bin
            - trans: integrated transmitted flux
            - loop: feedback-loop photon count
            - total: total excitations at each time
    """
    
    delta_t = params.delta_t
    d_sys_total = params.d_sys_total
    tau = params.tau
    sbins=bins.sys_b
    tbins=bins.time_b
    taubins=bins.tau_b
    
    n=len(sbins) 
    d_sys=np.prod(d_sys_total)
    pop=np.array([expectation(s, tls_pop(d_sys)) for s in sbins])
    tbins=np.array([expectation(t, a_pop(delta_t)) for t in tbins])
    tbins2=np.real([expectation(taus, a_pop(delta_t)) for taus in taubins])
    ph_loop=np.zeros(n,dtype=complex)
    trans=np.zeros(n,dtype=complex)
    total=np.zeros(n,dtype=complex)
    
    l=int(round(tau/delta_t,0))
    temp_out=0
    in_loop=0
    for i in range(n):
        temp_out+=tbins2[i]
        trans[i]=temp_out
        if i<=l:
            in_loop=np.sum(tbins[:i+1]) 
            ph_loop[i]=in_loop
            total[i]=pop[i]+ph_loop[i]+trans[i]
        if i>l:
            in_loop=np.sum(tbins[i-l+1:i+1]) 
            ph_loop[i]=in_loop
            total[i]  = pop[i] +  trans[i] + ph_loop[i]
    return Pop1Channel(pop=pop,tbins=tbins,trans=trans,loop=ph_loop,total=total)

def pop_dynamics_2tls(bins:Bins,params:InputParams) -> Pop2TLS:
    """
    Calculates the main population dynamics for 2 TLSs in an infinite waveguide

    Parameters
    ----------
    bins : Bins
        Bins returned by t_evol_mar
    
    params : InputParams
        Simulation parameters

    Returns
    -------
    Pop2TLS: Dataclass
         containing:
            - pop1/pop2: populations of TLS1 and TLS2
            - tbins_r/tbins_l:  right/left fluxes per time bin
            - tbins_r2/tbins_l2: feedback-line fluxes (when tau != 0)
            - int_n_r/int_n_l: integrated right/left flux
            - in_r/in_l : in-loop flux (for feedback)
            - total: total excitations at each time
    """
    
    sbins=bins.sys_b
    tbins=bins.time_b
    taubins=bins.tau_b
    delta_t,d_sys_total,d_t_total,tau = params.delta_t,params.d_sys_total,params.d_t_total,params.tau 
    
    n=len(sbins)
    d_sys1=d_sys_total[0]
    d_sys2=d_sys_total[1]

    pop1=np.zeros(n,dtype=complex)
    pop2=np.zeros(n,dtype=complex)
    tbins_r=np.zeros(n,dtype=complex)
    tbins_l=np.zeros(n,dtype=complex)
    tbins_r2=np.zeros(n,dtype=complex)
    tbins_l2=np.zeros(n,dtype=complex)
    in_r=np.zeros(n,dtype=complex)
    in_l=np.zeros(n,dtype=complex)
    int_n_r=np.zeros(n,dtype=complex)
    int_n_l=np.zeros(n,dtype=complex)
    total=np.zeros(n,dtype=complex)
    temp_in_r=0
    temp_in_l=0
    temp_outR=0
    temp_outL=0
    l=int(round(tau/delta_t,0))
    temp_trans=0
    temp_ref=0
    for i in range(n):
        i_s=sbins[i]
        i_sm=i_s.reshape(i_s.shape[0]*2,i_s.shape[-1]*2)
        u,sm,vt=svd(i_sm,full_matrices=False) #SVD
        i_s1 = u[:,range(len(sm))].reshape(i_s.shape[0],2,len(sm))  
        i_s1 = ncon([i_s1,np.diag(sm)],[[-1,-2,1],[1,-3]]) 
        i_s2 = vt[range(len(sm)),:].reshape(len(sm),2,i_s.shape[-1]) 
        i_s2 = ncon([np.diag(sm),i_s2],[[-1,1],[1,-2,-3]]) 
        pop1[i]=expectation(i_s1, tls_pop(d_sys1))
        pop2[i]=expectation(i_s2, tls_pop(d_sys2))    
        tbins_r[i]=np.real(expectation(tbins[i], a_r_pop(delta_t,d_t_total)))
        tbins_l[i]=np.real(expectation(tbins[i], a_l_pop(delta_t,d_t_total)))
        if tau != 0:
            tbins_r2[i]=np.real(expectation(taubins[i], a_r_pop(delta_t,d_t_total)))
            tbins_l2[i]=np.real(expectation(taubins[i], a_l_pop(delta_t,d_t_total)))
            temp_outR+=tbins_r2[i]
            temp_outL+=tbins_l2[i]
            int_n_r[i]=temp_outR
            int_n_l[i]=temp_outL
            if i <=l:
                temp_in_r+=expectation(tbins[i], a_r_pop(delta_t,d_t_total))
                in_r[i] = temp_in_r
                temp_in_l+= expectation(tbins[i], a_l_pop(delta_t,d_t_total))
                in_l[i] = temp_in_l
                total[i]  = pop1[i] + pop2[i]  + in_r[i] + in_l[i]  + int_n_r[i] + int_n_l[i]
            if i>l:
                temp_in_r=np.sum(tbins_r[i-l+1:i+1]) 
                temp_in_l=np.sum(tbins_l[i-l+1:i+1])
                total[i]  = pop1[i] + pop2[i]  + temp_in_r + temp_in_l + int_n_r[i] + int_n_l[i]
                in_r[i] = temp_in_r
                in_l[i] = temp_in_l
        if tau==0:
            temp_trans+= tbins_r[i]
            int_n_r[i] = temp_trans
            temp_ref += tbins_l[i]
            int_n_l[i] = temp_ref
            total[i]  = pop1[i] + pop2[i]  + int_n_r[i] + int_n_l[i]
        
    return Pop2TLS(pop1,pop2,tbins_r,tbins_l,tbins_r2,tbins_l2,int_n_r,int_n_l,in_r,in_l,total)

def entanglement(sch:list[np.ndarray]) -> list[float]:
    """
    Compute von Neumann entanglement entropy across a list of Schmidt coefficient arrays.

    Parameters
    ----------
    sch : list[np.ndarray]
        List of Schmidt coefficient arrays (s) for each bipartition.

    Returns
    -------
    list[float]
        Entanglement entropies computed as -sum(p * log2 p) where p = s**2.
    """
    ent_list=[]
    for s in sch:
        a=s**2   
        a=np.trim_zeros(a) 
        b=np.log2(a)
        c=a*b
        ent=-sum(c)
        ent_list.append(ent)
    return ent_list

def spectrum_w(delta_t:float, g1_list: np.ndarray) -> [np.ndarray, np.ndarray]:
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

# ----------------------
# Correlation functions
# ----------------------

def first_order_correlation(bins:Bins, params:InputParams,single_channel=False) -> np.ndarray|G1Correl:
    """
    Calculates the first order correlation function g1(t,t+tau) for the outgoing field.

    This routine builds a matrix of complex values with indices [t, tau] where tau >= 0.
    
    Parameters
    ----------
    bins : Bins
        Bins returned by time evolution functions
        cor_b field must contain the correlation tensors.
    
    params : InputParams
        Simulation parameters 
    
    single_channel : bool
       if True compute correlations for a single channel; 
       if False compute left/right channel cross-correlations separately 
       and return a G1Correl dataclass.
       
    Returns
    -------
    np.ndarray or G1Correl
        If single_channel True: returns a 2D g1 matrix (complex).
        Otherwise returns a G1Correl dataclass with separate matrices for rr, ll, rl, lr.
    """
    
    import time as t
    
    d_t_total=params.d_t_total
    bond=params.bond
    delta_t=params.delta_t
    cor_list1 = bins.cor_b
    
    cor_list2 =  cor_list1
    d_t=np.prod(d_t_total)
    swap_t_t=swap(d_t,d_t)
    start_time_c = t.time()
    
    if single_channel==True:
        g1_matrix= np.zeros((len(cor_list1),len(cor_list1)),dtype='complex') 

        #Bring the OC to the first bin
        for i in range(len(cor_list2)-1,0,-1):
            # print('iteration', i)
            two_cor=ncon([cor_list2[i-1],cor_list2[i]],[[-1,-2,1],[1,-3,-4]])
            cor_l1,stemp,cor_l2=_svd_tensors(two_cor, bond,d_t,d_t)
            cor_list2[i]=cor_l2
            cor_list2[i-1]= ncon([cor_l1,np.diag(stemp)],[[-1,-2,1],[1,-3]])
        
        for j in range(len(cor_list1)-1):            
            
            i_1=cor_list2[0]
            i_2=cor_list2[1]     
            
            g1_matrix[0,j] = expectation(i_1,(delta_b_dag(delta_t, d_t_total) @ delta_b(delta_t, d_t_total)))/(delta_t**2)    
            
            for i in range(len(cor_list2)-1):  
                # print('iteration', i)
                state=ncon([i_1,i_2],[[-1,-2,1],[1,-3,-4]]) 
                
                g1_matrix[i+1,j]=expectation_2(state, g1_rr(delta_t,d_t_total))/(delta_t**2) 
           
                swaps=ncon([i_1,i_2,swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #swapping the feedback bin to the left so it is next to the next bin
                i_t2,stemp,i_t1=_svd_tensors(swaps,bond,d_t,d_t)
                i_1 = ncon([np.diag(stemp),i_t1],[[-1,1],[1,-2,-3]])
                
                if i < (len(cor_list2)-2):                
                    i_2=cor_list2[i+2] #next time bin for the next correlation
                    cor_list2[i]=i_t2 #update of the increasing bin
                if i == len(cor_list2)-2:
                    cor_list2[i]=i_t2
                    cor_list2[i+1]=i_1
                    
            for i in range(len(cor_list2)-1,0,-1):            
                two_cor=ncon([cor_list2[i-1],cor_list2[i],swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]])
                cor_l,stemp,cor_l2=_svd_tensors(two_cor,bond,d_t,d_t)        
                if i>1:
                    cor_list2[i] = cor_l2  
                    cor_list2[i-1]= ncon([cor_l,np.diag(stemp)],[[-1,-2,1],[1,-3]]) #OC on left bin
                if i == 1:
                   cor_l = cor_l2
                   cor_list2[i] = ncon([np.diag(stemp),cor_l],[[-1,1],[1,-2,-3]]) 
            cor_list2=cor_list2[1:]   
        t_c=t.time() - start_time_c    
        print("--- %s seconds correlation---" %(t_c)) 
        return g1_matrix         
    else:
        g1_rr_matrix= np.zeros((len(cor_list1),len(cor_list1)),dtype='complex') 
        g1_ll_matrix= np.zeros((len(cor_list1),len(cor_list1)),dtype='complex') 
        g1_rl_matrix= np.zeros((len(cor_list1),len(cor_list1)),dtype='complex') 
        g1_lr_matrix= np.zeros((len(cor_list1),len(cor_list1)),dtype='complex') 
    
    
        #Bring the OC to the first bin
        for i in range(len(cor_list2)-1,0,-1):
            # print('iteration', i)
            two_cor=ncon([cor_list2[i-1],cor_list2[i]],[[-1,-2,1],[1,-3,-4]])
            cor_l1,stemp,cor_l2=_svd_tensors(two_cor, bond,d_t,d_t)
            cor_list2[i]=cor_l2
            cor_list2[i-1]= ncon([cor_l1,np.diag(stemp)],[[-1,-2,1],[1,-3]])
        
        for j in range(len(cor_list1)-1):            
            
            i_1=cor_list2[0]
            i_2=cor_list2[1]     
            
            g1_rr_matrix[0,j] = expectation(i_1,(delta_b_dag_r(delta_t, d_t_total) @ delta_b_r(delta_t, d_t_total)))/(delta_t**2)    
            g1_ll_matrix[0,j] = expectation(i_1, (delta_b_dag_l(delta_t, d_t_total) @ delta_b_l(delta_t, d_t_total)))/(delta_t**2)  
            g1_rl_matrix[0,j] = expectation(i_1,(delta_b_dag_r(delta_t, d_t_total) @ delta_b_l(delta_t, d_t_total)))/(delta_t**2) 
            g1_lr_matrix[0,j] = expectation(i_1, (delta_b_dag_l(delta_t, d_t_total) @ delta_b_r(delta_t, d_t_total)))/(delta_t**2) 
            
            for i in range(len(cor_list2)-1):  
                # print('iteration', i)
                state=ncon([i_1,i_2],[[-1,-2,1],[1,-3,-4]]) 
                
                g1_rr_matrix[i+1,j]=expectation_2(state, g1_rr(delta_t,d_t_total))/(delta_t**2) 
                g1_ll_matrix[i+1,j]=expectation_2(state, g1_ll(delta_t,d_t_total))/(delta_t**2) 
                g1_rl_matrix[i+1,j]=expectation_2(state, g1_rl(delta_t,d_t_total))/(delta_t**2) 
                g1_lr_matrix[i+1,j]=expectation_2(state, g1_lr(delta_t,d_t_total))/(delta_t**2) 
                
                swaps=ncon([i_1,i_2,swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #swapping the feedback bin to the left so it is next to the next bin
                i_t2,stemp,i_t1=_svd_tensors(swaps,bond,d_t,d_t)
                i_1 = ncon([np.diag(stemp),i_t1],[[-1,1],[1,-2,-3]])
                
                if i < (len(cor_list2)-2):                
                    i_2=cor_list2[i+2] #next time bin for the next correlation
                    cor_list2[i]=i_t2 #update of the increasing bin
                if i == len(cor_list2)-2:
                    cor_list2[i]=i_t2
                    cor_list2[i+1]=i_1
                    
            for i in range(len(cor_list2)-1,0,-1):            
                two_cor=ncon([cor_list2[i-1],cor_list2[i],swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]])
                cor_l,stemp,cor_l2=_svd_tensors(two_cor,bond,d_t,d_t)        
                if i>1:
                    cor_list2[i] = cor_l2  
                    cor_list2[i-1]= ncon([cor_l,np.diag(stemp)],[[-1,-2,1],[1,-3]]) #OC on left bin
                if i == 1:
                   cor_l = cor_l2
                   cor_list2[i] = ncon([np.diag(stemp),cor_l],[[-1,1],[1,-2,-3]]) 
            cor_list2=cor_list2[1:]   
        t_c=t.time() - start_time_c    
        print("--- %s seconds correlation---" %(t_c)) 
        return G1Correl(g1_rr_matrix,g1_ll_matrix,g1_rl_matrix,g1_lr_matrix)    

def second_order_correlation(bins:Bins,params:InputParams,single_channel=False) -> np.ndarray|G2Correl:
    """
    Calculates the second order correlation function g2(t,t+tau) for the outgoing field.

    This routine builds a matrix of complex values with indices [t, tau] where tau >= 0.
    
    Parameters
    ----------
    bins : Bins
        Bins returned by time evolution functions
        cor_b field must contain the correlation tensors.
    
    params : InputParams
        Simulation parameters 
    
    single_channel : bool
       if True compute correlations for a single channel; 
       if False compute left/right channel cross-correlations separately 
       and return a G2Correl dataclass.
       
    Returns
    -------
    np.ndarray or G2Correl
        If single_channel True: returns a 2D g2 matrix (complex).
        Otherwise returns a G2Correl dataclass with separate matrices for rr, ll, rl, lr.
    """
    
    import time as t
    
    d_t_total=params.d_t_total
    bond=params.bond
    delta_t=params.delta_t
    cor_list1 = bins.cor_b
    
    start_time_c = t.time()
    cor_list2 =  cor_list1
    d_t=np.prod(d_t_total)
    swap_t_t=swap(d_t,d_t)
    
    if single_channel==True:
        g2_rr_matrix= np.zeros((len(cor_list1),len(cor_list1)),dtype='complex') 

        
        #Bring the OC to the first bin
        for i in range(len(cor_list2)-1,0,-1):
            # print('iteration', i)
            two_cor=ncon([cor_list2[i-1],cor_list2[i]],[[-1,-2,1],[1,-3,-4]])
            cor_l1,stemp,cor_l2=_svd_tensors(two_cor, bond,d_t,d_t)
            cor_list2[i]=cor_l2
            cor_list2[i-1]= ncon([cor_l1,np.diag(stemp)],[[-1,-2,1],[1,-3]])
        
        for j in range(len(cor_list1)-1):            
            
            i_1=cor_list2[0]
            i_2=cor_list2[1]     
                        
            g2_rr_matrix[0,j] = expectation(i_1,(delta_b_dag(delta_t, d_t_total) @ delta_b_dag(delta_t, d_t_total) 
                                                 @ delta_b(delta_t, d_t_total) @ delta_b(delta_t, d_t_total)))/(delta_t**4)    
            for i in range(len(cor_list2)-1):  
                state=ncon([i_1,i_2],[[-1,-2,1],[1,-3,-4]]) 
                
                g2_rr_matrix[i+1,j]=expectation_2(state, g2(delta_t,d_t_total))/(delta_t**4) 
    
                swaps=ncon([i_1,i_2,swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #swapping the feedback bin to the left so it is next to the next bin
                i_t2,stemp,i_t1=_svd_tensors(swaps,bond,d_t,d_t)
                i_1 = ncon([np.diag(stemp),i_t1],[[-1,1],[1,-2,-3]])
                
                if i < (len(cor_list2)-2):                
                    i_2=cor_list2[i+2] #next time bin for the next correlation
                    cor_list2[i]=i_t2 #update of the increasing bin
                if i == len(cor_list2)-2:
                    cor_list2[i]=i_t2
                    cor_list2[i+1]=i_1
                    
            for i in range(len(cor_list2)-1,0,-1):            
                two_cor=ncon([cor_list2[i-1],cor_list2[i],swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]])
                cor_l,stemp,cor_l2=_svd_tensors(two_cor,bond,d_t,d_t)        
                if i>1:
                    cor_list2[i] = cor_l2  
                    cor_list2[i-1]= ncon([cor_l,np.diag(stemp)],[[-1,-2,1],[1,-3]]) #OC on left bin
                if i == 1:
                   cor_l = cor_l2
                   cor_list2[i] = ncon([np.diag(stemp),cor_l],[[-1,1],[1,-2,-3]]) 
            cor_list2=cor_list2[1:]   
        return g2_rr_matrix 
    else:    
        g2_rr_matrix= np.zeros((len(cor_list1),len(cor_list1)),dtype='complex') 
        g2_ll_matrix= np.zeros((len(cor_list1),len(cor_list1)),dtype='complex') 
        g2_rl_matrix= np.zeros((len(cor_list1),len(cor_list1)),dtype='complex') 
        g2_lr_matrix= np.zeros((len(cor_list1),len(cor_list1)),dtype='complex') 
        
        #Bring the OC to the first bin
        for i in range(len(cor_list2)-1,0,-1):
            two_cor=ncon([cor_list2[i-1],cor_list2[i]],[[-1,-2,1],[1,-3,-4]])
            cor_l1,stemp,cor_l2=_svd_tensors(two_cor, bond,d_t,d_t)
            cor_list2[i]=cor_l2
            cor_list2[i-1]= ncon([cor_l1,np.diag(stemp)],[[-1,-2,1],[1,-3]])
        
        for j in range(len(cor_list1)-1):            
            
            i_1=cor_list2[0]
            i_2=cor_list2[1]     
                        
            g2_rr_matrix[0,j] = expectation(i_1,(delta_b_dag_r(delta_t, d_t_total) @ delta_b_dag_r(delta_t, d_t_total) 
                                                 @ delta_b_r(delta_t, d_t_total) @ delta_b_r(delta_t, d_t_total)))/(delta_t**4)    
            g2_ll_matrix[0,j] = expectation(i_1, (delta_b_dag_l(delta_t, d_t_total) @ delta_b_dag_l(delta_t, d_t_total)
                                                  @ delta_b_l(delta_t, d_t_total) @ delta_b_l(delta_t, d_t_total)))/(delta_t**4)  
            g2_rl_matrix[0,j] = expectation(i_1,(delta_b_dag_r(delta_t, d_t_total) @ delta_b_dag_l(delta_t, d_t_total)
                                                 @ delta_b_l(delta_t, d_t_total) @ delta_b_l(delta_t, d_t_total)))/(delta_t**4) 
            g2_lr_matrix[0,j] = expectation(i_1, (delta_b_dag_l(delta_t, d_t_total) @ delta_b_dag_r(delta_t, d_t_total)
                                                  @ delta_b_r(delta_t, d_t_total) @ delta_b_r(delta_t, d_t_total)))/(delta_t**4) 
            
            for i in range(len(cor_list2)-1):  
                state=ncon([i_1,i_2],[[-1,-2,1],[1,-3,-4]]) 
                
                g2_rr_matrix[i+1,j]=expectation_2(state, g2_rr(delta_t,d_t_total))/(delta_t**4) 
                g2_ll_matrix[i+1,j]=expectation_2(state, g2_ll(delta_t,d_t_total))/(delta_t**4) 
                g2_rl_matrix[i+1,j]=expectation_2(state, g2_rl(delta_t,d_t_total))/(delta_t**4) 
                g2_lr_matrix[i+1,j]=expectation_2(state, g2_lr(delta_t,d_t_total))/(delta_t**4) 
                
                swaps=ncon([i_1,i_2,swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #swapping the feedback bin to the left so it is next to the next bin
                i_t2,stemp,i_t1=_svd_tensors(swaps,bond,d_t,d_t)
                i_1 = ncon([np.diag(stemp),i_t1],[[-1,1],[1,-2,-3]])
                
                if i < (len(cor_list2)-2):                
                    i_2=cor_list2[i+2] #next time bin for the next correlation
                    cor_list2[i]=i_t2 #update of the increasing bin
                if i == len(cor_list2)-2:
                    cor_list2[i]=i_t2
                    cor_list2[i+1]=i_1
                    
            for i in range(len(cor_list2)-1,0,-1):            
                two_cor=ncon([cor_list2[i-1],cor_list2[i],swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]])
                cor_l,stemp,cor_l2=_svd_tensors(two_cor,bond,d_t,d_t)        
                if i>1:
                    cor_list2[i] = cor_l2  
                    cor_list2[i-1]= ncon([cor_l,np.diag(stemp)],[[-1,-2,1],[1,-3]]) #OC on left bin
                if i == 1:
                   cor_l = cor_l2
                   cor_list2[i] = ncon([np.diag(stemp),cor_l],[[-1,1],[1,-2,-3]]) 
            cor_list2=cor_list2[1:]   
        return G2Correl(g2_rr_matrix,g2_ll_matrix,g2_rl_matrix,g2_lr_matrix)   

def two_time_correlations(time_bin_list:list[np.ndarray], ops_same_time:list[np.ndarray], ops_two_time:list[np.ndarray], params:InputParams, oc_end_list_flag:bool=True, completion_print_flag:bool=True) -> list[np.ndarray]:
    """ 
    General two-time correlation calculator.
    Take in list of time ordered normalized (with OC) time bins at position of relevance.
    Calculate a list of arbitrary two time point correlation functions at t and t+tau for nonnegative tau. 
        
    Parameters
    ----------
    time_bin_list : [ndarray]
        List of time bins with the OC in either the initial or final bin in the list.
    
    ops_same_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that tau=0 (same time). These should exist in a single time-bin tensor space.

    ops_two_time : [ndarray]
        List of operators of which correlation functions should be calculated in the case that tau > 0. These should be ordered in a corresponding order to
        ops_same_time and should exist in a tensor space that is the outer product of two time bin tensor spaces, with the right space corresponding to the greater time.

    params : InputParams
        Simulation parameters 

    oc_end_list_flag : bool, default=True 
        Leaves the chain of bins unchanged for the calculation if the OC is at the end of the chain. Else assumes it is at the beginning and moves it to the end at start of calculation.
       
    completion_print_flag : bool, default=True
        Flag to print completion loop number percent of the calculation (note this is not the percent completion, and later loops complete faster than earlier ones). 

    Returns
    -------
    result : [np.ndarray]
        List of 2D arrays, each a two time correlation function corresponding by index to the operators in ops_same_time and ops_two_time.
        The two time correlation function is stored as f[t,tau], with non-negative tau and time increments between points given by the simulation.
    """
    d_t_total=params.d_t_total
    bond=params.bond
    d_t=np.prod(d_t_total)
    
    time_bin_list_copy = copy.deepcopy(time_bin_list)
    swap_matrix = swap(d_t, d_t)
    
    
    correlations = [np.zeros((len(time_bin_list_copy), len(time_bin_list_copy)), dtype=complex) for i in ops_two_time]
    
    # If the OC is at end of the time bin list, move it to the start (shifts OC from one end to other, index 0)
    if oc_end_list_flag:
        for i in range(len(time_bin_list_copy)-1,0,-1):
            bin_contraction = ncon([time_bin_list_copy[i-1],time_bin_list_copy[i]],[[-1,-2,1],[1,-3,-4]])
            left_bin, stemp, right_bin = _svd_tensors(bin_contraction, bond, d_t, d_t)
            time_bin_list_copy[i] = right_bin #right normalized system bin    
            time_bin_list_copy[i-1]= left_bin * stemp[None,None,:] #OC on left bin
    
    # Loop over to fill in correlation matrices values
    print('Correlation Calculation Completion:')
    loop_num = len(time_bin_list_copy) - 1
    print_rate = max(round(loop_num / 100.0), 1)
    for i in range(len(time_bin_list_copy)-1):

        i_1=time_bin_list_copy[0]
        i_2=time_bin_list_copy[1] 
        
        #for the first raw (tau=0)
        for k in range(len(correlations)):
            correlations[k][i,0] = expectation(i_1, ops_same_time[k]) #this means I'm storing [t,tau] 
        
        #for the rest of the rows (column by column)
        for j in range(len(time_bin_list_copy)-1):       
            state=ncon([i_1,i_2],[[-1,-2,1],[1,-3,-4]]) 
            for k in range(len(correlations)):
                correlations[k][i,j+1] = expectation_n(state, ops_two_time[k]) #this means I'm storing [t,tau] 

            
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
        
        if i % print_rate == 0 and completion_print_flag == True:
            print((float(i)/loop_num)*100, '%')
    return correlations


#-------------------------------------------
#Steady-state index helper, and correlations
#-------------------------------------------

def steady_state_index(pop:list,window: int=10, tol: float=1e-5) -> int or None:
    """
    Steady-state index helper function to find the time step 
    when the steady state is reached in the population dynamics.
    
    Parameters
    ----------
    pop : list
        List of population values
    
    window : int, default: 10
        Number of recent points to analyze
    
    tol : float, default: 1e-5
        Maximum deviation allowed in the final window
    
    Returns
    -------
    int or None
        The index of the start of the steady window, or None if none found.
    """
    pop = np.asarray(pop)
    for i in range(window, len(pop)):
        tail = pop[i-window:i]
        if tail.max() - tail.min() > tol:
            continue
        if np.max(np.abs(np.diff(tail))) > tol:
            continue
        print('Steady state found at list index i = ', i - window)
        return i - window
    return None

def steady_state_correlations(bins:Bins,pop:Pop1TLS,params:InputParams) -> SSCorrel|SSCorrel1Channel:
    """
    Efficient steady-state correlation calculation for continuous-wave pumping.
    This computes time differences starting from a convergence index (steady-state
    index). It returns a compact structure (SSCorrel or SSCorrel1Channel) 
    containing normalized g1 and g2 lists and raw correlation arrays.
    
    Parameters
    ----------
    bins : Bins
        Bins returned by time evolution functions
        cor_b field must contain the correlation tensors.
    
    params : InputParams
        Simulation parameters 
    
    pop : Pop1TLS
        Population dataclass used to detect steady-state   
        (it can be extended for other cases)
        
    Returns
    -------
    SSCorrel or SSCorrel1Channel
        Dataclass with precomputed time correlation lists for steady-state
        (for more info see the corresponding dataclasses in parameters.py)
    """
    
    pop=pop.pop
    cor_list=bins.cor_b
    delta_t, d_t_total,bond=params.delta_t,params.d_t_total,params.bond
    #First check convergence:
    conv_index =  steady_state_index(pop,10)  
    if conv_index is None:
        raise ValueError("tmax not long enough for steady state")
    # cor_list1=cor_list
    cor_list1=cor_list[conv_index:]
    t_cor=[0]
    i_1=cor_list1[-2]
    i_2=cor_list1[-1] #OC is in here
    p=0
    d_t=np.prod(d_t_total)
    swap_t_t=swap(d_t,d_t)
    
    if d_t==2:
        exp_0=expectation(i_2,delta_b_dag(delta_t, d_t_total))
        exp2_0=expectation(i_2, delta_b(delta_t, d_t_total))
        c1=[expectation(i_2, delta_b_dag(delta_t, d_t_total)@ delta_b(delta_t, d_t_total))]
        c2=[expectation(i_2, delta_b_dag(delta_t, d_t_total) @ delta_b_dag(delta_t, d_t_total) @ delta_b(delta_t, d_t_total) @delta_b(delta_t, d_t_total))]
        coher_list=[exp_0*exp2_0]
        denom=expectation(i_2,  delta_b_dag(delta_t, d_t_total)@ delta_b(delta_t, d_t_total))
        
        for i in range(len(cor_list1)-2,0,-1):
            state=ncon([i_1,i_2],[[-1,-2,1],[1,-3,-4]]) 
            # Separating between left and right spectra
            c1.append(expectation_2(state, g1(delta_t,d_t_total))) #for calculating the total spectra
            c2.append(expectation_2(state, g2(delta_t,d_t_total)))
            coher_list.append(exp_0*expectation(i_2, delta_b(delta_t, d_t_total))) #for calculating the coherent spectra           
            swaps=ncon([i_1,i_2,swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #swapping the feedback bin to the left so it is next to the next bin
            i_t1,stemp,i_t2=_svd_tensors(swaps,bond,d_t,d_t)
            i_2 = ncon([i_t1,np.diag(stemp)],[[-1,-2,1],[1,-3]]) #OC tau bin
            i_1=cor_list1[i-1] #next past bin for the next time step
            p+=1    
            t_cor.append(p*delta_t)       
        g1_list=c1/denom
        g2_list=c2/denom**2
        return SSCorrel1Channel(t_cor=t_cor,g1_list=g1_list,g2_list=g2_list,c1=c1,c2=c2)

    else:
        exp_0l=expectation(i_2,delta_b_dag_l(delta_t, d_t_total))
        exp2_0l=expectation(i_2, delta_b_l(delta_t, d_t_total))
        exp_0r=expectation(i_2, delta_b_dag_r(delta_t, d_t_total))
        exp2_0r=expectation(i_2, delta_b_r(delta_t, d_t_total))
        c1_l=[expectation(i_2, delta_b_dag_l(delta_t, d_t_total)@ delta_b_l(delta_t, d_t_total))]
        c1_r=[expectation(i_2, delta_b_dag_r(delta_t, d_t_total) @ delta_b_r(delta_t, d_t_total))]
        c2_l=[expectation(i_2, delta_b_dag_l(delta_t, d_t_total) @ delta_b_dag_l(delta_t, d_t_total) @ delta_b_l(delta_t, d_t_total) @delta_b_l(delta_t, d_t_total))]
        c2_r=[expectation(i_2,  delta_b_dag_r(delta_t, d_t_total) @ delta_b_dag_r(delta_t, d_t_total) @ delta_b_r(delta_t, d_t_total) @delta_b_r(delta_t, d_t_total))]
        coher_listl=[exp_0l*exp2_0l]
        coher_listr=[exp_0r*exp2_0r]    
        denoml=expectation(i_2,  delta_b_dag_l(delta_t, d_t_total)@ delta_b_l(delta_t, d_t_total))
        denomr = expectation(i_2,  delta_b_dag_r(delta_t, d_t_total)@ delta_b_r(delta_t, d_t_total))
        
        for i in range(len(cor_list1)-2,0,-1):
            state=ncon([i_1,i_2],[[-1,-2,1],[1,-3,-4]]) 
            # Separating between left and right spectra
            c1_l.append(expectation_2(state, g1_ll(delta_t,d_t_total))) #for calculating the total spectra
            c1_r.append(expectation_2(state, g1_rr(delta_t,d_t_total)))
            c2_l.append(expectation_2(state, g2_ll(delta_t,d_t_total)))
            c2_r.append(expectation_2(state, g2_rr(delta_t,d_t_total)))
            coher_listl.append(exp_0l*expectation(i_2, delta_b_l(delta_t, d_t_total))) #for calculating the coherent spectra
            coher_listr.append(exp_0r*expectation(i_2, delta_b_r(delta_t, d_t_total)))
            
            swaps=ncon([i_1,i_2,swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #swapping the feedback bin to the left so it is next to the next bin
            i_t1,stemp,i_t2=_svd_tensors(swaps,bond,d_t,d_t)
            i_2 = ncon([i_t1,np.diag(stemp)],[[-1,-2,1],[1,-3]]) #OC tau bin
            i_1=cor_list1[i-1] #next past bin for the next time step
            p+=1    
            t_cor.append(p*delta_t)
        
        g1_listl=c1_l/denoml
        g2_listl=c2_l/denoml**2
        g1_listr=c1_r/denomr
        g2_listr=c2_r/denomr**2
        
        return SSCorrel(t_cor=t_cor,g1_listl=g1_listl,g1_listr=g1_listr,g2_listl=g2_listl,g2_listr=g2_listr,c1_l=c1_l,c1_r=c1_r,c2_l=c2_l,c2_r=c2_r)
    
