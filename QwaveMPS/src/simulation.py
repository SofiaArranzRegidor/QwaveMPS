#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ADDING A COMMENT HERE TO CHECK

This module contains the simulation to evolve the system 
and calculate the main observables.

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

#%%

def _svd_tensors(tensor:np.ndarray, bond:int, d_1:int, d_2:int) -> np.ndarray:
    """
    Application of the SVD and reshaping of the tensors

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


def t_evol_mar(ham:np.ndarray|list, i_s0:np.ndarray, i_n0:np.ndarray, params:InputParams) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """ 
    Time evolution of the system without delay times
    
    Parameters
    ----------
    i_s0 : ndarray
        Initial system bin.
    
    input_field : Iterator
        Generator of time bins incident the system.
        
    i_n0: ndarray
    
    delta_t : float
        time step

    tmax : float
        max time

    bond : int
        max bond dimension
    
    d_sys_total : ndarray
        List of sizes of system Hilbert spaces.

    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

    Returns
    -------
    sbins : [ndarray]
        A list with the system bins.
    
    tbins : [ndarray]
        A list with the time bins.
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



def t_evol_nmar(ham:np.ndarray|list, i_s0:np.ndarray, i_n0:np.ndarray,params:InputParams) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """ 
    Time evolution of the system with delay times
    
    Parameters
    ----------
    i_s0 : ndarray
        Initial system bin
    
    input_field : Iterator
        Generator of time bins incident the system.

    tau : float
        Feedback time
    
    delta_t : float
        time step

    tmax : float
        max time

    bond : int
        max bond dimension
    
    d_sys_total : ndarray
        List of sizes of system Hilbert spaces.

    d_t_total : ndarray
        List of sizes of the photonic Hilbert spaces.

    Returns
    -------
    sbins : [ndarray]
        A list with the system bins.
    
    tbins : [ndarray]
        A list with the time bins (with OC).
    
    taubins : [ndarray]
        A list of the feedback bins (with OC)

    schmidt : [ndarray]
        A list of the Schmidt coefficients
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
            
    # return sbins,tbins,taubins,cor_list,schmidt,schmidt_tau
    return Bins(sys_b=sbins,time_b=tbins,tau_b=taubins,cor_b=cor_list,schmidt=schmidt,schmidt_tau=schmidt_tau)


def single_time_expectation(normalized_bins:list[np.ndarray], ops_list:list[np.ndarray]):
    """
    Takes the expectation values of several operators of a list of normalized bins. 

    Parameters
    ----------
    normalized_bins : list[ndarray]
        List of OC normalized bins in order of time to have localized expectation values taken.

    ops_list : list[ndarray]
        List of operators to take expectation values.
    
    Returns
    -------
    u : list[np.ndarray]
        List of time dependent expectation values for the different observables. Indexed first with operator number, second with time.
    """

    return np.array([[expectation(bin, op) for bin in normalized_bins] for op in ops_list])


def pop_dynamics(bins:Bins, params:InputParams):
    """
    Calculates the main population dynamics

    Parameters
    ----------
    sbins : [ndarray] 
        A list with the system bins.

    tbins : [ndarray]
        A list with the time bins.

    delta_t : float
        Time step size.

    Returns
    -------
    pop : ndarray
        1D array containing TLS population.

    tbinsR : ndarray
        1D array containing the right photon flux.

    tbinsL : ndarray
        1D array containing the left photon flux.

    trans : ndarray
        1D array containing the integrated flux to right.

    ref : ndarray
        1D array containing the integrated flux to the left.
    
    total : ndarray
        1D array containig the total quanta leaving the system, must be equal 
    to total # excitations.
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


def pop_dynamics_1tls_nmar(bins:Bins, params:InputParams):
    """
    Calculates the main population dynamics

    Parameters
    ----------
    sbins : [ndarray] 
        A list with the system bins.

    tbins : [ndarray]
        A list with the time bins.

    delta_t : float
        Time step size.

    Returns
    -------
    pop : ndarray
        1D array containing TLS population.

    tbinsR : ndarray
        1D array containing the right photon flux.

    tbinsL : ndarray
        1D array containing the left photon flux.

    trans : ndarray
        1D array containing the integrated flux to right.

    ref : ndarray
        1D array containing the integrated flux to the left.
    
    total : ndarray
        1D array containig the total quanta leaving the system, must be equal 
    to total # excitations.
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


def pop_dynamics_2tls(bins:Bins,params:InputParams):
                      #delta_t:float,d_sys_total:np.array,d_t_total:np.array, tau:float=0):
    """
    Calculates the main population dynamics

    Parameters
    ----------
    sbins : [ndarray] 
        A list with the system bins.

    tbins : [ndarray]
        A list with the time bins.

    delta_t : float
        Time step size.

    Returns
    -------
    pop : ndarray
        1D array containing TLS population.

    tbins_r : ndarray
        1D array containing the right photon flux.

    tbins_l : ndarray
        1D array containing the left photon flux.

    trans : ndarray
        1D array containing the integrated flux to right.

    ref : ndarray
        1D array containing the integrated flux to the left.
    
    total : ndarray
        1D array containig the total quanta leaving the system, must be equal 
    to total # excitations.
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


def first_order_correlation(bins:Bins, params:InputParams,single_channel=False):
    """
    Calculates the first order correlation function of the right moving photons

    Parameters
    ----------

    cor_bins : [ndarray]
        A list with the time bins involved in the correlation.

    delta_t : float
        Time step size.

    Returns
    -------
    Expectation value of g1_r
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


def second_order_correlation(bins:Bins,params:InputParams,single_channel=False):
    """
    Calculates the first order correlation function of the right moving photons

    Parameters
    ----------

    cor_bins : [ndarray]
        A list with the time bins involved in the correlation.

    delta_t : float
        Time step size.

    Returns
    -------
    Expectation value of g1_r
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


# Expectation operation ket for larger/arbitrary tensor spaces
# Saving previous MPO rank avoids creation of lists for repeated calculations
def expectation_n(ket:np.ndarray, mpo:np.ndarray) -> complex:
    """ 
    Takes the expectation value of an nth rank tensor with an nth rank MPO.
    
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
    """

    curr_rank_op = len(mpo.shape)+2 #Adjusted for indices numbering
    if expectation_n.prev_rank != curr_rank_op:
        expectation_n.prev_rank = curr_rank_op
        half_rank_op = int(curr_rank_op/2)+1
        expectation_n.ket_indices = np.concatenate((np.arange(1,half_rank_op, dtype=int), [curr_rank_op])).tolist()
        expectation_n.op_indices = np.concatenate((np.arange(half_rank_op, curr_rank_op, dtype=int), np.arange(2,half_rank_op, dtype=int))).tolist()
        expectation_n.bra_indices = np.concatenate(([1], np.arange(half_rank_op,curr_rank_op+1, dtype=int))).tolist()

    return ncon([np.conj(ket), mpo, ket], [expectation_n.ket_indices, expectation_n.op_indices, expectation_n.bra_indices])
expectation_n.prev_rank = None

'''
Takes in list of time ordered normalized (with OC) time bins at position of relevance
'''
def two_time_correlations(time_bin_list, ops_same_time, ops_two_time, d_t, bond, oc_end_list_flag=True, completion_print_flag=True):
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

'''
def general_field_correlation(cor_list1:list[np.array],operator1:np.ndarray,operator2:np.ndarray, delta_t:float,d_t_total:np.array,bond:int):
    """
    Calculates the first order correlation function of the right moving photons

    Parameters
    ----------

    cor_bins : [ndarray]
        A list with the time bins involved in the correlation.

    delta_t : float
        Time step size.

    Returns
    -------
    Expectation value of g1_r
    """
    cor_list2 =  cor_list1
    cor_matrix= np.zeros((len(cor_list1),len(cor_list1)),dtype='complex') 
    d_t=np.prod(d_t_total)
    swap_t_t=swap(d_t,d_t)
    
    operator12 = np.kron(operator1,operator2).reshape(d_t,d_t,d_t,d_t)
    
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
        
        cor_matrix[0,j] = expectation(i_1,(operator1 @ operator2)) 
        
        for i in range(len(cor_list2)-1):  
            # print('iteration', i)
            state=ncon([i_1,i_2],[[-1,-2,1],[1,-3,-4]]) 
            
            cor_matrix[i+1,j]=expectation_2(state, operator12)
            
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
    return cor_matrix 
'''

def steady_state_correlations(bins:Bins,pop:Pop1TLS,params:InputParams):
    """For faster calculations when we have a CW classical pump"""
    
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
    
def entanglement(sch):
    ent_list=[]
    for s in sch:
        a=s**2   
        a=np.trim_zeros(a) 
        b=np.log2(a)
        c=a*b
        ent=-sum(c)
        ent_list.append(ent)
    return ent_list

def spectrum_w(delta_t,g1_list):
    #Fourier Transform
    s_w = np.fft.fftshift(np.fft.fft(g1_list))
    n=s_w.size
    wlist = np.fft.fftshift(np.fft.fftfreq(n,d=delta_t))*2*np.pi   
    return s_w,wlist