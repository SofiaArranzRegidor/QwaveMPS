#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ADDING A COMMENT HERE TO CHECK

This module contains the simulation to evolve the system 
and calculate the main observables.

It requires the module ncon (pip install --user ncon)

"""


import numpy as np
from ncon import ncon
from scipy.linalg import svd,norm
from .operators import * #basic_operators,observables

# op=basic_operators()
# obs=observables()

#%%

def _svd_tensors(tensor:np.ndarray, left_shape:int, right_shape:int, bond:int, d_1:int, d_2:int) -> np.ndarray:
    """
    Application of the SVD and reshaping of the tensors

    Parameters
    ----------
    tensor : ndarray
        tensor to decompose

    left_shape : int
        left reshaping for decomposition
    
    right_shape : int
        right resaping for decomposition
    
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
    u, s, vt = svd(tensor.reshape(left_shape, right_shape), full_matrices=False)
    chi = min(bond, len(s))
    epsilon = 1e-12 #to avoid dividing by zero
    s_norm = s[:chi] / (norm(s[:chi])+ epsilon)
    u = u[:, :chi].reshape(tensor.shape[0],d_1,chi)
    vt = vt[:chi, :].reshape(chi,d_2,tensor.shape[-1])
    return u, s_norm, vt


def t_evol_mar(H:np.ndarray, i_s0:np.ndarray, i_n0:np.ndarray, delta_t:float, tmax:float, bond:int, d_sys:int, d_t:int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """ 
    Time evolution of the system without delay times
    
    Parameters
    ----------
    i_s0 : ndarray
        Initial system bin
    
    i_n0 : ndarray
        Initial time bin
    
    delta_t : float
        time step

    tmax : float
        max time

    bond : int
        max bond dimension
    
    d_sys : int, default: 2
        system bin dimension

    d_t : int, default: 2
        time bin dimension

    Returns
    -------
    sbins : [ndarray]
        A list with the system bins.
    
    tbins : [ndarray]
        A list with the time bins.
    """
    sbins=[] 
    sbins.append(i_s0)
    tbins=[]
    tbins.append(i_n0)

    N=int(tmax/delta_t)
    t_k=0
    i_s=i_s0
    Ham=H
    evO=u(Ham,d_sys,d_t)
    swap_sys_t=swap(d_sys,d_t)
           
    for k in range(1,N+1):      
        phi1=ncon([i_s,i_n0,evO],[[-1,2,3],[3,4,-4],[-2,-3,2,4]]) #system bin, time bin + u operator contraction  
        i_s,stemp,i_n=_svd_tensors(phi1,d_sys*phi1.shape[0],d_t*phi1.shape[-1], bond,d_sys,d_t)
        i_s=i_s*stemp[None,None,:] #OC system bin
        sbins.append(i_s)
        tbins.append(stemp[:,None,None]*i_n)
                    
        phi2=ncon([i_s,i_n,swap_sys_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #system bin, time bin + swap contraction
        i_n,stemp,i_st=_svd_tensors(phi2,d_t*phi2.shape[0],d_sys*phi2.shape[-1], bond,d_t,d_sys)
        i_s=stemp[:,None,None]*i_st   #OC system bin
        t_k += delta_t
    return sbins,tbins


def t_evol_nmar(H:np.ndarray, i_s0:np.ndarray, i_n0:np.ndarray, tau:float, delta_t:float, tmax:float, bond:int, d_t:int, d_sys:int) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """ 
    Time evolution of the system with delay times
    
    Parameters
    ----------
    i_s0 : ndarray
        Initial system bin
    
    i_n0 : ndarray
        Initial time bin

    tau : float
        Feedback time
    
    delta_t : float
        time step

    tmax : float
        max time

    bond : int
        max bond dimension
    
    d_sys : int, default: 2
        system bin dimension

    d_t : int, default: 2
        time bin dimension

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
    sbins=[] 
    tbins=[]
    taubins=[]
    nbins=[]
    schmidt=[]
    sbins.append(i_s0)   
    tbins.append(i_n0)
    taubins.append(i_n0)
    
    
    N=int(round(tmax/delta_t,0))
    t_k=0
    t_0=0
    Ham=H
    evO=u(Ham,d_t,d_sys,2) #Feedback loop means time evolution involves an input and a feedback time bin. Can generalize this later, leaving 2 for now so it runs.
    swap_t_t=swap(d_t,d_t)
    swap_sys_t=swap(d_sys,d_t)
    l=int(round(tau/delta_t,0)) #time steps between system and feedback
    
    while t_0 < tau:
        nbins.append(i_n0)
        t_0+=delta_t
    
    i_stemp=i_s0      
    
    for k in range(N):   
        #swap of the feedback until being next to the system
        i_tau= nbins[k] #starting from the feedback bin
        for i in range(k,k+l-1): 
            i_n=nbins[i+1] 
            swaps=ncon([i_tau,i_n,swap_t_t],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) 
            i_n2,stemp,i_t=_svd_tensors(swaps,d_t*swaps.shape[0],d_t*swaps.shape[3],bond,d_t,d_t)
            i_tau = ncon([np.diag(stemp),i_t],[[-1,1],[1,-3,-4]]) 
            nbins[i]=i_n2 
            
        #Make the system bin the OC
        i_1=ncon([i_tau,i_stemp],[[-1,-2,1],[1,-3,-4]]) #feedback-system contraction
        i_t,stemp,i_stemp=_svd_tensors(i_1,d_t*i_1.shape[0],d_sys*i_1.shape[-1], bond,d_t,d_sys)
        i_s=stemp[:,None,None]*i_stemp #OC system bin
        
        #now contract the 3 bins and apply u, followed by 2 svd to recover the 3 bins                 
        phi1=ncon([i_t,i_s,i_n0,evO],[[-1,3,1],[1,4,2],[2,5,-5],[-2,-3,-4,3,4,5]]) #tau bin, system bin, future time bin + u operator contraction
        i_t,stemp,i_2=_svd_tensors(phi1,d_t*phi1.shape[0],d_t*d_sys*phi1.shape[-1], bond,d_t,d_t*d_sys)
        i_2=stemp[:,None,None]*i_2
        i_stemp,stemp,i_n=_svd_tensors(i_2,d_sys*i_2.shape[0],d_t*i_2.shape[-1], bond,d_sys,d_t)
        i_s = i_stemp*stemp[None,None,:]
        sbins.append(i_s) 
        
        #swap system and i_n
        phi2=ncon([i_s,i_n,swap_sys_t],[[-1,3,2],[2,4,-4],[-2,-3,3,4]]) #system bin, time bin + swap contraction
        i_n,stemp,i_stemp=_svd_tensors(phi2,d_sys*phi2.shape[0],d_t*phi2.shape[-1], bond,d_sys,d_t)   
        i_n=i_n*stemp[None,None,:] #the OC in time bin     
        
        cont= ncon([i_t,i_n],[[-1,-2,1],[1,-3,-4]]) 
        i_t,stemp,i_n=_svd_tensors(cont,d_t*cont.shape[0],d_t*cont.shape[-1], bond,d_t,d_t)   
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
            i_t,stemp,i_n2=_svd_tensors(swaps,d_t*swaps.shape[0],d_t*swaps.shape[-1], bond,d_t,d_t)   
            i_tau = i_t*stemp[None,None,:] #OC tau bin         
            nbins[i]=i_n2    #update nbins            
        if k<(N-1):         
            nbins[k+1] = stemp[:,None,None]*i_n2 #new tau bin for the next time step
    return sbins,tbins,taubins#,schmidt



def pop_dynamics(sbins:list[np.ndarray], tbins:list[np.ndarray], delta_t:float):
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
    pop=np.array([expectation(s, TLS_pop()) for s in sbins])
    tbinsR=np.array([expectation(t, a_R_pop(delta_t)) for t in tbins])
    tbinsL=np.array([expectation(t, a_L_pop(delta_t)) for t in tbins])
   
    # Cumulative sums
    trans = np.cumsum(tbinsR)
    ref = np.cumsum(tbinsL)
    total = trans + ref + pop

        
    return pop,tbinsR,tbinsL,trans,ref,total


def pop_dynamics_1tls_nmar(sbins:list[np.ndarray], tbins:list[np.ndarray], taubins:list[np.ndarray], tau:float, delta_t:float):
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

    N=len(sbins) 
    pop=np.array([expectation(s, TLS_pop()) for s in sbins])
    tbins=np.array([expectation(t, a_pop(delta_t)) for t in tbins])
    tbins2=np.real([expectation(taus, a_pop(delta_t)) for taus in taubins])
    ph_loop=np.zeros(N,dtype=complex)
    trans=np.zeros(N,dtype=complex)
    total=np.zeros(N,dtype=complex)
    
    l=int(round(tau/delta_t,0))
    temp_out=0
    in_loop=0
    for i in range(N):
        temp_out+=tbins2[i]
        trans[i]=temp_out
        if i<=l:
            # temp_out+=tbins2[i]
            # trans[i]=temp_out
            in_loop=np.sum(tbins[:i+1]) 
            ph_loop[i]=in_loop
            total[i]=pop[i]+ph_loop[i]+trans[i]
        if i>l:
            # temp_out=np.sum(tbins2[i-l+1:i+1]) 
            # trans[i]=temp_out
            in_loop=np.sum(tbins[i-l+1:i+1]) 
            ph_loop[i]=in_loop
            # trans[i]=np.sum(tbins[i-l+1:i+1]) 
            total[i]  = pop[i] +  trans[i] + ph_loop[i]
        # trans[i]=np.sum(tbins2[0:i]) 
        # total[i]  = pop[i] + trans[i]*2
    return pop,tbins,trans,ph_loop,total

def pop_dynamics_2tls(sbins:list[np.ndarray], tbins:list[np.ndarray], delta_t:float, taubins:list[np.ndarray]=[], tau:float=0):
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
    N=len(sbins)
    pop1=np.zeros(N,dtype=complex)
    pop2=np.zeros(N,dtype=complex)
    tbinsR=np.zeros(N,dtype=complex)
    tbinsL=np.zeros(N,dtype=complex)
    tbinsR2=np.zeros(N+1,dtype=complex)
    tbinsL2=np.zeros(N+1,dtype=complex)
    in_R=np.zeros(N,dtype=complex)
    in_L=np.zeros(N,dtype=complex)
    trans=np.zeros(N,dtype=complex)
    ref=np.zeros(N,dtype=complex)
    total=np.zeros(N,dtype=complex)
    temp_inR=0
    temp_inL=0
    temp_outR=0
    temp_outL=0
    l=int(round(tau/delta_t,0))
    temp_trans=0
    temp_ref=0
    for i in range(N):
        i_s=sbins[i]
        i_sm=i_s.reshape(i_s.shape[0]*2,i_s.shape[-1]*2)
        u,sm,vt=svd(i_sm,full_matrices=False) #SVD
        i_s1 = u[:,range(len(sm))].reshape(i_s.shape[0],2,len(sm))  
        i_s1 = ncon([i_s1,np.diag(sm)],[[-1,-2,1],[1,-3]]) 
        i_s2 = vt[range(len(sm)),:].reshape(len(sm),2,i_s.shape[-1]) 
        i_s2 = ncon([np.diag(sm),i_s2],[[-1,1],[1,-2,-3]]) 
        pop1[i]=expectation(i_s1, TLS_pop())
        pop2[i]=expectation(i_s2, TLS_pop())    
        tbinsR[i]=np.real(expectation(tbins[i], a_R_pop(delta_t)))
        tbinsL[i]=np.real(expectation(tbins[i], a_L_pop(delta_t)))
        if tau != 0:
            tbinsR2[i]=np.real(expectation(taubins[i], a_R_pop(delta_t)))
            tbinsL2[i]=np.real(expectation(taubins[i], a_L_pop(delta_t)))
            temp_outR+=tbinsR2[i]
            temp_outL+=tbinsL2[i]
            trans[i]=temp_outR
            ref[i]=temp_outL
            if i <=l:
                temp_inR+=expectation(tbins[i], a_R_pop(delta_t))
                in_R[i] = temp_inR
                temp_inL+= expectation(tbins[i], a_L_pop(delta_t))
                in_L[i] = temp_inL
                total[i]  = pop1[i] + pop2[i]  + in_R[i] + in_L[i]  + trans[i] + ref[i]
            if i>l:
                temp_inR=np.sum(tbinsR[i-l+1:i+1]) 
                temp_inL=np.sum(tbinsL[i-l+1:i+1])
                total[i]  = pop1[i] + pop2[i]  + temp_inR + temp_inL + trans[i] + ref[i]
                in_R[i] = temp_inR
                in_L[i] = temp_inL
        if tau==0:
            temp_trans+= tbinsR[i]
            trans[i] = temp_trans
            temp_ref += tbinsL[i]
            ref[i] = temp_ref
            total[i]  = pop1[i] + pop2[i]  + trans[i] + ref[i]
        
    return pop1,pop2,tbinsR,tbinsL,trans,ref,total
