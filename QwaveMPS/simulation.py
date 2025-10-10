#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the simulation to evolve the system 
and calculate the main observables.

It requires the module ncon (pip install --user ncon)

"""


import numpy as np
from ncon import ncon
from scipy.linalg import svd,norm
from .operators import basic_operators,observables

op=basic_operators()
obs=observables()

#%%

def svd_tensors(tensor, left_shape, right_shape, bond,d_1,d_2):
    """
    Application of the SVD and reshaping of the tensors

    Parameters
    ----------
    tensor : tensor to decompose
    left_shape : left reshaping for decomposition
    right_shape : right resaping for decomposition
    bond : max. bond dimension
    d_1 : physical dimension of first tensor
    d_2 : physical dimension of second tensor

    Returns
    -------
    u : left normalized tensor
    s_norm : smichdt coefficients normalized 
    vt : transposed right normalized tensor
    """
    u, s, vt = svd(tensor.reshape(left_shape, right_shape), full_matrices=False)
    chi = min(bond, len(s))
    epsilon = 1e-12 #to avoid dividing by zero
    s_norm = s[:chi] / (norm(s[:chi])+ epsilon)
    u = u[:, :chi].reshape(tensor.shape[0],d_1,chi)
    vt = vt[:chi, :].reshape(chi,d_2,tensor.shape[-1])
    return u, s_norm, vt


def t_evol_M(H,i_s0,i_n0,Deltat,tmax,bond,d_sys,d_t):
    """ 
    Time evolution of the system without delay times
    
    Parameters
    ----------
    i_s0 : Initial system bin
    i_n0 : Initial time bin
    Deltat : time step
    tmax : max time 
    bond : max bond dimension
    d_sys : system bin dimension. The default is 2.
    d_t : time bin dimension. The default is 2.

    Returns
    -------
    sbins : A list with the system bins.
    tbins : A list with the time bins.
    """
    sbins=[] 
    i_s0.reshape(1,d_sys,1)
    sbins.append(i_s0)
    tbins=[]
    tbins.append(i_n0)

    N=int(tmax/Deltat)
    t_k=0
    i_s=i_s0
    Ham=H
    evO=op.U(Ham,d_sys)
           
    for k in range(1,N+1):      
        phi1=ncon([i_s,i_n0,evO],[[-1,2,3],[3,4,-4],[-2,-3,2,4]]) #system bin, time bin + U operator contraction  
        i_s,stemp,i_n=svd_tensors(phi1,d_sys*phi1.shape[0],d_t*phi1.shape[-1], bond,d_sys,d_t)
        i_s=i_s*stemp[None,None,:] #OC system bin
        sbins.append(i_s)
        tbins.append(stemp[:,None,None]*i_n)
                    
        phi2=ncon([i_s,i_n,op.swap(d_t,d_sys)],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #system bin, time bin + swap contraction
        i_n,stemp,i_st=svd_tensors(phi2,d_t*phi2.shape[0],d_sys*phi2.shape[-1], bond,d_t,d_sys)
        i_s=stemp[:,None,None]*i_st   #OC system bin
        t_k += Deltat
    return sbins,tbins


def t_evol_NM(H,i_s0,i_n0,tau,Deltat,tmax,bond,d_t,d_sys):
    """ 
    Time evolution of the system with delay times
    
    Parameters
    ----------
    i_s0 : Initial system bin
    i_n0 : Initial time bin
    tau: Feedback time
    Deltat : time step
    tmax : max time 
    bond : max bond dimension
    d_sys : system bin dimension. The default is 2.
    d_t : time bin dimension. The default is 2.

    Returns
    -------
    sbins : A list with the system bins.
    tbins : A list with the time bins (with OC).
    taubins: A list with the feedback bins (with OC). 
    schmidt: A list with the Schmidt coefficients
    """
    sbins=[] 
    tbins=[]
    taubins=[]
    nbins=[]
    schmidt=[]
    i_s0.reshape(1,d_sys,1)
    sbins.append(i_s0)   
    tbins.append(i_n0)
    taubins.append(i_n0)
    
    
    N=int(round(tmax/Deltat,0))
    t_k=0
    t_0=0
    Ham=H
    evO=op.U_NM(Ham,d_t,d_sys)
    l=int(round(tau/Deltat,0)) #time steps between system and feedback
    
    while t_0 < tau:
        nbins.append(i_n0)
        t_0+=Deltat
    
    i_stemp=i_s0      

    for k in range(N):   
        #swap of the feedback until being next to the system
        i_tau= nbins[k] #starting from the feedback bin
        for i in range(k,k+l-1): 
            i_n=nbins[i+1] 
            swaps=ncon([i_tau,i_n,op.swap_t(d_t)],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) 
            i_n2,stemp,i_t=svd_tensors(swaps,d_t*swaps.shape[0],d_t*swaps.shape[3],bond,d_t,d_t)
            i_tau = ncon([np.diag(stemp),i_t],[[-1,1],[1,-3,-4]]) 
            nbins[i]=i_n2 
            
        #Make the system bin the OC
        i_1=ncon([i_tau,i_stemp],[[-1,-2,1],[1,-3,-4]]) #feedback-system contraction
        i_t,stemp,i_stemp=svd_tensors(i_1,d_t*i_1.shape[0],d_sys*i_1.shape[-1], bond,d_t,d_sys)
        i_s=stemp[:,None,None]*i_stemp #OC system bin
        
        #now contract the 3 bins and apply U, followed by 2 svd to recover the 3 bins                 
        phi1=ncon([i_t,i_s,i_n0,evO],[[-1,3,1],[1,4,2],[2,5,-5],[-2,-3,-4,3,4,5]]) #tau bin, system bin, future time bin + U operator contraction
        i_t,stemp,i_2=svd_tensors(phi1,d_t*phi1.shape[0],d_t*d_sys*phi1.shape[-1], bond,d_t,d_t*d_sys)
        i_2=stemp[:,None,None]*i_2
        i_stemp,stemp,i_n=svd_tensors(i_2,d_sys*i_2.shape[0],d_t*i_2.shape[-1], bond,d_sys,d_t)
        i_s = i_stemp*stemp[None,None,:]
        sbins.append(i_s) 
        
        #swap system and i_n
        phi2=ncon([i_s,i_n,op.swap(d_t,d_sys)],[[-1,3,2],[2,4,-4],[-2,-3,3,4]]) #system bin, time bin + swap contraction
        i_n,stemp,i_stemp=svd_tensors(phi2,d_sys*phi2.shape[0],d_t*phi2.shape[-1], bond,d_sys,d_t)   
        i_n=i_n*stemp[None,None,:] #the OC in time bin     
        
        cont= ncon([i_t,i_n],[[-1,-2,1],[1,-3,-4]]) 
        i_t,stemp,i_n=svd_tensors(cont,d_t*cont.shape[0],d_t*cont.shape[-1], bond,d_t,d_t)   
        i_tau = i_t*stemp[None,None,:] #OC in feedback bin     
        tbins.append(stemp[:,None,None]*i_n)
        
        #feedback bin, time bin contraction
        taubins.append(i_tau) 
        nbins[k+l-1]=i_tau #update of the feedback bin
        nbins.append(i_n)         
        t_k += Deltat
        schmidt.append(stemp) #storing the Schmidt coeff for calculating the entanglement

        #swap back of the feedback bin      
        for i in range(k+l-1,k,-1): #goes from the last time bin to first one
            i_n=nbins[i-1] #time bin
            swaps=ncon([i_n,i_tau,op.swap_t(d_t)],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #time bin, feedback bin + swap contraction
            i_t,stemp,i_n2=svd_tensors(swaps,d_t*swaps.shape[0],d_t*swaps.shape[-1], bond,d_t,d_t)   
            i_tau = i_t*stemp[None,None,:] #OC tau bin         
            nbins[i]=i_n2    #update nbins            
        if k<(N-1):         
            nbins[k+1] = stemp[:,None,None]*i_n2 #new tau bin for the next time step
    return sbins,tbins,taubins#,schmidt



def pop_dynamics(sbins,tbins,Deltat):
    """
    It calculates the main population dynamics

    Parameters
    ----------
    sbins : A list with the system bins.
    tbins : A list with the time bins.

    Returns
    -------
    pop : TLS population.
    tbinsR : Right photon flux.
    tbinsL : Left photon flux.
    trans : Integrated flux to right.
    ref : Integrated flux to the left.
    total : Total leaving the system, must be equal 
    to total # excitations.

    """
    pop=np.array([op.expectation(s, obs.TLS_pop()) for s in sbins])
    tbinsR=np.array([op.expectation(t, obs.a_R_pop(Deltat)) for t in tbins])
    tbinsL=np.array([op.expectation(t, obs.a_L_pop(Deltat)) for t in tbins])
   
    # Cumulative sums
    trans = np.cumsum(tbinsR)
    ref = np.cumsum(tbinsL)
    total = trans + ref + pop

        
    return pop,tbinsR,tbinsL,trans,ref,total


def pop_dynamics_1TLS_NM(sbins,tbins,taubins,tau,Deltat):
    """
    It calculates the main population dynamics

    Parameters
    ----------
    sbins : A list with the system bins.
    tbins : A list with the time bins.

    Returns
    -------
    pop : TLS population.
    tbinsR : Right photon flux.
    tbinsL : Left photon flux.
    trans : Integrated flux to right.
    ref : Integrated flux to the left.
    total : Total leaving the system, must be equal 
    to total # excitations.

    """
    N=len(sbins) 
    pop=np.array([op.expectation(s, obs.TLS_pop()) for s in sbins])
    tbins=np.array([op.expectation(t, obs.a_pop(Deltat)) for t in tbins])
    tbins2=np.real([op.expectation(taus, obs.a_pop(Deltat)) for taus in taubins])
    trans=np.zeros(N,dtype=complex)
    total=np.zeros(N,dtype=complex)
    
    l=int(round(tau/Deltat,0))
    temp_outR=0
    for i in range(N):
        if i<=l:
            total[i]=pop[i]
        if i>l:
            trans[i]=np.sum(tbins[i-l+1:i+1]) 
            total[i]  = pop[i] + trans[i] + tbins2[i]
        
    return pop,tbins,trans,total

def pop_dynamics_2TLS(sbins,tbins,taubins,tau,Deltat):
    """
    It calculates the main population dynamics

    Parameters
    ----------
    sbins : A list with the system bins.
    tbins : A list with the time bins.

    Returns
    -------
    pop : TLS population.
    tbinsR : Right photon flux.
    tbinsL : Left photon flux.
    trans : Integrated flux to right.
    ref : Integrated flux to the left.
    total : Total leaving the system, must be equal 
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
    l=int(round(tau/Deltat,0))
    
    for i in range(N):
        i_s=sbins[i]
        i_sm=i_s.reshape(i_s.shape[0]*2,i_s.shape[-1]*2)
        u,sm,vt=svd(i_sm,full_matrices=False) #SVD
        i_s1 = u[:,range(len(sm))].reshape(i_s.shape[0],2,len(sm))  
        i_s1 = ncon([i_s1,np.diag(sm)],[[-1,-2,1],[1,-3]]) 
        i_s2 = vt[range(len(sm)),:].reshape(len(sm),2,i_s.shape[-1]) 
        i_s2 = ncon([np.diag(sm),i_s2],[[-1,1],[1,-2,-3]]) 
        pop1[i]=op.expectation(i_s1, obs.TLS_pop())
        pop2[i]=op.expectation(i_s2, obs.TLS_pop())    
        tbinsR[i]=np.real(op.expectation(tbins[i], obs.a_R_pop(Deltat)))
        tbinsL[i]=np.real(op.expectation(tbins[i], obs.a_L_pop(Deltat)))
        tbinsR2[i]=np.real(op.expectation(taubins[i], obs.a_R_pop(Deltat)))
        tbinsL2[i]=np.real(op.expectation(taubins[i], obs.a_L_pop(Deltat)))
        temp_outR+=tbinsR2[i]
        temp_outL+=tbinsL2[i]
        trans[i]=temp_outR
        ref[i]=temp_outL
        
        if i <=l:
            temp_inR+=op.expectation(tbins[i], obs.a_R_pop(Deltat))
            in_R[i] = temp_inR
            temp_inL+= op.expectation(tbins[i], obs.a_L_pop(Deltat))
            in_L[i] = temp_inL
            total[i]  = pop1[i] + pop2[i]  + in_R[i] + in_L[i]  + trans[i] + ref[i]
        if i>l:
            temp_inR=np.sum(tbinsR[i-l+1:i+1]) 
            temp_inL=np.sum(tbinsL[i-l+1:i+1])
            total[i]  = pop1[i] + pop2[i]  + temp_inR + temp_inL + trans[i] + ref[i]
            in_R[i] = temp_inR
            in_L[i] = temp_inL
    
        
    return pop1,pop2,tbinsR,tbinsL,trans,ref,total