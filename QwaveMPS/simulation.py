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

def t_evol(H,i_s0,i_n0,Deltat,tmax,bond,d_sys=2,d_t=2):
    """ 
    Time evolution of the system
    
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
    tlist : A list with the time steps.

    """
    sbins=[] 
    tlist=[] 
    sbins.append(i_s0)
    tlist.append(0.)   
    tbins=[]
    tbins.append(i_n0)

    N=int(tmax/Deltat)
    t_k=0
    i_s=i_s0
    Ham=H
    evO=op.U(Ham)
           
    for k in range(1,N+1):      
        phi1=ncon([i_s,i_n0,evO],[[-1,2,3],[3,4,-4],[-2,-3,2,4]]) #system bin, future time bin + U operator contraction  
        phi1b=phi1.reshape(d_sys*phi1.shape[0],d_t*d_t*phi1.shape[-1]) #reshape for using SVD function
    
        u,sm,vt=svd(phi1b,full_matrices=False) #SVD
        chi=min(bond,len(sm)) #bond length
        i_s = u[:,range(chi)].reshape(phi1.shape[0],d_sys,chi) #left normalized system bin
        i_n = vt[range(chi),:].reshape(chi,d_t*d_t,phi1.shape[-1]) # right normalized time bin
        stemp = sm[range(chi)]/norm(sm[range(chi)]) #Schmidt coeff
        i_s = ncon([i_s,np.diag(stemp)],[[-1,-2,1],[1,-3],]) #OC system bin
        sbins.append(i_s)
        tbins.append(ncon([np.diag(stemp),i_n],[[-1,1],[1,-2,-3],]))
        
            
        phi2=ncon([i_s,i_n,op.swap()],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) #system bin, time bin + swap contraction
        phi2b=phi2.reshape(d_t*d_t*phi2.shape[0],d_sys*phi2.shape[-1]) #reshape for using SVD function
    
        u,sm,vt=svd(phi2b,full_matrices=False) #SVD
        chi=min(bond,len(sm))     #bond length
        stemp = sm[range(chi)]/norm(sm[range(chi)]) #Schmidt coeff
        i_n = u[:,range(chi)].reshape(phi2.shape[0],d_t*d_t,chi) #left normalized time bins
        i_st = vt[range(chi),:].reshape(chi,d_sys,phi2.shape[-1]) #right normalized system bin    
        i_s = ncon([np.diag(stemp),i_st],[[-1,1],[1,-3,-4]]) #OC system bin
            
        t_k += Deltat
        tlist.append(t_k)
    
    return sbins,tbins

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
    N=len(sbins)
    pop=np.zeros(N,dtype=complex)
    tbinsR=np.zeros(N,dtype=complex)
    tbinsL=np.zeros(N,dtype=complex)
    trans=np.zeros(N,dtype=complex)
    ref=np.zeros(N,dtype=complex)
    total=np.zeros(N,dtype=complex)
    temp_trans=0
    temp_ref=0
    
    for i in range(N):
        pop[i]=op.expectation(sbins[i], obs.TLS_pop())
        tbinsR[i]=np.real(op.expectation(tbins[i], obs.a_R_pop(Deltat)))
        tbinsL[i]=np.real(op.expectation(tbins[i], obs.a_L_pop(Deltat)))
        temp_trans+=tbinsR[i]
        trans[i] = temp_trans
        temp_ref+= tbinsL[i]
        ref[i] = temp_ref
        total[i] = temp_trans + temp_ref + pop[i]
        
    return pop,tbinsR,tbinsL,trans,ref,total