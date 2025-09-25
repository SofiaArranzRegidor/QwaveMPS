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

def t_evol_M(H,i_s0,i_n0,Deltat,tmax,bond,d_sys=2,d_t=2):
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
    
    return sbins,tbins


def t_evol_NM(H,i_s0,i_n0,tau,Deltat,tmax,bond,d_sys=2,d_t=2):
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
    
    
    N=int(tmax/Deltat)
    t_k=0
    t_0=0
    i_s=i_s0
    Ham=H
    evO=op.U_NM(Ham,d_sys)
    l=int(round(tau/Deltat,0)) #time steps between system and feedback
    
    while t_0 < tau:
        nbins.append(i_n0)
        t_0+=Deltat
    
    i_stemp=i_s      
    
    for k in range(N):  
        
        i_tau= nbins[k] #starting from the feedback bin
        for i in range(k,k+l-1): #swap of the feedback until being next to the system
            i_n=nbins[i+1] 
            swaps=ncon([i_tau,i_n,op.swap_t(d_t*d_t)],[[-1,5,2],[2,6,-4],[-2,-3,5,6]]) 
            swapsb=swaps.reshape(d_t*d_t*swaps.shape[0],d_t*d_t*swaps.shape[3]) 
            u,sm,vt=svd(swapsb,full_matrices=False)
            chi=min(bond,len(sm))
            i_n2 = u[:,range(chi)].reshape(swaps.shape[0],d_t*d_t,chi)
            i_t = vt[range(chi),:].reshape(chi,d_t*d_t,swaps.shape[3]) 
            stemp = sm[range(chi)]/norm(sm[range(chi)]) 
            i_tau = ncon([np.diag(stemp),i_t],[[-1,1],[1,-3,-4]]) 
            nbins[i]=i_n2 
            
        #Make the system bin the OC
        i_1=ncon([i_tau,i_stemp],[[-1,-2,1],[1,-3,-4]]) #feedback-system contraction
        i_1b=i_1.reshape(i_1.shape[0]*d_t*d_t,i_1.shape[3]*d_sys)
        u,sm,vt=svd(i_1b,full_matrices=False) #SVD
        chi=min(bond,len(sm))  #bond length
        stemp = sm[range(chi)]/norm(sm[range(chi)]) #Schmidt coeff
        i_t = u[:,range(chi)].reshape(i_1.shape[0],d_t*d_t,chi) #left normalized tau bin
        i_stemp = vt[range(chi),:].reshape(chi,d_sys,i_1.shape[3]) #right normalized system bin
        i_s = ncon([np.diag(stemp),i_stemp],[[-1,1],[1,-3,-4]]) #OC system bin
        
        #now contract the 3 bins and apply U, followed by 2 svd to recover the 3 bins                 
        phi1=ncon([i_t,i_s,i_n0,evO],[[-1,3,1],[1,4,2],[2,5,-5],[-2,-3,-4,3,4,5]]) #tau bin, system bin, future time bin + U operator contraction
        phi1b=phi1.reshape(d_t*d_t*phi1.shape[0],d_sys*d_t*d_t*phi1.shape[4]) #d*d in the second bc I do the double svd with the right part
        u,sm,vt=svd(phi1b,full_matrices=False) #SVD       
        chi1=min(bond,len(sm))  #Bond legth
        i_t = u[:,range(chi1)].reshape(phi1.shape[0],d_t*d_t,chi1) #left normalized feedback bin    
        i_2 = vt[range(chi1),:].reshape(chi1,d_sys,d_t*d_t,phi1.shape[4]) #right normalized 2site bin
        stemp = sm[range(chi1)]/norm(sm[range(chi1)])   #Schmidt coeff 
        
        i_2 = ncon([np.diag(stemp),i_2],[[-1,1],[1,-2,-3,-4]]).reshape(d_sys*chi1,d_t*d_t*phi1.shape[4]) #OC 2site bin
        u,sm,vt=svd(i_2,full_matrices=False) #SVD
        chi2=min(bond,len(sm)) #Bond length
        i_stemp = u[:,range(chi2)].reshape(chi1,d_sys,chi2) #left normalized system bin      
        i_n = vt[range(chi2),:].reshape(chi2,d_t*d_t,phi1.shape[4]) #right normalized time bin
        stemp = sm[range(chi2)]/norm(sm[range(chi2)]) #Schmidt coeff 
        # schmidt_tau.append(stemp)
        i_s = ncon([i_stemp,np.diag(stemp)],[[-1,-2,1],[1,-3]])  #the OC is in i_s              
        sbins.append(i_s) #the system bin is computed here as it is the moment it is the OC
        
        #swap system and i_n
        phi2=ncon([i_s,i_n,op.swap(d_sys,d_t)],[[-1,3,2],[2,4,-4],[-2,-3,3,4]]) #system bin, time bin + swap contraction
        # tbinsbins.append(phi2)
        phi2b=phi2.reshape(d_sys*phi2.shape[0],d_t*d_t*phi2.shape[3])
        u,sm,vt=svd(phi2b,full_matrices=False) #SVD
        chi=min(bond,len(sm))    #Bond legth
        stemp = sm[range(chi)]/norm(sm[range(chi)]) #Schmidt coeff 
        
        i_n = u[:,range(chi)].reshape(phi2.shape[0],d_t*d_t,chi)  #left normalized time bin
        i_stemp = vt[range(chi),:].reshape(chi,d_sys,phi2.shape[3]) #right normalized system bin 
        i_n = ncon([i_n,np.diag(stemp)],[[-1,-2,1],[1,-3]]) #the OC in time bin     
        cont= ncon([i_t,i_n],[[-1,-2,1],[1,-3,-4]]) #feedback bin, time bin contraction
        contb=cont.reshape(cont.shape[0]*d_t*d_t,cont.shape[3]*d_t*d_t)
        u,sm,vt=svd(contb,full_matrices=False) #SVD
        chi=min(bond,len(sm))   #Bond legth
        stemp = sm[range(chi)]/norm(sm[range(chi)])  #Schmidt coeff 
        
        
        i_t = u[:,range(chi)].reshape(cont.shape[0],d_t*d_t,chi) #left normalized feedback bin 
        i_n = vt[range(chi),:].reshape(chi,d_t*d_t,cont.shape[3]) #right normalized time bin
        i_tau = ncon([i_t,np.diag(stemp)],[[-1,-2,1],[1,-3]]) #OC in feedback bin     
        tbins.append(ncon([np.diag(stemp),i_n],[[-1,1],[1,-2,-3],]))
        
        taubins.append(i_tau) 
        nbins[k+l-1]=i_tau #update of the feedback bin
        nbins.append(i_n)         
        t_k += Deltat
        schmidt.append(stemp) #storing the Schmidt coeff for calculating the entanglement

    return sbins,tbins#,taubins,schmidt



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

def pop_dynamics_2TLS(sbins,tbins,Deltat):
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
    trans=np.zeros(N,dtype=complex)
    ref=np.zeros(N,dtype=complex)
    total=np.zeros(N,dtype=complex)
    temp_trans=0
    temp_ref=0

    
    for i in range(N):
        i_s=sbins[i]
        i_s.reshape(i_s.shape[0]*2,i_s.shape[-1]*2)
        # print(i_s.shape)
        u,sm,vt=svd(i_s,full_matrices=False) #SVD
        print(len(sm))
        i_s1 = u[:,range(len(sm))].reshape(i_s.shape[0],2,len(sm)) #left normalized system bin      
        i_s2 = vt[range(len(sm)),:].reshape(len(sm),2,i_s.shape[-1]) #right normalized time bin
        pop1[i]=op.expectation(i_s1, obs.TLS_pop())
        pop2[i]=op.expectation(i_s2, obs.TLS_pop())
        tbinsR[i]=np.real(op.expectation(tbins[i], obs.a_R_pop(Deltat)))
        tbinsL[i]=np.real(op.expectation(tbins[i], obs.a_L_pop(Deltat)))
        temp_trans+=tbinsR[i]
        trans[i] = temp_trans
        temp_ref+= tbinsL[i]
        ref[i] = temp_ref
        total[i] = temp_trans + temp_ref + pop1[i] + pop2[i]
        
    return pop1,pop2,tbinsR,tbinsL,trans,ref,total