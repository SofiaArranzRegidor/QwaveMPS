#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains the main quantum operators

It requires the module ncon (pip install --user ncon)

"""

import numpy as np
from scipy.linalg import expm
from ncon import ncon

#%%

class basic_operators:
    """
    This includes the basic boson and TLS operators used to build the Hamiltonian 
    """
    def sigmaplus(self):  
        """Raising operator"""
        a = np.zeros((2,2),dtype=complex)
        a[1,0]=1.       
        return a
    
    def sigmaminus(self):  
        """Lowering operator"""
        a = np.zeros((2,2),dtype=complex)
        a[0,1]=1.       
        return a
        
    def DeltaBdagL(self,Deltat):  
        """Left noise creation operator. Deltat is the time step"""
        a=np.kron(np.sqrt(Deltat)*self.sigmaplus(),np.eye(2))     
        return a
    
    def DeltaBdagR(self,Deltat):  
        """Right noise creation operator. Deltat is the time step"""
        a=np.kron(np.eye(2),np.sqrt(Deltat)*self.sigmaplus())     
        return a
    
    def DeltaBL(self,Deltat):  
        """Left noise annihilation operator. Deltat is the time step"""
        a=np.kron(np.sqrt(Deltat)*self.sigmaminus(),np.eye(2))     
        return a
    
    def DeltaBR(self,Deltat):  
        """Right noise annihilation operator. Deltat is the time step"""
        a=np.kron(np.eye(2),np.sqrt(Deltat)*self.sigmaminus())       
        return a
    
    def e(self,d_sys=2):
        exc = np.zeros((d_sys,d_sys),dtype=complex)
        exc[1,1]=1.      
        return exc

    def U(self,Hm,d_sys=2,d_t=2):
        """Time evolution operator. H is the Hamiltonian """
        sol= expm(-Hm.reshape(d_sys*d_t*d_t,d_sys*d_t*d_t))
        return sol.reshape(d_sys,d_t*d_t,d_sys,d_t*d_t)

    def U_NM(self,Hm,d_sys=2,d_t=2):
        """Time evolution operator with feedback. H is the Hamiltonian """
        sol= expm(-Hm.reshape(d_sys*d_t*d_t*d_t*d_t,d_sys*d_t*d_t*d_t*d_t))
        return sol.reshape(d_sys,d_t*d_t,d_t*d_t,d_sys,d_t*d_t,d_t*d_t)

    def swap(self,d_sys=2,d_t=2):
        d_t=d_t*d_t
        swap = np.zeros([d_sys*d_t,d_sys*d_t],dtype=complex)
        for i in range(d_sys):
            swap[i,i*d_t]=1
            swap[i+d_sys,(i*d_t)+1]=1
            swap[i+2*d_sys,(i*d_t)+2]=1
            
        return swap.reshape(d_sys,d_t,d_sys,d_t)   
    
    def swap_t(self,d_t):
        swap_t1 = np.zeros([d_t*d_t,d_t*d_t],dtype=complex)
        for i in range(d_t):
            swap_t1[i,i*d_t]=1
            swap_t1[i+d_t,(i*d_t)+1]=1
            swap_t1[i+2*d_t,(i*d_t)+2]=1
            swap_t1[i+3*d_t,(i*d_t)+3]=1
        swap_t1= swap_t1.reshape(d_t,d_t,d_t,d_t)   
        return swap_t1
 
    def expectation(self,AList, MPO):
        sol = ncon([np.conj(AList),MPO,AList],[[1,2,4],[2,3],[1,3,4]])
        return sol


class observables:
    
    def __init__(self, operators=None):
        if operators is None:
            operators = basic_operators()  
        self.op = operators

    def TLS_pop(self,d_sys=2):
        return (self.op.sigmaplus() @ self.op.sigmaminus()).reshape(d_sys,d_sys)
        
    def a_R_pop(self,Deltat,d_t=2):
        return (self.op.DeltaBdagR(Deltat) @ self.op.DeltaBR(Deltat)).reshape(d_t*d_t,d_t*d_t)/Deltat

    def a_L_pop(self,Deltat,d_t=2):  
        return (self.op.DeltaBdagL(Deltat) @ self.op.DeltaBL(Deltat)).reshape(d_t*d_t,d_t*d_t)/Deltat
    
    