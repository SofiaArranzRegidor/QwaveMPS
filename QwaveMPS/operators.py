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
        """
        Raising operator for the Pauli spins.

        Returns
        -------
        oper : ndarray
            ndarray for the Pauli spin raising operator.
        
        Examples
        -------- 
        """
        a = np.zeros((2,2),dtype=complex)
        a[1,0]=1.       
        return a
    
    def sigmaminus(self):  
        """
        Lowering operator for the Pauli spins.

        Returns
        -------
        oper : ndarray
            ndarray for the Pauli spin lowering operator.
        
        Examples
        -------- 
        """
        a = np.zeros((2,2),dtype=complex)
        a[0,1]=1.       
        return a
    
    def DeltaBdag(self,Deltat):  
        """
        Time bin noise creation (raising) operator.

        Parameters
        ----------
        Deltat : float
            Time step for system evolution.

        Returns
        -------
        oper : ndarray
            ndarray time bin noise creation operator.
        
        Examples
        -------- 
        """        
        a = np.zeros((2,2),dtype=complex)
        a[1,0]=np.sqrt(Deltat)
        return a
    
    def DeltaB(self,Deltat):  
        """
        Time bin noise annihilation (lowering) operator.

        Parameters
        ----------
        Deltat : float
            Time step for system evolution.

        Returns
        -------
        oper : ndarray
            ndarray time bin noise creation operator.
        
        Examples
        -------- 
        """         
        a = np.zeros((2,2),dtype=complex)
        a[0,1]=np.sqrt(Deltat)
        return a
    
    def DeltaBdagL(self,Deltat):  
        """
        Left time bin noise creation (raising) operator for a system with two channels of light, left and right moving.

        Parameters
        ----------
        Deltat : float
            Time step for system evolution.

        Returns
        -------
        oper : ndarray
            ndarray left time bin noise creation operator.
        
        Examples
        -------- 
        """ 
        a=np.kron(np.sqrt(Deltat)*self.sigmaplus(),np.eye(2))     
        return a
    
    def DeltaBdagR(self,Deltat): 
        """
        Right time bin noise creation (raising) operator for a system with two channels of light, left and right moving.

        Parameters
        ----------
        Deltat : float
            Time step for system evolution.

        Returns
        -------
        oper : ndarray
            ndarray right time bin noise creation operator.
        
        Examples
        -------- 
        """ 
        a=np.kron(np.eye(2),np.sqrt(Deltat)*self.sigmaplus())     
        return a
    
    def DeltaBL(self,Deltat):  
        """
        Left time bin noise annihilation (lowering) operator for a system with two channels of light, left and right moving.

        Parameters
        ----------
        Deltat : float
            Time step for system evolution.

        Returns
        -------
        oper : ndarray
            ndarray left time bin noise annihilation operator.
        
        Examples
        -------- 
        """ 
        a=np.kron(np.sqrt(Deltat)*self.sigmaminus(),np.eye(2))     
        return a
    
    def DeltaBR(self,Deltat):  
        """
        Right time bin noise annihilation (lowering) operator for a system with two channels of light, left and right moving.

        Parameters
        ----------
        Deltat : float
            Time step for system evolution.

        Returns
        -------
        oper : ndarray
            ndarray right time bin noise annihilation operator.
        
        Examples
        -------- 
        """ 
        a=np.kron(np.eye(2),np.sqrt(Deltat)*self.sigmaminus())       
        return a
    
    def e(self,d_sys=2):
        """
        <Description>

        Parameters
        ----------
        d_sys : float
            Size of the Hilbert space of the matter system.

        Returns
        -------
        oper : ndarray
            ndarray <What it is>.
        
        Examples
        -------- 
        """ 
        exc = np.zeros((d_sys,d_sys),dtype=complex)
        exc[1,1]=1.      
        return exc

    def U(self,Hm,d_sys=2,d_t=2):
        """
        Time evolution operator in the case of Markovian dynamics.

        Parameters
        ----------
        Hm : ndarray
            Hamiltonian of the system.

        d_sys : int, default: 2
            Size of the Hilbert space of the matter system.
        
        d_t : int, default: 2
            Size of the truncated Hilbert space of the light field.

        Returns
        -------
        oper : ndarray
            ndarray time evolution operator.
        
        Examples
        -------- 
        """ 
        sol= expm(-Hm)
        return sol.reshape(d_sys,d_t*d_t,d_sys,d_t*d_t)

    def U_NM(self,Hm,d_t,d_sys):
        """
        Time evolution operator in the case of non-Markovian dynamics.

        Parameters
        ----------
        Hm : ndarray
            Hamiltonian of the system.

        d_sys : int, default: 2
            Size of the Hilbert space of the matter system.
        
        d_t : int, default: 2
            Size of the truncated Hilbert space of the light field.

        Returns
        -------
        oper : ndarray
            ndarray time evolution for non-Markovian operator.
        
        Examples
        -------- 
        """         
        sol = expm(-Hm)
        return sol.reshape(d_sys,d_t,d_t,d_sys,d_t,d_t)
    
    def swap(self,dim1,dim2):
        """
        Swap tensor to swap the contents of adjacent MPS bins.

        Parameters
        ----------
        dim1 : int
            Size of the first Hilbert space.
                
        dim2 : int, default: 2
            Size of the second Hilbert space.

        Returns
        -------
        oper : ndarray
            ndarray of shape (dim1,dim2,dim1,dim2) swap operator.
        
        Examples
        -------- 
        """ 
        size = dim1 * dim2
        swap = np.zeros([size,size],dtype=complex)
        for i in range(dim1):
            for j in range(dim2):
                swap[i + j*dim1,(i*dim2)+j]=1
        return swap.reshape(dim1,dim2,dim1,dim2)   
     
    # I think slightly less performant for small dims, but slightly faster for large dims, could just remove
    # Reduces to single explicit python loop
    def vectorizedswap(self,dim1,dim2):
        """
        Swap tensor to swap the contents of adjacent MPS bins.

        Parameters
        ----------
        dim1 : int
            Size of the first Hilbert space.
                
        dim2 : int, default: 2
            Size of the second Hilbert space.

        Returns
        -------
        oper : ndarray
            ndarray of shape (dim1,dim2,dim1,dim2) swap operator.
        
        Examples
        -------- 
        """ 
        size = dim1*dim2
        swap = np.zeros((size, size), dtype=complex)
        indices = np.array([(i%dim2)*dim1 + int(i/dim2) for i in range(size)], dtype=int)
        swap[indices, np.arange(swap.shape[0])] = 1
        return swap.reshape(dim1,dim2,dim1,dim2)


    def expectation(self,AList, MPO):
        """
        The expectation value of a MPS bin with a given operator.

        Parameters
        ----------
        AList : ndarray
            MPS bin defining the state having an expectation taken with respect to some operator.

        MPO : ndarray
            Operator whose expectation value is being taken.
        
        Returns
        -------
        expectation value : float
            The expectation value of the operator for the given state.
        
        Examples
        -------- 
        """ 
        sol = ncon([np.conj(AList),MPO,AList],[[1,2,4],[2,3],[1,3,4]])
        return sol


class observables:
    
    def __init__(self, operators=None):
        if operators is None:
            operators = basic_operators()  
        self.op = operators

    def TLS_pop(self,d_sys=2):    
        return (self.op.sigmaplus() @ self.op.sigmaminus())
        
    def a_R_pop(self,Deltat,d_t=2):
        return (self.op.DeltaBdagR(Deltat) @ self.op.DeltaBR(Deltat))/Deltat

    def a_L_pop(self,Deltat,d_t=2):  
        return (self.op.DeltaBdagL(Deltat) @ self.op.DeltaBL(Deltat))/Deltat
    
    def a_pop(self,Deltat,d_t=2):  
        return (self.op.DeltaBdag(Deltat) @ self.op.DeltaB(Deltat))/Deltat