#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains initial states for the waveguide and the TLSs

"""

import numpy as np


class initial_state:

    def i_ng(d_t,bond0=1):
        """
        Waveguide vacuum state.

        Parameters
        ----------
        d_t : int
            Size of the truncated Hilbert space of the light field.

        bond0 : int, default: 1
            Size of the bond dimension.
        
        Returns
        -------
        state : ndarray
            ndarray vacuum state.
        
        Examples
        -------- 
        """ 
        i= np.zeros([bond0,d_t,bond0],dtype=complex) 
        i[:,0,:]=1.
        return i

    def i_sg(d_sys1=2,bond0=1):
        """
        TLS ground state.

        Parameters
        ----------
        d_sys1 : int, default: 2
            Size of the Hilbert space of the matter system.

        bond0 : int, default: 1
            Size of the bond dimension.
        
        Returns
        -------
        state : ndarray
            ndarray ground state atom.
        
        Examples
        -------- 
        """ 
        i_s = np.zeros([bond0,d_sys1,bond0],dtype=complex) 
        i_s[:,0,:]=1.
        return i_s
        
    def i_se(d_sys1=2,bond0=1):
        """
        TLS excited state.

        Parameters
        ----------
        d_sys1 : int, default: 2
            Size of the Hilbert space of the matter system.

        bond0 : int, default: 1
            Size of the bond dimension.
        
        Returns
        -------
        state : ndarray
            ndarray excited state atom.
        
        Examples
        -------- 
        """ 
        i_s = np.zeros([bond0,d_sys1,bond0],dtype=complex) 
        i_s[:,1,:]=1.
        return i_s

def coupling(coupl='symmetrical',gamma=1):
    """ Coupling can be chiral or symmetrical.
    Symmetrical by default."""   
    if coupl == 'chiral': 
        gammaR=gamma
        gammaL=gamma - gammaR
    if coupl == 'symmetrical':
        gammaR=gamma/2.
        gammaL=gamma - gammaR
    return gammaL,gammaR       