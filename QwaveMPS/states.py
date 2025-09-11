#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains initial states for the waveguide and the TLSs

"""

import numpy as np


class initial_state:

    def i_ng(d_t=2,bond0=1):
        """Waveguide initially in vacuum state"""
        i= np.zeros([bond0,d_t*d_t,bond0],dtype=complex) 
        i[0,0,0]=1.
        return i

    def i_sg(d_sys=2,bond0=1):
        "Atom initially in ground state"
        i_s = np.zeros([bond0,d_sys,bond0],dtype=complex) 
        i_s[:,0,:]=1.
        return i_s
        
    def i_se(d_sys=2,bond0=1):
        "Atom initially excited"
        i_s = np.zeros([bond0,d_sys,bond0],dtype=complex) 
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