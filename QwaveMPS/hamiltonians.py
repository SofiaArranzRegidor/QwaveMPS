#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains the hamiltonians for different cases

"""

import numpy as np
from .operators import basic_operators

op=basic_operators()


def Hamiltonian_1TLS(Deltat,gammaL,gammaR,d_sys=2,d_t=2,Omega=0,Delta=0):
    '''
    Hamilltonian for 1 TLS in the waveguide
    Deltat is the timestep
    gammaL and gammaR are the left and right decay rates 
    d is the TLS bin dimension (2 by default)
    Omega is a possible classical pump (turned off by default)
    Delta is the detuning between the pump and TLS  (turned off by default)
    '''
    M=1j*(Omega*Deltat*(np.kron(op.sigmaplus(),np.eye(d_t*d_t)) + np.kron(op.sigmaminus(),np.eye(d_t*d_t))) \
          + np.sqrt(gammaL)*(np.kron(op.sigmaplus(),op.DeltaBL(Deltat)) + np.kron(op.sigmaminus(),op.DeltaBdagL(Deltat))) \
          + np.sqrt(gammaR)*(np.kron(op.sigmaplus(),op.DeltaBR(Deltat)) + np.kron(op.sigmaminus(),op.DeltaBdagR(Deltat))) \
          +Deltat*Delta* np.kron(op.e1(),np.eye(d_t*d_t)) \
          ).reshape(d_sys,d_t*d_t,d_sys,d_t*d_t)
    return M

    



