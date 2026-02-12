"""
Analytical/numerical benchmarks used to compare dynamics to that produced by MPS
"""

import numpy as np
import scipy as sci
import QwaveMPS as qmps

# Closeness check configuration
atol = 1e-7
rtol = 1e-1 # Allow 10% error

def check_close(A, B):
    return np.allclose(A, B, atol=atol, rtol=rtol)

# Helpful function for determining tolerances (check relative error)
def max_error(A,B):
    max_val = np.max(np.abs((A-B)/B))
    max_ind = np.argmax(np.abs((A-B)/B))
    return max_val, max_ind

#%%% Input output theory for TLS population and Fluxes
# M = N-1, N >= 1
def sigmaPlus0N0Nmin1(tList, pulseEnv, N, initialPop=0, delta=0, gamma=1, nInR=1, chirality=False):
    if N < 1:
        return np.zeros(len(tList))

    zeta = -1j*delta - gamma/2
    chiralGamma = gamma / (1 + int(not chirality))
    
    integrand = np.exp(-zeta * tList) * np.conj(pulseEnv) * (1 - 2*sigmaPlusSigmaMinus0N0N(tList, pulseEnv, N-1, initialPop, delta, gamma, nInR, chirality))
    return -np.sqrt(N*chiralGamma*nInR) * np.exp(zeta * tList) *\
        sci.integrate.cumulative_trapezoid(integrand, dx=tList[1] - tList[0], initial=0) 

def sigmaPlusSigmaMinus0N0N(tList, pulseEnv, N, initialPop=0, delta=0, gamma=1, nInR=1, chirality=False):
    zeta = -1j*delta - gamma/2
    chiralGamma = gamma / (1 + int(not chirality))
    # I.c.
    if N == 0:
        return initialPop * np.exp(-gamma * tList)
        
    integrand = np.exp(gamma * tList) * pulseEnv * sigmaPlus0N0Nmin1(tList, pulseEnv, N, initialPop, delta, gamma, nInR, chirality)
    return -np.exp(-gamma * tList) * np.sqrt(N*chiralGamma*nInR) *\
        sci.integrate.cumulative_trapezoid(integrand + np.conj(integrand), dx=tList[1] - tList[0], initial=0)+\
        initialPop*np.exp(-gamma*tList) # I.c. addition

def photonFluxMu(tList, pulseEnv, N, mu, initialPop=0, delta=0, gamma=1, nInR=1, chirality=False):
    # Assuming symmetric coupling
    chiralGamma = gamma / (1 + int(not chirality))
    gammaMu = 1.0/2
    
    if str(mu).upper() in {'L','0'}:
        muIndex = 0
    else:
        muIndex = 1
    
    term1 = muIndex * N * np.conj(pulseEnv) * pulseEnv
    term3 = muIndex * np.sqrt(N*chiralGamma) * pulseEnv * sigmaPlus0N0Nmin1(tList, pulseEnv, N, initialPop, delta, gamma, nInR, chirality)
    term2 = np.conj(term3)
    if muIndex == 0 and chirality:
        term4 = np.zeros(len(tList))
    else:
        term4 = chiralGamma * sigmaPlusSigmaMinus0N0N(tList, pulseEnv, N, initialPop, delta, gamma, nInR, chirality)
    return term1 + term2 + term3 + term4

#%% Scattering theory Analysis of input state for same time correlations
# Evaluate <N|(a^\dag)^m(t) a^m(t)|N> for the input state (before TLS interaction)
def anal_same_time_correlation(t, photon_num, m, pulse_func, w_max=200, dw=0.0001):
    if m > photon_num:
        return np.zeros(len(t))
    
    prefactor = sci.special.factorial(photon_num) / sci.special.factorial(photon_num-m)

    sample_num = int(round(2*w_max / dw))
    delta_t = 2*np.pi / (sample_num * dw)

    fourier_ts = np.arange(0, sample_num) * delta_t
    ws = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(sample_num, d=delta_t))

    # Have factors for analytical fourier transform
    fourier_transform = np.fft.fftshift(np.fft.fft(pulse_func(fourier_ts))) * delta_t / (np.sqrt(2*np.pi))
    integral_result = np.sum(np.abs(fourier_transform)**2) * dw
    integral_factor = integral_result**(photon_num-m-photon_num/2)


    return prefactor * integral_factor * pulse_func(t)**(2*m)

#%% Pulse envelopes for the analytical checks
def gaussian(t, sigma, mu):
    return np.exp(-(t-mu)**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

def gaussian_square_normed(t, sigma, mu):
    return np.exp(-(t-mu)**2 / (2*sigma**2)) / np.sqrt(sigma * np.sqrt(np.pi))

def tophat(t, pulse_time):
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    y[t <= pulse_time] = np.sqrt(1 / pulse_time)
    return y
