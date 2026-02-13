"""
Analytical/numerical benchmarks used to compare dynamics to that produced by MPS
"""

import numpy as np
import scipy as sci
import QwaveMPS as qmps

# Closeness check configuration
atol = 1e-7
rtol = 2e-1 # Allow 20% error

def check_close(A, B):
    return np.allclose(A, B, atol=atol, rtol=rtol)

# Helpful function for determining tolerances (check relative error)
def max_error(A,B):
    max_val = np.max(np.abs((A-B)/B))
    max_ind = np.argmax(np.abs((A-B)/B))
    return max_val, max_ind

# Find indices where two arrays are not considered equal
def find_failures(A,B):
    return np.where(~np.isclose(A,B, atol=atol, rtol=rtol))[0]

# Get list of absolute and relative errors where two arrays are not equal
def errs(A,B):
    fails = find_failures(A,B)
    abs_err = np.abs(A[fails] - B[fails])
    rel_err = np.abs((A[fails] - B[fails]) / B[fails])

    return list(zip(abs_err, rel_err)) 

# Print out of details regarding the errors between two arrays
def err_printout(A,B, ts):
    fails = find_failures(A, B)
    print("Error time points:", ts[fails])
    values_at_errors = list(zip(A[fails], B[fails]))
    errors = errs(A, B)
    print("t", "Value of A", "Value of B", "Absolute Error", "Relative Error")
    for i in range(len(errors)):
        print(f"{ts[fails[i]]:.2f}, {values_at_errors[i][0]:.2e},{values_at_errors[i][1]:.2e},  \
              {errors[i][0]:.2e}, {errors[i][1]:.3f}")


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


def test_fock_pulse(photon_num, initial_pop, gaussian_env):
    input_params = qmps.parameters.InputParams(
        delta_t=0.05, 
        tmax = 8,
        d_sys_total=np.array([2]),
        d_t_total=np.array([photon_num+1,photon_num+1]),
        gamma_l=0.5,
        gamma_r = 0.5,  
        bond_max=min(32,2**(photon_num+1)),
    )


    tmax=input_params.tmax
    delta_t=input_params.delta_t
    tlist=np.arange(0,tmax+delta_t,delta_t)

    sys_initial_state = np.zeros((1,2,1))
    sys_initial_state[:,0,:] = np.sqrt(1 - initial_pop)
    sys_initial_state[:,1,:] = np.sqrt(initial_pop)
    wg_initial_state = None


    if gaussian_env:
        pulse_center = 3
        sigma = 1
        pulse_time = input_params.tmax
        pulse_env = qmps.gaussian_envelope(pulse_time, input_params, sigma, pulse_center)
        anal_env = lambda t: gaussian_square_normed(t, sigma, pulse_center)
    else:
        pulse_time = 1
        pulse_env = qmps.tophat_envelope(pulse_time, input_params)
        anal_env = lambda t: tophat(t, pulse_time)

    wg_initial_state = qmps.fock_pulse(pulse_env,pulse_time, photon_num, input_params)

    Hm=qmps.hamiltonian_1tls(input_params)
    bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)


    # Calculate the two level system population
    photon_pop_ops = [qmps.b_pop_l(input_params), qmps.b_pop_r(input_params)]

    same_time_corrs = [qmps.b_pop_r(input_params)]
    for i in range(photon_num+1):
        same_time_corrs.append(qmps.b_dag_r(input_params) @ same_time_corrs[-1] @ qmps.b_r(input_params))

    photon_fluxes = np.real(qmps.single_time_expectation(bins.output_field_states, photon_pop_ops))
    tls_pop = np.real(qmps.single_time_expectation(bins.system_states, qmps.tls_pop()))

    total_quanta = tls_pop + np.cumsum(np.sum(photon_fluxes, axis=0)) * delta_t
    same_time_corrs = np.real(qmps.single_time_expectation(bins.input_field_states, same_time_corrs))

    chiral = False
    pulse_env = np.append(pulse_env, np.zeros(len(tlist) - len(pulse_env)))
    pulse_env = qmps.normalize_pulse_envelope(input_params.delta_t, pulse_env)
    anal_pops = sigmaPlusSigmaMinus0N0N(tlist, pulse_env, photon_num, initial_pop, chirality=chiral)
    anal_flux_l = photonFluxMu(tlist, pulse_env,photon_num,'L', initial_pop, chirality=chiral)
    anal_flux_r = photonFluxMu(tlist, pulse_env,photon_num,'R', initial_pop, chirality=chiral)

    assert check_close(total_quanta, np.cumsum(same_time_corrs[0])*delta_t + initial_pop)
    assert check_close(tls_pop, anal_pops)

    # Check fluxes and input state correlations
    # Exclude initial condition of vacuum from flux (and shift by 1 to right to line up by dt)
    import matplotlib.pyplot as plt
    plt.plot(tlist[:-1], photon_fluxes[1][1:])
    plt.plot(tlist[:-1], anal_flux_r[:-1])
    plt.show()
    err_printout(photon_fluxes[1][1:],anal_flux_r[:-1], tlist[:-1])
    assert check_close(photon_fluxes[0][1:], anal_flux_l[:-1])
    assert check_close(photon_fluxes[1][1:], anal_flux_r[:-1])

    for i in range(photon_num+1):
        anal_result = anal_same_time_correlation(tlist, photon_num, i+1, anal_env)
        assert check_close(same_time_corrs[i][1:], anal_result[:-1])

