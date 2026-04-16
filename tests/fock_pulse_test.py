#%% 
# Imports
#--------

import QwaveMPS as qmps
import matplotlib.pyplot as plt
import numpy as np
import time as t
import scipy as sci

#%% Functions used for testing

#%%% Input output theory for TLS population and Fluxes
# M = N-1, N >= 1
def sigmaPlus0N0Nmin1(tList, pulseEnv, N, initialPop=0, zeta=-0.5, gamma=1, nInR=1, chirality=False):
    chiralGamma = gamma / (1 + int(not chirality))
    
    integrand = np.exp(-zeta * tList) * np.conj(pulseEnv) * (1 - 2*sigmaPlusSigmaMinus0N0N(tList, pulseEnv, N-1, initialPop, zeta, gamma, nInR, chirality))
    return -np.sqrt(N*chiralGamma*nInR) * np.exp(zeta * tList) *\
        sci.integrate.cumulative_trapezoid(integrand, dx=tList[1] - tList[0], initial=0) #+\
        #np.sqrt(initialPop*(1-initialPop))*np.exp(zeta*tList) # I.C.

def sigmaPlusSigmaMinus0N0N(tList, pulseEnv, N, initialPop=0, zeta=-0.5, gamma=1, nInR=1, chirality=False):
    chiralGamma = gamma / (1 + int(not chirality))
    # I.c.
    if N == 0:
        return initialPop * np.exp(-gamma * tList)
        
    integrand = np.exp(gamma * tList) * pulseEnv * sigmaPlus0N0Nmin1(tList, pulseEnv, N, initialPop, zeta, gamma, nInR, chirality)
    return -np.exp(-gamma * tList) * np.sqrt(N*chiralGamma*nInR) *\
        sci.integrate.cumulative_trapezoid(integrand + np.conj(integrand), dx=tList[1] - tList[0], initial=0)+\
        initialPop*np.exp(-gamma*tList) # I.c. addition

def photonFluxMu(tList, pulseEnv, N, mu, initialPop=0, zeta=-0.5, gamma=1, nInR=1, chirality=False):
    # Assuming symmetric coupling
    chiralGamma = gamma / (1 + int(not chirality))
    gammaMu = 1.0/2
    
    if str(mu).upper() in {'L','0'}:
        muIndex = 0
    else:
        muIndex = 1
    
    term1 = muIndex * N * np.conj(pulseEnv) * pulseEnv
    term3 = muIndex * np.sqrt(N*chiralGamma) * pulseEnv * sigmaPlus0N0Nmin1(tList, pulseEnv, N, initialPop, zeta, gamma, nInR, chirality)
    term2 = np.conj(term3)
    term4 = chiralGamma * sigmaPlusSigmaMinus0N0N(tList, pulseEnv, N, initialPop, zeta, gamma, nInR, chirality)
    return term1 + term2 + term3 + term4

#%%% Scattering theory Analysis of input state
# Evaluate <N|(a^\dag)^m(t) a^m(t)|N> for the input state (before TLS interaction)
def anal_same_time_correlation(t, photon_num, m, pulse_func, w_max=200, dw=0.0001):
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

# Overwrite with normed for square of Gaussian envelope
def gaussian(t, sigma, mu):
    return np.exp(-(t-mu)**2 / (2*sigma**2)) / np.sqrt(sigma * np.sqrt(np.pi))

def tophat(t, pulse_time):
    t = np.asarray(t)
    y = np.zeros_like(t, dtype=float)
    y[t <= pulse_time] = np.sqrt(1 / pulse_time)
    return y

#%% Setup for the loops
gaussian_pulse = True

""""Choose the simulation parameters"""
#Define input parameters
input_params = qmps.parameters.InputParams(
    delta_t=0.02,
    tmax = 8,
    d_sys_total=np.array([2]),
    d_t_total=np.array([0,0]),
    gamma_l=0.5,
    gamma_r = 0.5,  
    bond_max=16
)

#Make a tlist for plots:
tmax=input_params.tmax
delta_t=input_params.delta_t
tlist=np.arange(0,tmax+delta_t,delta_t)


""" Choose the initial state and tophat pulse parameters"""
initial_pop = 0.5
ground_state_pop = 1- initial_pop
sys_initial_state=np.zeros((1,2,1))
sys_initial_state[:,0,:] = np.sqrt(ground_state_pop)
sys_initial_state[:,1,:] = np.sqrt(initial_pop)



# Pulse parameters for a 2-photon gaussian pulse
if gaussian_pulse:
    gaussian_center = 3
    gaussian_width = 1

    pulse_time = tmax
    pulse_envelope = qmps.states.gaussian_envelope(pulse_time, input_params, gaussian_width, gaussian_center)
    anal_env = lambda t: gaussian(t, gaussian_width, gaussian_center)
else:
    pulse_time = 1
    pulse_envelope = qmps.states.tophat_envelope(pulse_time, input_params)
    anal_env = lambda t: tophat(t, pulse_time)

#%% Execute the loops
N = 3
for i in range(1,N+1):
    photon_num = i
    input_params.d_t_total = np.array([i+1, i+1])

    wg_initial_state = qmps.states.fock_pulse(pulse_envelope,pulse_time, photon_num, input_params, direction='R')

    start_time=t.time()

    #Calculate the time evolution
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    """Calculate time evolution of the system"""
    # Create the Hamiltonian again for this larger Hilbert space
    #Hm=np.zeros((np.kron(qmps.b_pop_l(input_params), qmps.tls_pop())).shape)
    Hm = qmps.hamiltonian_1tls(input_params)

    bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)


    #Calculate the population dynamics
    #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    """Calculate population dynamics"""
    photon_flux_ops = [qmps.b_pop_l(input_params), qmps.b_pop_r(input_params)]

    # Calculate same time G2 in transmission
    same_time_corrs = [qmps.b_pop_r(input_params)]
    for i in range(N+1):
        same_time_corrs.append(qmps.b_dag_r(input_params) @ same_time_corrs[-1] @ qmps.b_r(input_params))

    tls_pop = qmps.single_time_expectation(bins.system_states, qmps.tls_pop())
    photon_fluxes = qmps.single_time_expectation(bins.output_field_states, photon_flux_ops)
    same_time_corrs = qmps.single_time_expectation(bins.input_field_states, same_time_corrs)

    # Act on input states to characterize the input field with bosonic/field operators
    flux_in = qmps.single_time_expectation(bins.input_field_states, photon_flux_ops)


    total_quanta = tls_pop + np.cumsum(photon_fluxes[0] + photon_fluxes[1]) * delta_t

    print("%d-photon pops--- %s seconds ---" %(photon_num, (t.time() - start_time)))
    #^^^^^^^^^^^^^^^^
    #Plot the results
    #^^^^^^^^^^^^^^^^
    # First Plot Input/Output Theory results
    anal_dt = 0.001
    anal_ts = np.arange(0, input_params.tmax, anal_dt)

    #'''
    plt.plot(tlist,np.real(photon_fluxes[1]),linewidth = 3,color = 'violet',linestyle='-',label=r'$n_{R}$') # Photons transmitted to the right channel
    plt.plot(tlist,np.real(photon_fluxes[0]),linewidth = 3,color = 'green',linestyle='--',label=r'$n_{L}$') # Photons reflected to the left channel
    plt.plot(tlist,np.real(tls_pop),linewidth = 3, color = 'k',linestyle='-',label=r'$n_{TLS}$') # TLS population
    plt.plot(tlist,np.real(total_quanta),linewidth = 3,color = 'g',linestyle='-',label='Total') # Conservation check (for one excitation it should be 1)

    # Analytical plots
    anal_pulse_env = qmps.normalize_pulse_envelope(anal_dt, anal_env(anal_ts))
    
    anal_pops = sigmaPlusSigmaMinus0N0N(anal_ts, anal_pulse_env, photon_num, initialPop=initial_pop)
    anal_flux_l = photonFluxMu(anal_ts, anal_pulse_env, photon_num, 'L', initialPop=initial_pop)
    anal_flux_r = photonFluxMu(anal_ts, anal_pulse_env, photon_num, 'R', initialPop=initial_pop)


    plt.plot(anal_ts,np.real(anal_flux_r),linewidth = 3,color = 'brown',linestyle=':',label=r'$n_{R}$') # Photons transmitted to the right channel
    plt.plot(anal_ts,np.real(anal_flux_l),linewidth = 3,color = 'cyan',linestyle=':',label=r'$n_{L}$') # Photons reflected to the left channel
    plt.plot(anal_ts,np.real(anal_pops),linewidth = 3, color = 'pink',linestyle=':',label=r'$n_{TLS}$') # TLS population
    #plt.plot(tlist,np.real(total_quanta),linewidth = 3,color = 'g',linestyle='-',label='Total') # Conservation check (for one excitation it should be 1)

    plt.legend(ncol=2)
    plt.xlabel(r'Time, $\gamma t$')
    plt.ylabel('Populations')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim([0.,None])
    plt.xlim([0.,tmax])
    plt.show()
    #'''

    # Second Plot: Input State Characterization

    # Plot input pulse direct characterization of same time correlations to order N
    for i in range(photon_num+1):
        plt.plot(tlist, np.real(same_time_corrs[i]), linewidth=3, linestyle='-', label=r'$G_{RR}^{(%s)}$' % (i+1))
    plt.plot(tlist,np.real(total_quanta),linewidth = 3,color = 'g',linestyle='-',label='Total') # Conservation check (for one excitation it should be 1)

    # Analytical plots
    for i in range(1,photon_num+1):
        anal_result = anal_same_time_correlation(anal_ts, photon_num, i, anal_env)
        print('Correction Factor:', np.real(max(same_time_corrs[i-1]) / max(anal_result)))
        plt.plot(anal_ts, np.real(anal_result), linewidth=3, linestyle=':', label=r'$G_{RR}^{(%s)}$' % (i))
    plt.plot(anal_ts, np.zeros(len(anal_ts)), linewidth=3, linestyle=':', label=r'$G_{RR}^{(%s)}$' % (photon_num+1))


    plt.legend(ncol=2)
    plt.xlabel(r'Time, $\gamma t$')
    plt.ylabel('Populations')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim([0.,None])
    plt.xlim([0.,tmax])
    plt.show()



    print('Correlation \t Value \t\t Integration')
    for i in range(photon_num+1):
        index= np.argmax(np.real(same_time_corrs[i]))
        print(f"{i+1} \t\t {np.real(same_time_corrs[i][index]):.8f} \t {np.real(np.sum(same_time_corrs[i]*delta_t)):.8f}")

    print('='*50)
