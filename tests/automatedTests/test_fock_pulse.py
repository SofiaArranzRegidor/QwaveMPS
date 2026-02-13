"""
High level tests to verify dynamics with a single TLS
"""
#%%
import pytest
import QwaveMPS as qmps
import numpy as np
from .benchmarks import *
import itertools

photon_nums = [1,3,5]
initial_pops = [0,0.5]
gaussian_env = [True]
initial_test_conditions = list(itertools.product(photon_nums, initial_pops, gaussian_env))

# Doesn't work well for tophat, needs tighter numerics/looser tolerances
@pytest.mark.parametrize("photon_num, initial_pop, gaussian_env", initial_test_conditions)
def test_single_TLS(photon_num, initial_pop, gaussian_env):
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
    assert check_close(photon_fluxes[0][1:], anal_flux_l[:-1])
    assert check_close(photon_fluxes[1][1:], anal_flux_r[:-1])

    for i in range(photon_num+1):
        anal_result = anal_same_time_correlation(tlist, photon_num, i+1, anal_env)
        assert check_close(same_time_corrs[i][1:], anal_result[:-1])


# %%
