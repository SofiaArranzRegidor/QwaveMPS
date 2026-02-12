"""
High level tests to verify dynamics with a single TLS
"""
#%%
import pytest
import QwaveMPS as qmps
import numpy as np
from .benchmarks import *

@pytest.mark.parametrize("gamma_l, gamma_r, initial_pop", 
                          [(0.5,0.5,1), (0,1,1), (0.5,0.5,0.5), (0,1,0.5)])
def test_single_TLS(gamma_l, gamma_r, initial_pop):
    input_params = qmps.parameters.InputParams(
        delta_t=0.02, 
        tmax = 6,
        d_sys_total=np.array([2]),
        d_t_total=np.array([2,2]),
        gamma_l=gamma_l,
        gamma_r = gamma_r,  
        bond_max=4,
    )


    tmax=input_params.tmax
    delta_t=input_params.delta_t
    tlist=np.arange(0,tmax+delta_t,delta_t)

    sys_initial_state = np.zeros((1,2,1))
    sys_initial_state[:,0,:] = np.sqrt(1 - initial_pop)
    sys_initial_state[:,1,:] = np.sqrt(initial_pop)
    wg_initial_state = None

    Hm=qmps.hamiltonian_1tls(input_params)
    bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)


    # Calculate the two level system population
    photon_pop_ops = [qmps.b_pop_l(input_params), qmps.b_pop_r(input_params)]
    tls_pop = np.real(qmps.single_time_expectation(bins.system_states, qmps.tls_pop()))
    photon_fluxes = np.real(qmps.single_time_expectation(bins.output_field_states, photon_pop_ops))
    total_quanta = tls_pop + np.cumsum(np.sum(photon_fluxes, axis=0)) * delta_t

    N = 0; anal_pulse_env = np.zeros(len(tlist))
    if gamma_l == 0:
        chiral = True
    else:
        chiral = False

    anal_pops = sigmaPlusSigmaMinus0N0N(tlist, anal_pulse_env, N, initial_pop, chirality=chiral)
    anal_flux_l = photonFluxMu(tlist, anal_pulse_env,N,'L', initial_pop, chirality=chiral)
    anal_flux_r = photonFluxMu(tlist, anal_pulse_env,N,'R', initial_pop, chirality=chiral)
    
    assert check_close(total_quanta, np.ones(len(total_quanta))*initial_pop)
    assert check_close(tls_pop, anal_pops)

    # Exclude initial condition of vacuum from flux
    assert check_close(photon_fluxes[0][1:], anal_flux_l[1:])
    assert check_close(photon_fluxes[1][1:], anal_flux_r[1:])

# %%
