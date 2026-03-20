#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performs performance tests of the library

Could use a similar structure likely for automated testing (though can likely collapse a few together)
"""

#%% Imports
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FuncFormatter
import numpy as np

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import QwaveMPS.src as qmps
import timeit

#%% Test set 1, population dynamics
#Set relevant parameters of timing
runs = 50
repeat = 5
#%%% 1 TLS, Markovian 

#Choose the coupling:
gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1) # same as gamma_l, gamma_r = (0.5,0.5)
#Define input parameters (dataclass)
input_params = qmps.parameters.InputParams(
    delta_t=0.05, # Time step of the simulation
    tmax = 8, # Maximum simulation time
    d_sys_total=[2],
    d_t_total=[2,2],
    gamma_l=gamma_l,
    gamma_r = gamma_r,  
    bond_max=4 # Maximum bond dimension, simulation parameter that adjusts truncation of entanglement information
)
#Make a tlist for plots
tlist=np.arange(0,input_params.tmax+input_params.delta_t/2,input_params.delta_t)

""" Choose the initial state"""

sys_initial_state=qmps.states.tls_excited() #TLS initially excited
wg_initial_state = None
Hm=qmps.hamiltonian_1tls(input_params) # Create the Hamiltonian for a single TLS

tls_pop_op = qmps.tls_pop() 
photon_pop_ops = [qmps.b_pop_l(input_params), qmps.b_pop_r(input_params)]

def timedProcess():
    bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)
    tls_pop = qmps.single_time_expectation(bins.system_states, tls_pop_op)
    photon_fluxes = qmps.single_time_expectation(bins.output_field_states, photon_pop_ops)
    net_flux_l = np.cumsum(photon_fluxes[0]) * input_params.delta_t
    net_flux_r = np.cumsum(photon_fluxes[1]) * input_params.delta_t
    total_quanta = tls_pop + net_flux_l + net_flux_r


times = timeit.repeat(timedProcess, repeat=repeat, number=runs)

print('1 TLS, Markovian time:', min(times)/runs)

#%%% 1TLS, Nonmarkovian
#Choose the coupling
gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1)
#Define input parameters
input_params = qmps.parameters.InputParams(
    delta_t=0.05, # simulation time step
    tmax = 5, # simulation total time length
    d_sys_total=[2],
    d_t_total=[2],
    gamma_l=gamma_l,
    gamma_r = gamma_r,  
    bond_max=4, #simulation maximum MPS bond dimension, truncates entanglement information
    tau=1, # Roundtrip feedback time
    phase=np.pi
)
tlist=np.arange(0,input_params.tmax+input_params.delta_t/2,input_params.delta_t)

sys_initial_state=qmps.states.tls_excited()
wg_initial_state = None

Hm=qmps.hamiltonian_1tls_feedback(input_params)

flux_op = qmps.b_pop(input_params)
tls_pop_op = qmps.tls_pop()

""" Time evolution of the system"""
def timedProcess():
    bins = qmps.t_evol_nmar(Hm,sys_initial_state,wg_initial_state,input_params)
    tls_pops = qmps.single_time_expectation(bins.system_states, tls_pop_op)
    transmitted_flux = qmps.single_time_expectation(bins.output_field_states, flux_op)
    net_transmitted_quanta = np.cumsum(transmitted_flux) * input_params.delta_t
    
    loop_flux = qmps.single_time_expectation(bins.loop_field_states, flux_op)
    loop_sum = qmps.loop_integrated_statistics(loop_flux, input_params)
    total_quanta = tls_pops + loop_sum + net_transmitted_quanta


times = timeit.repeat(timedProcess, repeat=repeat, number=runs)
print('1 TLS, Non-Markovian time:', min(times)/runs)

#%%% 2TLS, Markovian

#Choose the coupling for each TLS:
gamma_l1,gamma_r1=qmps.coupling('symmetrical',gamma=1)
gamma_l2,gamma_r2=qmps.coupling('symmetrical',gamma=1)
#Define input parameters
input_params = qmps.parameters.InputParams(
    delta_t=0.05, # Simulation time step
    tmax = 5, # Maximum simulation time
    d_sys_total=[2,2],
    d_t_total=[2,2],

    # Couplings
    gamma_l=gamma_l1,
    gamma_r = gamma_r1,
    gamma_l2 = gamma_l2,
    gamma_r2 = gamma_r2,

    bond_max=4, # Maximum MPS bond dimension, sets truncation of entanglement
    phase=np.pi # Phase of interaction between the 2 TLS's
)
tlist=np.arange(0,input_params.tmax+input_params.delta_t/2,input_params.delta_t)

""" Choose the initial state"""
tls1_initial_state=qmps.states.tls_excited()
tls2_initial_state= qmps.states.tls_ground()
sys_initial_state=np.kron(tls1_initial_state,tls2_initial_state)
wg_initial_state = None
"""Choose the Hamiltonian"""
hm=qmps.hamiltonian_2tls_mar(input_params)

pop_tls1_op = np.kron(qmps.tls_pop(), np.eye(input_params.d_sys_total[1]))
pop_tls2_op = np.kron(np.eye(input_params.d_sys_total[0]), qmps.tls_pop())
pop_ops = [pop_tls1_op, pop_tls2_op]

flux_ops = [qmps.b_pop_l(input_params), qmps.b_pop_r(input_params)]

"""Calculate time evolution of the system"""
def timedProcess():
    bins = qmps.t_evol_mar(hm,sys_initial_state,wg_initial_state,input_params)
    tls_pops = qmps.single_time_expectation(bins.system_states, pop_ops)
    fluxes = qmps.single_time_expectation(bins.output_field_states, flux_ops)

    net_flux_l = np.cumsum(fluxes[0]) * input_params.delta_t
    net_flux_r = np.cumsum(fluxes[1]) * input_params.delta_t
    total_quanta = tls_pops[0] + tls_pops[1] + net_flux_l + net_flux_r

times = timeit.repeat(timedProcess, repeat=repeat, number=runs)
print('2 TLS, Markovian time:', min(times)/runs)

#%%% 2TLS, Nonmarkovian

#Choose the coupling:
gamma_l1,gamma_r1=qmps.coupling('symmetrical',gamma=1)
gamma_l2,gamma_r2=qmps.coupling('symmetrical',gamma=1)
#Define input parameters
#Need to define the delay time tau and phase
input_params = qmps.parameters.InputParams(
    delta_t=0.05,
    tmax = 5,
    d_sys_total=[2,2],
    d_t_total=[2,2],
    gamma_l=gamma_l1,
    gamma_r = gamma_r1,
    gamma_l2 = gamma_l2,
    gamma_r2 = gamma_r2,
    bond_max=8,
    phase=np.pi,
    tau=0.5 # Time delay between the two TLS's
)
tlist=np.arange(0,input_params.tmax+input_params.delta_t/2,input_params.delta_t)

""" Choose the initial state and coupling"""
tls1_initial_state=qmps.states.tls_excited()
tls2_initial_state= qmps.states.tls_ground()
sys_initial_state=np.kron(tls1_initial_state,tls2_initial_state)

wg_initial_state = None

"""Choose the Hamiltonian"""
hm=qmps.hamiltonian_2tls_nmar(input_params)



""" Calculate population dynamics"""
pop_tls1_op = np.kron(qmps.tls_pop(), np.eye(input_params.d_sys_total[1]))
pop_tls2_op = np.kron(np.eye(input_params.d_sys_total[0]), qmps.tls_pop())
pop_ops = [pop_tls1_op, pop_tls2_op]

flux_ops = [qmps.b_pop_l(input_params), qmps.b_pop_r(input_params)]

""" Time evolution of the system"""
def timedProcess():
    bins = qmps.t_evol_nmar(hm,sys_initial_state,wg_initial_state,input_params)


    # Calculate time dependent TLS populations, and fluxes into/out of feedback loop
    tls_pops = qmps.single_time_expectation(bins.system_states, pop_ops)
    photon_fluxes_out = qmps.single_time_expectation(bins.output_field_states, flux_ops)
    photon_fluxes_loop = qmps.single_time_expectation(bins.loop_field_states, flux_ops)

    # Use helper function to integrate over the flux into the loop in windows to get loop population
    loop_sum_l = qmps.loop_integrated_statistics(photon_fluxes_loop[0], input_params)
    loop_sum_r = qmps.loop_integrated_statistics(photon_fluxes_loop[1], input_params)

    net_flux_l = np.cumsum(photon_fluxes_out[0]) * input_params.delta_t
    net_flux_r = np.cumsum(photon_fluxes_out[1]) * input_params.delta_t
    # Sum the population of the 2 TLS's, the integral of the flux out of the total system, and the population
    #  in the loop between the TLS's.
    total_quanta = tls_pops[0] + tls_pops[1] + net_flux_l + net_flux_r + loop_sum_l + loop_sum_r

times = timeit.repeat(timedProcess, repeat=repeat, number=runs)
print('2 TLS, Non-Markovian time:', min(times)/runs)

#%% Test set 2: Pops, entropy, and flux
runs = 50
repeat = 5

#%%% 2TLS, Markovian Case

#Choose the coupling for each TLS:
gamma_l1,gamma_r1=qmps.coupling('symmetrical',gamma=1)
gamma_l2,gamma_r2=qmps.coupling('symmetrical',gamma=1)
#Define input parameters
input_params = qmps.parameters.InputParams(
    delta_t=0.05, # Simulation time step
    tmax = 5, # Maximum simulation time
    d_sys_total=[2,2],
    d_t_total=[2,2],

    # Couplings
    gamma_l=gamma_l1,
    gamma_r = gamma_r1,
    gamma_l2 = gamma_l2,
    gamma_r2 = gamma_r2,

    bond_max=4, # Maximum MPS bond dimension, sets truncation of entanglement
    phase=np.pi # Phase of interaction between the 2 TLS's
)
tlist=np.arange(0,input_params.tmax+input_params.delta_t/2,input_params.delta_t)

""" Choose the initial state"""
tls1_initial_state=qmps.states.tls_excited()
tls2_initial_state= qmps.states.tls_excited()
sys_initial_state=np.kron(tls1_initial_state,tls2_initial_state)
wg_initial_state = None
"""Choose the Hamiltonian"""
hm=qmps.hamiltonian_2tls_mar(input_params)

pop_tls1_op = np.kron(qmps.tls_pop(), np.eye(input_params.d_sys_total[1]))
pop_tls2_op = np.kron(np.eye(input_params.d_sys_total[0]), qmps.tls_pop())
pop_ops = [pop_tls1_op, pop_tls2_op]

flux_ops = [qmps.b_pop_l(input_params), qmps.b_pop_r(input_params)]

"""Calculate time evolution of the system"""
def timedProcess():
    bins = qmps.t_evol_mar(hm,sys_initial_state,wg_initial_state,input_params)
    tls_pops = qmps.single_time_expectation(bins.system_states, pop_ops)
    fluxes = qmps.single_time_expectation(bins.output_field_states, flux_ops)

    net_flux_l = np.cumsum(fluxes[0]) * input_params.delta_t
    net_flux_r = np.cumsum(fluxes[1]) * input_params.delta_t

times = timeit.repeat(timedProcess, repeat=repeat, number=runs)
print('2 TLS, Markovian population time:', min(times)/runs)

bins = qmps.t_evol_mar(hm,sys_initial_state,wg_initial_state,input_params)

b_pop_r = flux_ops[1]
def timedProcess():
    flux_r = qmps.single_time_expectation(bins.output_field_states, b_pop_r)
    entropy_sys = qmps.entanglement(bins.schmidt)

times = timeit.repeat(timedProcess, repeat=repeat, number=runs)
print('2 TLS, Markovian entropy/flux time:', min(times)/runs)


#%%% 2TLS, Nonmarkovian Case

#Choose the coupling:
gamma_l1,gamma_r1=qmps.coupling('symmetrical',gamma=1)
gamma_l2,gamma_r2=qmps.coupling('symmetrical',gamma=1)
#Define input parameters
#Need to define the delay time tau and phase
input_params = qmps.parameters.InputParams(
    delta_t=0.05,
    tmax = 5,
    d_sys_total=[2,2],
    d_t_total=[2,2],
    gamma_l=gamma_l1,
    gamma_r = gamma_r1,
    gamma_l2 = gamma_l2,
    gamma_r2 = gamma_r2,
    bond_max=8,
    phase=np.pi,
    tau=0.5 # Time delay between the two TLS's
)
tlist=np.arange(0,input_params.tmax+input_params.delta_t/2,input_params.delta_t)

""" Choose the initial state and coupling"""
tls1_initial_state=qmps.states.tls_excited()
tls2_initial_state= qmps.states.tls_excited()
sys_initial_state=np.kron(tls1_initial_state,tls2_initial_state)
wg_initial_state = None

"""Choose the Hamiltonian"""
hm=qmps.hamiltonian_2tls_nmar(input_params)

""" Calculate population dynamics"""
pop_tls1_op = np.kron(qmps.tls_pop(), np.eye(input_params.d_sys_total[1]))
pop_tls2_op = np.kron(np.eye(input_params.d_sys_total[0]), qmps.tls_pop())
pop_ops = [pop_tls1_op, pop_tls2_op]

flux_ops = [qmps.b_pop_l(input_params), qmps.b_pop_r(input_params)]

""" Time evolution of the system"""
def timedProcess():
    bins = qmps.t_evol_nmar(hm,sys_initial_state,wg_initial_state,input_params)


    # Calculate time dependent TLS populations, and fluxes into/out of feedback loop
    tls_pops = qmps.single_time_expectation(bins.system_states, pop_ops)
    photon_fluxes_out = qmps.single_time_expectation(bins.output_field_states, flux_ops)
    photon_fluxes_loop = qmps.single_time_expectation(bins.loop_field_states, flux_ops)

    # Use helper function to integrate over the flux into the loop in windows to get loop population
    loop_sum_l = qmps.loop_integrated_statistics(photon_fluxes_loop[0], input_params)
    loop_sum_r = qmps.loop_integrated_statistics(photon_fluxes_loop[1], input_params)

    net_flux_l = np.cumsum(photon_fluxes_out[0]) * input_params.delta_t
    net_flux_r = np.cumsum(photon_fluxes_out[1]) * input_params.delta_t

times = timeit.repeat(timedProcess, repeat=repeat, number=runs)
print('2 TLS, Non-Markovian pops time:', min(times)/runs)

bins = qmps.t_evol_nmar(hm,sys_initial_state,wg_initial_state,input_params)
b_pop_r = flux_ops[1]
def timedProcess():
    flux_r = qmps.single_time_expectation(bins.output_field_states, b_pop_r)
    flux_r_loop = qmps.single_time_expectation(bins.loop_field_states, b_pop_r)
    entropy_sys = qmps.entanglement(bins.schmidt)
    entropy_tau = qmps.entanglement(bins.schmidt_tau)

times = timeit.repeat(timedProcess, repeat=repeat, number=runs)
print('2 TLS, Non-Markovian entropy/flux time:', min(times)/runs)

#%% Test set 3, CW Pumping
#Set relevant parameters of timing
runs = 20
repeat = 5
#%%% 1 TLS, Markovian 

#Choose the coupling:
gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1) # same as gamma_l, gamma_r = (0.5,0.5)
#Define input parameters (dataclass)
input_params = qmps.parameters.InputParams(
    delta_t=0.05, # Time step of the simulation
    tmax = 40, # Maximum simulation time
    d_sys_total=[2],
    d_t_total=[2,2],
    gamma_l=gamma_l,
    gamma_r = gamma_r,  
    bond_max=18 # Maximum bond dimension, simulation parameter that adjusts truncation of entanglement information
)
#Make a tlist for plots
tlist=np.arange(0,input_params.tmax+input_params.delta_t/2,input_params.delta_t)

""" Choose the initial state"""

sys_initial_state=qmps.states.tls_ground() #TLS initially excited
wg_initial_state = None
cw_pump=2*np.pi

Hm=qmps.hamiltonian_1tls(input_params, cw_pump) # Create the Hamiltonian for a single TLS

tls_pop_op = qmps.tls_pop() 
photon_pop_ops = [qmps.b_pop_l(input_params), qmps.b_pop_r(input_params)]

# Pops timing
def timedProcess():
    bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)
    tls_pop = qmps.single_time_expectation(bins.system_states, tls_pop_op)
    photon_fluxes = qmps.single_time_expectation(bins.output_field_states, photon_pop_ops)

times = timeit.repeat(timedProcess, repeat=repeat, number=runs)
print('1 TLS, Markovian pops time:', min(times)/runs)

# Prep for timing
bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)
photon_fluxes = qmps.single_time_expectation(bins.output_field_states, photon_pop_ops)


ops_same_time = []; ops_two_time = []
b_dag_r = qmps.b_dag_r(input_params); b_dag_l = qmps.b_dag_l(input_params)
b_r = qmps.b_r(input_params); b_l = qmps.b_l(input_params)
b_pop_r = qmps.b_pop_r(input_params); b_pop_l = qmps.b_pop_l(input_params)

# G1's
ops_same_time.append(b_pop_r); ops_same_time.append(b_pop_l)
ops_two_time.append(np.kron(b_dag_r, b_r)); ops_two_time.append(np.kron(b_dag_l, b_l))

#G2's
ops_same_time.append(b_dag_r@b_dag_r@b_r@b_r); ops_same_time.append(b_dag_l@b_dag_l@b_l@b_l)
ops_two_time.append(np.kron(b_pop_r, b_pop_r)); ops_two_time.append(np.kron(b_pop_l, b_pop_l))


# Pops timing
def timedProcess():
    correlations, taus, t_ss = qmps.correlation_ss_1t(bins.correlation_bins, bins.output_field_states, ops_same_time, ops_two_time, input_params)
    t_ss_index = int(round(t_ss/input_params.delta_t))
    
    # Calculating steady state little g's
    g1_rr = correlations[0] / photon_fluxes[1][t_ss_index]
    g1_ll = correlations[1] / photon_fluxes[0][t_ss_index]
    g2_rr = correlations[2] / photon_fluxes[1][t_ss_index]**2
    g2_ll = correlations[3] / photon_fluxes[0][t_ss_index]**2

    spectra, w_list = qmps.spectrum_w(input_params.delta_t, correlations[0])

times = timeit.repeat(timedProcess, repeat=repeat, number=runs)
print('1 TLS, Markovian Correlations/spectra time:', min(times)/runs)
correlations, taus, t_ss = qmps.correlation_ss_1t(bins.correlation_bins, bins.output_field_states, ops_same_time, ops_two_time, input_params)
print('\tSteady State time:', t_ss)

#%%% 1TLS, Nonmarkovian
#Choose the coupling
gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1)
#Define input parameters
input_params = qmps.parameters.InputParams(
    delta_t=0.05, # simulation time step
    tmax = 40, # simulation total time length
    d_sys_total=[2],
    d_t_total=[2],
    gamma_l=gamma_l,
    gamma_r = gamma_r,  
    bond_max=18, #simulation maximum MPS bond dimension, truncates entanglement information
    tau=1, # Roundtrip feedback time
    phase= 0#np.pi
)
tlist=np.arange(0,input_params.tmax+input_params.delta_t/2,input_params.delta_t)

sys_initial_state=qmps.states.tls_ground()
wg_initial_state = None
cw_pump=2*np.pi

Hm=qmps.hamiltonian_1tls_feedback(input_params, cw_pump)

flux_op = qmps.b_pop(input_params)
tls_pop_op = qmps.tls_pop()

# Pops timing
def timedProcess():
    bins = qmps.t_evol_nmar(Hm,sys_initial_state,wg_initial_state,input_params)
    tls_pop = qmps.single_time_expectation(bins.system_states, tls_pop_op)
    photon_flux = qmps.single_time_expectation(bins.output_field_states, flux_op)
    loop_flux = qmps.single_time_expectation(bins.loop_field_states, flux_op)

times = timeit.repeat(timedProcess, repeat=repeat, number=runs)
print('1 TLS, Non-Markovian pops time:', min(times)/runs)

# Prep for timing
bins = qmps.t_evol_nmar(Hm,sys_initial_state,wg_initial_state,input_params)
tls_pop = qmps.single_time_expectation(bins.system_states, tls_pop_op)
photon_flux = qmps.single_time_expectation(bins.output_field_states, flux_op)

ops_same_time = []; ops_two_time = []
b_dag = qmps.b_dag(input_params); b = qmps.b(input_params)

# G1
ops_same_time.append(flux_op); ops_two_time.append(np.kron(b_dag, b))

#G2
ops_same_time.append(b_dag@b_dag@b@b); ops_two_time.append(np.kron(flux_op, flux_op))


# Pops timing
def timedProcess():
    correlations, taus, t_ss = qmps.correlation_ss_1t(bins.correlation_bins, bins.output_field_states, ops_same_time, ops_two_time, input_params)
    t_ss_index = int(round(t_ss/input_params.delta_t))
    
    # Calculating steady state little g's
    g1 = correlations[0] / photon_flux[t_ss_index]
    g2 = correlations[1] / photon_flux[t_ss_index]**2

    spectra, w_list = qmps.spectrum_w(input_params.delta_t, correlations[0])


times = timeit.repeat(timedProcess, repeat=repeat, number=runs)
print('1 TLS, Non-Markovian Correlations/spectra time:', min(times)/runs)
correlations, taus, t_ss = qmps.correlation_ss_1t(bins.correlation_bins, bins.output_field_states, ops_same_time, ops_two_time, input_params)
print('\tSteady State time:', t_ss)



#%% Test set 4, Fock Pulses
#Set relevant parameters of timing
runs = 20
repeat = 5
#%%% Test : 1 TLS, 1 photon Fock Pulse 

#Choose the coupling:
gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1) # same as gamma_l, gamma_r = (0.5,0.5)
#Define input parameters (dataclass)
input_params = qmps.parameters.InputParams(
    delta_t=0.05, # Time step of the simulation
    tmax = 8, # Maximum simulation time
    d_sys_total=[2],
    d_t_total=[2,2],
    gamma_l=gamma_l,
    gamma_r = gamma_r,  
    bond_max=4 # Maximum bond dimension, simulation parameter that adjusts truncation of entanglement information
)
#Make a tlist for plots
tlist=np.arange(0,input_params.tmax+input_params.delta_t/2,input_params.delta_t)

""" Choose the initial state"""

sys_initial_state=qmps.states.tls_ground() #TLS initially excited
gaussian_width, gaussian_center = 0.5,1.5
env = qmps.states.gaussian_envelope(input_params.tmax, input_params, gaussian_width,gaussian_center)

wg_initial_state = qmps.states.fock_pulse(env, input_params.tmax, 1, input_params)

Hm=qmps.hamiltonian_1tls(input_params) # Create the Hamiltonian for a single TLS

tls_pop_op = qmps.tls_pop() 
photon_pop_ops = [qmps.b_pop_l(input_params), qmps.b_pop_r(input_params)]

def timedProcess():
    bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)
    tls_pop = qmps.single_time_expectation(bins.system_states, tls_pop_op)
    photon_fluxes = qmps.single_time_expectation(bins.output_field_states, photon_pop_ops)
    net_flux_l = np.cumsum(photon_fluxes[0]) * input_params.delta_t
    net_flux_r = np.cumsum(photon_fluxes[1]) * input_params.delta_t


times = timeit.repeat(timedProcess, repeat=repeat, number=runs)
print('1 TLS, Markovian, 1 photon pulse, pops time:', min(times)/runs)

bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)

ops_same_time = []; ops_two_time = []
# Add in G1, both directions
ops_same_time.append(photon_pop_ops[0]); ops_same_time.append(photon_pop_ops[1])
ops_two_time.append(np.kron(qmps.b_dag_l(input_params), qmps.b_l(input_params))) 
ops_two_time.append(np.kron(qmps.b_dag_r(input_params), qmps.b_r(input_params))) 

def timedProcess():
    correlation, correlation_times = qmps.correlations_2t(bins.correlation_bins, ops_same_time, ops_two_time, input_params, completion_print_flag=False)

times = timeit.repeat(timedProcess, repeat=repeat, number=runs)
print('1 TLS, Markovian, 1 photon pulse, G1 correlation time:', min(times)/runs)

#%%% Test : 1 TLS, 2 photon Fock Pulse 

#Choose the coupling:
gamma_l,gamma_r=qmps.coupling('symmetrical',gamma=1) # same as gamma_l, gamma_r = (0.5,0.5)
#Define input parameters (dataclass)
input_params = qmps.parameters.InputParams(
    delta_t=0.05, # Time step of the simulation
    tmax = 8, # Maximum simulation time
    d_sys_total=[2],
    d_t_total=[3,3],
    gamma_l=gamma_l,
    gamma_r = gamma_r,  
    bond_max=8 # Maximum bond dimension, simulation parameter that adjusts truncation of entanglement information
)
#Make a tlist for plots
tlist=np.arange(0,input_params.tmax+input_params.delta_t/2,input_params.delta_t)

""" Choose the initial state"""

sys_initial_state=qmps.states.tls_ground() #TLS initially excited
gaussian_width, gaussian_center = 0.5,1.5
env = qmps.states.gaussian_envelope(input_params.tmax, input_params, gaussian_width,gaussian_center)

wg_initial_state = qmps.states.fock_pulse(env, input_params.tmax, 2, input_params)

Hm=qmps.hamiltonian_1tls(input_params) # Create the Hamiltonian for a single TLS

tls_pop_op = qmps.tls_pop() 
photon_pop_ops = [qmps.b_pop_l(input_params), qmps.b_pop_r(input_params)]

def timedProcess():
    bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)
    tls_pop = qmps.single_time_expectation(bins.system_states, tls_pop_op)
    photon_fluxes = qmps.single_time_expectation(bins.output_field_states, photon_pop_ops)
    net_flux_l = np.cumsum(photon_fluxes[0]) * input_params.delta_t
    net_flux_r = np.cumsum(photon_fluxes[1]) * input_params.delta_t


times = timeit.repeat(timedProcess, repeat=repeat, number=runs)
print('1 TLS, Markovian, 2 photon pulse, pops time:', min(times)/runs)

bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)

ops_same_time = []; ops_two_time = []
b_dag_l = qmps.b_dag_l(input_params) ; b_dag_r = qmps.b_dag_r(input_params)
b_l = qmps.b_l(input_params) ; b_r = qmps.b_r(input_params)


# Add in G2, both directions
ops_same_time.append(b_dag_l@b_dag_l@b_l@b_l); ops_same_time.append(b_dag_r@b_dag_r@b_r@b_r)
ops_two_time.append(np.kron(photon_pop_ops[0], photon_pop_ops[0])) 
ops_two_time.append(np.kron(photon_pop_ops[1], photon_pop_ops[1])) 

def timedProcess():
    correlation, correlation_times = qmps.correlations_2t(bins.correlation_bins, ops_same_time, ops_two_time, input_params, completion_print_flag=False)

times = timeit.repeat(timedProcess, repeat=repeat, number=runs)
print('1 TLS, Markovian, 2 photon pulse, G2 correlation time:', min(times)/runs)
