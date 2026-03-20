#%%
import QwaveMPS as qmps
import numpy as np
import itertools
import matplotlib.pyplot as plt


photon_num_l = 5
photon_num_r = 5
photon_num = max(photon_num_l, photon_num_r)
gaussian_env = False
input_params = qmps.parameters.InputParams(
    delta_t=0.02, 
    tmax = 8,
    d_sys_total=np.array([2]),
    d_t_total=np.array([photon_num+1,photon_num+1]),
    gamma_l=0.5,
    gamma_r = 0.5,  
    bond_max=max(32,2**(photon_num)),
)


tmax=input_params.tmax
delta_t=input_params.delta_t
tlist=np.arange(0,tmax+delta_t,delta_t)

sys_initial_state = qmps.tls_ground()

if gaussian_env:
    pulse_center = 3
    sigma = 1
    pulse_time = input_params.tmax
    pulse_env = qmps.gaussian_envelope(pulse_time, input_params, sigma, pulse_center)
else:
    pulse_time = 1
    pulse_env = qmps.tophat_envelope(pulse_time, input_params)

pulse_env_r = pulse_env
pulse_env_l = pulse_env

#wg_initial_state = qmps.fock_pulse(pulse_env,pulse_time, photon_num_r, input_params)
#print('='*50)
wg_initial_state = qmps._fock_pulse(pulse_env_r,pulse_time,input_params, pulse_env_l, photon_num_l, photon_num_r)   
#wg_initial_state = qmps._fock_pulse2([pulse_env_l, pulse_env_r], pulse_time, input_params, [photon_num_l, photon_num_r])
#wg_initial_state = qmps._noom_state([pulse_env_l, pulse_env_r], pulse_time, input_params, [photon_num_l, photon_num_r])


Hm=qmps.hamiltonian_1tls(input_params)
bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,input_params)


# Calculate the two level system population
photon_pop_ops = [qmps.b_pop_l(input_params), qmps.b_pop_r(input_params)]

same_time_corrs_r = [qmps.b_pop_r(input_params)]
for i in range(photon_num+1):
    same_time_corrs_r.append(qmps.b_dag_r(input_params) @ same_time_corrs_r[-1] @ qmps.b_r(input_params))

same_time_corrs_l = [qmps.b_pop_l(input_params)]
for i in range(photon_num+1):
    same_time_corrs_l.append(qmps.b_dag_l(input_params) @ same_time_corrs_l[-1] @ qmps.b_l(input_params))


photon_fluxes = np.real(qmps.single_time_expectation(bins.output_field_states, photon_pop_ops))
tls_pop = np.real(qmps.single_time_expectation(bins.system_states, qmps.tls_pop()))
same_time_corrs_r = np.real(qmps.single_time_expectation(bins.input_field_states, same_time_corrs_r))
same_time_corrs_l = np.real(qmps.single_time_expectation(bins.input_field_states, same_time_corrs_l))

total_quanta = np.cumsum(same_time_corrs_r[0] + same_time_corrs_l[0]) * delta_t
total_quanta2 = tls_pop + np.cumsum(np.sum(photon_fluxes, axis=0)) * delta_t

#%%
lw = 2
plt.plot(tlist, tls_pop, linewidth=lw, label=r'$n_{TLS}$')
plt.plot(tlist, total_quanta,linewidth=lw, label=r'Quanta')
plt.plot(tlist, total_quanta2,linewidth=lw, linestyle = ':',label=r'Quanta 2')
plt.plot(tlist, same_time_corrs_l[0],linewidth=lw, label=r'$n^{\rm in}_{L}$')
plt.plot(tlist, same_time_corrs_r[0],linewidth=lw, linestyle=':', label=r'$n^{\rm in}_{R}$')

plt.hlines(photon_num_r, 0, 4)

plt.xlim((0,4))
plt.ylim((0,None))
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'Pops')
plt.legend()
# %%
