#%%
import QwaveMPS as qmps
import numpy as np
import itertools
import matplotlib.pyplot as plt


photon_num_l = 0
photon_num_r = 2
photon_num = max(photon_num_l, photon_num_r)
gaussian_env = False
input_params = qmps.parameters.InputParams(
    delta_t=0.05, 
    tmax = 8,
    d_sys_total=np.array([2]),
    d_t_total=np.array([photon_num_l+1,photon_num_r+1]),
    gamma_l=0.5,
    gamma_r = 0.5,  
    bond_max=max(32,2**(photon_num+3)),
)


tmax=input_params.tmax
delta_t=input_params.delta_t
tlist=np.arange(0,tmax+delta_t,delta_t)

sys_initial_state = qmps.tls_ground()

pulse_centers = [4,6]
sigmas = [1,1]
pulse_time_gauss = input_params.tmax
pulse_envs_gaus = []
for i in range(2):
    pulse_envs_gaus.append(qmps.gaussian_envelope(pulse_time_gauss, input_params, sigmas[i], pulse_centers[i]))
    pulse_envs_gaus[i] = qmps.normalize_pulse_envelope(pulse_envs_gaus[i])

wg_initial_state = qmps.product_fock_pulse([pulse_envs_gaus, pulse_envs_gaus],pulse_time_gauss,input_params, [photon_num_l, photon_num_r])   
#wg_initial_state = qmps.coherent_pulse([pulse_env_l, pulse_env_r],pulse_time_gauss,input_params, [0.2, 0.5])   

'''
# Example trying to make a superposition state, here a (|N0> + |0N>) state
wg_initial_state = qmps.fock_pulse([pulse_env_l, pulse_env_l],pulse_time_rect,input_params, [1, 0])   
wg_initial_state2 = qmps.fock_pulse([pulse_env_l, pulse_env_l],pulse_time_rect,input_params, [0, 1])   

wg_initial_state = qmps.addMPSs(wg_initial_state, wg_initial_state2)
print(wg_initial_state[0].shape, wg_initial_state[-1].shape)
#wg_initial_state = qmps.left_normalize_bins(wg_initial_state, input_params.bond_max)
print(wg_initial_state[0].shape, wg_initial_state[-1].shape)
'''

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

total_quanta = tls_pop + np.cumsum(np.sum(photon_fluxes, axis=0)) * delta_t

#%%
lw = 2
plt.plot(tlist, tls_pop, linewidth=lw, label=r'$n_{TLS}$')
plt.plot(tlist, total_quanta,linewidth=lw, label=r'Quanta')
plt.plot(tlist, same_time_corrs_l[0],linewidth=lw, label=r'$n^{\rm in}_{L}$')
plt.plot(tlist, same_time_corrs_r[0],linewidth=lw, linestyle=':', label=r'$n^{\rm in}_{R}$')

#plt.hlines(photon_num_r, 0, 4)

plt.xlim((0,8))
plt.ylim((0,None))
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'Pops')
plt.legend()
plt.show()
# %%
