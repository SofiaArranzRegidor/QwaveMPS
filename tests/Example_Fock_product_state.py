#%%
import QwaveMPS as qmps
import numpy as np
import itertools
import matplotlib.pyplot as plt


photon_num_l = 0
photon_num_r = 3
photon_num = max(photon_num_l, photon_num_r)
gaussian_env = False
params = qmps.parameters.InputParams(
    delta_t=0.05, 
    tmax = 20,
    d_sys_total=np.array([2]),
    d_t_total=np.array([photon_num_l+1,photon_num_r+1]),
    gamma_l=0,
    gamma_r = 1,  
    bond_max=max(32,2**(photon_num+3)),
)


tmax=params.tmax
delta_t=params.delta_t
tlist=np.arange(0,tmax+delta_t,delta_t)

sys_initial_state = qmps.tls_ground()

pulse_centers = [3,8,13]
sigmas = [1]*photon_num_r
pulse_time_gauss = params.tmax
pulse_envs_gaus = []
for i in range(photon_num_r):
    pulse_envs_gaus.append(qmps.gaussian_envelope(pulse_time_gauss, params, sigmas[i], pulse_centers[i]))
    pulse_envs_gaus[i] = qmps.normalize_pulse_envelope(pulse_envs_gaus[i])
pulse_envs_left = np.ones(np.array(pulse_envs_gaus).shape)
wg_initial_state = qmps.product_fock_pulse([pulse_envs_left, pulse_envs_gaus],pulse_time_gauss,params, [photon_num_l, photon_num_r])   


Hm=qmps.hamiltonian_1tls(params)
bins = qmps.t_evol_mar(Hm,sys_initial_state,wg_initial_state,params)


# Calculate the two level system population
photon_pop_ops = [qmps.b_pop_l(params), qmps.b_pop_r(params)]

same_time_corrs_r = [qmps.b_pop_r(params)]
for i in range(photon_num+1):
    same_time_corrs_r.append(qmps.b_dag_r(params) @ same_time_corrs_r[-1] @ qmps.b_r(params))

same_time_corrs_l = [qmps.b_pop_l(params)]
for i in range(photon_num+1):
    same_time_corrs_l.append(qmps.b_dag_l(params) @ same_time_corrs_l[-1] @ qmps.b_l(params))


photon_fluxes = np.real(qmps.single_time_expectation(bins.output_field_states, photon_pop_ops))
tls_pop = np.real(qmps.single_time_expectation(bins.system_states, qmps.tls_pop()))
same_time_corrs_r_in = np.real(qmps.single_time_expectation(bins.input_field_states, same_time_corrs_r))
same_time_corrs_l_in = np.real(qmps.single_time_expectation(bins.input_field_states, same_time_corrs_l))

same_time_corrs_r_out = np.real(qmps.single_time_expectation(bins.output_field_states, same_time_corrs_r))
same_time_corrs_l_out = np.real(qmps.single_time_expectation(bins.output_field_states, same_time_corrs_l))


total_quanta = tls_pop + np.cumsum(np.sum(photon_fluxes, axis=0)) * delta_t

#%%
lw = 2
plt.plot(tlist, tls_pop, linewidth=lw, label=r'$n_{TLS}$')
plt.plot(tlist, total_quanta,linewidth=lw, label=r'Quanta')
plt.plot(tlist, same_time_corrs_l_in[0],linewidth=lw, label=r'$n^{\rm in}_{L}$')
plt.plot(tlist, same_time_corrs_r_in[0],linewidth=lw, linestyle=':', label=r'$n^{\rm in}_{R}$')
plt.plot(tlist, same_time_corrs_r_out[0],linewidth=lw, linestyle='--', label=r'$n^{\rm out}_{R}$')
plt.plot(tlist, same_time_corrs_r_out[1],linewidth=lw, linestyle=':', label=r'$n^{\rm out}_{RR}$')
plt.plot(tlist, same_time_corrs_l_out[0],linewidth=lw, linestyle='--', label=r'$n^{\rm out}_{L}$')


plt.xlim((0,tmax))
plt.ylim((0,None))
plt.xlabel(r'$\gamma t$')
plt.ylabel(r'Pops')
plt.legend()
plt.show()
# %%
b_dag_r = qmps.b_dag_r(params); b_r = qmps.b_r(params)
G2_rr, t_corr = qmps.correlation_4op_2t(bins.correlation_bins, b_dag_r, b_dag_r, b_r, b_r, params)
#%%
import cmasher as cmr
import matplotlib.colors as colors
"""Graph Examples"""
fonts=15
"""Example: Spectral Intensity"""

X,Y = np.meshgrid(t_corr,t_corr)
z = np.real(qmps.transform_t_tau_to_t1_t2(G2_rr))
absMax = np.abs(z).max()

cmap = cmr.get_sub_cmap('seismic', 0.5, 1)
norm = colors.PowerNorm(gamma=0.5)

fig, ax = plt.subplots(figsize=(4.5, 4.5))
cf = ax.pcolormesh(X,Y,z,shading='gouraud',cmap=cmap, norm=norm, rasterized=True)
cbar = fig.colorbar(cf,ax=ax, pad=0)

ax.set_xlabel(r'Time, $\gamma t$')
ax.set_ylabel(r'Time, $\gamma (t+\tau)$')
cbar.set_label(r'$G^{(2)}_{RR}(t,\tau)\ [\gamma^{2}]$',labelpad=0)
#ax.xaxis.set_major_formatter(formatter)
#ax.yaxis.set_major_formatter(formatter)
#cbar.ax.yaxis.set_major_formatter(formatter)

cbar.ax.tick_params(width=0.75)
plt.xlim([0,tmax])
plt.ylim([0,tmax])
plt.show()

# %%
