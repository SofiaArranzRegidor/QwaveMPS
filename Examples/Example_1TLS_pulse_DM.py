"""
1 TLS - Pulse scattering with dephasing
======================================

Density-matrix example of a single-photon pulse scattering from a TLS with
pure dephasing.
"""

import matplotlib.pyplot as plt
import numpy as np

import QwaveMPS as qmps


gamma_l, gamma_r = qmps.coupling("chiral_r", gamma=1)
input_params = qmps.parameters.InputParams(
    delta_t=0.05,
    tmax=15.0,
    d_sys_total=np.array([2]),
    d_t_total=np.array([2, 2]),
    gamma_l=gamma_l,
    gamma_r=gamma_r,
    bond_max=10,
)

tlist = np.arange(0, input_params.tmax + input_params.delta_t / 2, input_params.delta_t)
pulse_env = qmps.gaussian_envelope(input_params.tmax, input_params, gaussian_width=1.0, gaussian_center=3.0)

sys_initial_state_dm = qmps.convert_to_dm(qmps.states.tls_ground())
wg_initial_state_dm = qmps.convert_to_dm(
    qmps.states.fock_pulse(pulse_env, input_params.tmax, 1, input_params, direction="R")
)

H = qmps.hamiltonian_1tls(input_params)
d_t = int(np.prod(input_params.d_t_total))
gamma_phi = 0.0
gamma_decay = 0.00
c_deph = np.sqrt(gamma_phi) * np.kron(qmps.tls_pop(), np.eye(d_t))
c_decay = np.sqrt(gamma_decay) * np.kron(qmps.sigmaminus(), np.eye(d_t))
L = qmps.liouvillian_dm(H / input_params.delta_t, [c_deph, c_decay])

bins_dm = qmps.t_evol_mar_dm(L, sys_initial_state_dm, wg_initial_state_dm, input_params)

tls_pops = qmps.single_time_expectation_dm(bins_dm.system_states, qmps.spre(qmps.tls_pop()))
flux_out_r = qmps.single_time_expectation_dm(
    bins_dm.output_field_states, qmps.spre(qmps.b_pop_r(input_params))
)

g1, corr_tlist = qmps.correlation_2op_2t_dm(
    bins_dm.correlation_bins,
    qmps.b_dag_r(input_params),
    qmps.b_r(input_params),
    input_params,
)

#%%
spectrum, wlist = qmps.spectral_intensity(g1, input_params, padding=512)
sw_dm =np.trapezoid(spectrum,corr_tlist,axis=0)

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
axes[0].plot(tlist, np.real(tls_pops), color="black", label=r"$n_{\rm TLS}$")
axes[0].plot(
    tlist,
    np.cumsum(np.real(flux_out_r)) * input_params.delta_t,
    color="orange",
    label=r"$N_{R,\rm out}$",
)
axes[0].set_xlabel(r"Time, $\gamma t$")
axes[0].set_ylabel("Population")
axes[0].grid(True, linestyle="--", alpha=0.6)
axes[0].legend()

axes[1].plot(wlist, np.real(sw_dm), color="tab:red")

axes[1].plot(
    wlist,
    np.abs(np.sinc(wlist / np.pi)) ** 2,
    ls=(0, (1, 1)),
    linewidth=2,
    color="black",
)
axes[1].set_xlabel(r"Frequency [$\gamma$]")
axes[1].set_ylabel("Spectral intensity")
axes[1].grid(True, linestyle="--", alpha=0.6)
axes[1].set_xlim(-4,4)

fig.tight_layout()
plt.show()
