"""
1 TLS - Feedback with dephasing
===============================

Density-matrix example of a single TLS in a semi-infinite waveguide with
delayed feedback and pure dephasing.
"""

import matplotlib.pyplot as plt
import numpy as np

import QwaveMPS as qmps


d_sys_total = np.array([2])
d_t_total = np.array([2])
gamma_l, gamma_r = qmps.coupling("symmetrical", gamma=1)

input_params = qmps.parameters.InputParams(
    delta_t=0.05,
    tmax=4.0,
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=gamma_l,
    gamma_r=gamma_r,
    bond_max=4,
    tau=0.5,
    phase=np.pi,
)

tlist = np.arange(0, input_params.tmax + input_params.delta_t / 2, input_params.delta_t)
sys_initial_state_dm = qmps.convert_to_dm(qmps.states.tls_excited())
wg_initial_state_dm = qmps.convert_to_dm(qmps.states.vacuum(input_params.tmax, input_params))

H = qmps.hamiltonian_1tls_feedback(input_params)
d_t = int(np.prod(d_t_total))
gamma_phi = 0.2
c_deph = np.sqrt(gamma_phi) * np.kron(np.kron(np.eye(d_t), qmps.tls_pop()), np.eye(d_t))
L = qmps.liouvillian_dm(H / input_params.delta_t, [c_deph])

bins_dm = qmps.t_evol_nmar_dm(L, sys_initial_state_dm, wg_initial_state_dm, input_params)

tls_pops = qmps.single_time_expectation_dm(bins_dm.system_states, qmps.spre(qmps.tls_pop()))
flux_op = qmps.b_pop(input_params)
flux_super = qmps.spre(flux_op)
transmitted_flux = qmps.single_time_expectation_dm(bins_dm.output_field_states, flux_super)
loop_flux = qmps.single_time_expectation_dm(bins_dm.loop_field_states, flux_super)
net_transmitted = np.cumsum(transmitted_flux) * input_params.delta_t
loop_sum = qmps.loop_integrated_statistics(loop_flux, input_params)
total_quanta = tls_pops + loop_sum + net_transmitted

plt.figure(figsize=(5.0, 3.2))
plt.plot(tlist, np.real(tls_pops), label=r"$n_{\rm TLS}$", color="black")
plt.plot(tlist, np.real(net_transmitted), label=r"$N_{\rm out}$", color="orange")
plt.plot(tlist, np.real(loop_sum), label=r"$N_{\rm loop}$", color="tab:blue")
plt.plot(tlist, np.real(total_quanta), label="Total", color="tab:green")
plt.xlabel(r"Time, $\gamma t$")
plt.ylabel("Population")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
