"""
2 TLS - Delayed dynamics with dephasing
=======================================

Density-matrix example of two TLSs with delayed feedback, pure dephasing,
and weak additional decay.
"""

import matplotlib.pyplot as plt
import numpy as np

import QwaveMPS as qmps


d_t_total = np.array([2, 2])
d_sys_total = np.array([2, 2])
gamma_l1, gamma_r1 = qmps.coupling("symmetrical", gamma=1)
gamma_l2, gamma_r2 = qmps.coupling("symmetrical", gamma=1)

input_params = qmps.parameters.InputParams(
    delta_t=0.1,
    tmax=2.0,
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=gamma_l1,
    gamma_r=gamma_r1,
    gamma_l2=gamma_l2,
    gamma_r2=gamma_r2,
    bond_max=8,
    phase=0.0,
    tau=0.2,
)

tlist = np.arange(0, input_params.tmax + input_params.delta_t / 2, input_params.delta_t)
tls_excited = qmps.states.tls_excited()
tls_ground = qmps.states.tls_ground()
sys_initial_state_dm = qmps.convert_to_dm(np.kron(tls_excited, tls_ground))
wg_initial_state_dm = qmps.convert_to_dm(qmps.states.vacuum(input_params.tmax, input_params))

H = qmps.hamiltonian_2tls_nmar(input_params, delta1=0.0)
d_t = int(np.prod(d_t_total))
d_sys1, d_sys2 = d_sys_total
gamma_phi = 0.1
gamma_decay = 0.05
c_ops = [
    np.sqrt(gamma_phi) * np.kron(np.kron(np.eye(d_t), np.kron(qmps.tls_pop(), np.eye(d_sys2))), np.eye(d_t)),
    np.sqrt(gamma_phi) * np.kron(np.kron(np.eye(d_t), np.kron(np.eye(d_sys1), qmps.tls_pop())), np.eye(d_t)),
    np.sqrt(gamma_decay) * np.kron(np.kron(np.eye(d_t), np.kron(qmps.sigmaminus(), np.eye(d_sys2))), np.eye(d_t)),
    np.sqrt(gamma_decay) * np.kron(np.kron(np.eye(d_t), np.kron(np.eye(d_sys1), qmps.sigmaminus())), np.eye(d_t)),
]
L = qmps.liouvillian_dm(H / input_params.delta_t, c_ops)

bins_dm = qmps.t_evol_nmar_dm(L, sys_initial_state_dm, wg_initial_state_dm, input_params)

tls1_pop = np.kron(qmps.tls_pop(), np.eye(d_sys2))
tls2_pop = np.kron(np.eye(d_sys1), qmps.tls_pop())
tls1_pops = qmps.single_time_expectation_dm(bins_dm.system_states, qmps.spre(tls1_pop))
tls2_pops = qmps.single_time_expectation_dm(bins_dm.system_states, qmps.spre(tls2_pop))
entanglement = qmps.entanglement_dm(bins_dm.schmidt)

plt.figure(figsize=(5.0, 3.2))
plt.plot(tlist, np.real(tls1_pops), label="TLS 1", color="black")
plt.plot(tlist, np.real(tls2_pops), label="TLS 2", color="tab:blue")
plt.plot(tlist, np.real(entanglement), label="Entanglement", color="tab:red")
plt.xlabel(r"Time, $\gamma t$")
plt.ylabel("Observable")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
