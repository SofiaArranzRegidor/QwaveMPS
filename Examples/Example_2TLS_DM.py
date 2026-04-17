"""
2 TLS - Decay in waveguide with delay and decoherence
=====================================================

This is an example of two two-level systems (TLSs) coupled through a waveguide
with finite delay, including pure dephasing and weak additional loss in the
density-matrix formalism.

All the examples are in units of the TLS total decay rate, gamma. Hence, in
general, gamma=1.

Computes time evolution, population dynamics, and the entanglement entropy of
the density-matrix evolution.

Example plots:

1. TLS population dynamics

2. Density-matrix entanglement entropy
"""

#%%
# Imports
# -------

import matplotlib.pyplot as plt
import numpy as np

import QwaveMPS as qmps


#%%
# Choose the simulation parameters
# --------------------------------
#
# Choose the field and system Hilbert-space sizes together with the delay time
# ``tau``.

d_t_total = np.array([2, 2])
d_sys_total = np.array([2, 2])
gamma_l1, gamma_r1 = qmps.coupling("symmetrical", gamma=1)
gamma_l2, gamma_r2 = qmps.coupling("symmetrical", gamma=1)

input_params = qmps.parameters.InputParams(
    delta_t=0.05,
    tmax=6.0,
    d_sys_total=d_sys_total,
    d_t_total=d_t_total,
    gamma_l=gamma_l1,
    gamma_r=gamma_r1,
    gamma_l2=gamma_l2,
    gamma_r2=gamma_r2,
    bond_max=16,
    phase=0.0,
    tau=0.2,
)

tlist = np.arange(0, input_params.tmax + input_params.delta_t / 2, input_params.delta_t)


#%%
# Choose the initial state
# ------------------------
#
# The first TLS starts excited and the second one starts in the ground state.

tls_excited = qmps.states.tls_excited()
tls_ground = qmps.states.tls_ground()

sys_initial_state_dm = qmps.convert_to_dm(np.kron(tls_excited, tls_ground))
wg_initial_state_dm = qmps.convert_to_dm(qmps.states.vacuum(input_params.tmax, input_params))


#%%
# Build the Liouvillian
# ---------------------
#
# Include pure dephasing and weak off-chip decay for both emitters.

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

L = qmps.liouvillian(H / input_params.delta_t, c_ops)


#%%
# Time evolution
# --------------

bins_dm = qmps.t_evol_nmar_dm(L, sys_initial_state_dm, wg_initial_state_dm, input_params)


#%%
# Calculate observables
# ---------------------
#
# Evaluate the population of each TLS separately and compute the
# operator entanglement from the stored Schmidt values.

tls1_pop = np.kron(qmps.tls_pop(), np.eye(d_sys2))
tls2_pop = np.kron(np.eye(d_sys1), qmps.tls_pop())

tls1_pops = qmps.single_time_expectation_dm(bins_dm.system_states, qmps.spre(tls1_pop))
tls2_pops = qmps.single_time_expectation_dm(bins_dm.system_states, qmps.spre(tls2_pop))
entanglement = qmps.entanglement_dm(bins_dm.schmidt)


#%%
# Plot the results
# ----------------

plt.plot(tlist, np.real(tls1_pops), linewidth=3, color="k", linestyle="-", label=r"$n_{\rm TLS1}$")
plt.plot(tlist, np.real(tls2_pops), linewidth=3, color="skyblue", linestyle="--", label=r"$n_{\rm TLS2}$")
plt.plot(tlist, np.real(entanglement), linewidth=3, color="r", linestyle="-", label=r"Operator Entanglement")
plt.legend()
plt.xlabel(r"Time, $\gamma t$")
plt.ylabel("Entropy / Population")
plt.grid(True, linestyle="--", alpha=0.6)
plt.ylim([0.0, 1.05])
plt.xlim([0.0, input_params.tmax])
plt.show()
