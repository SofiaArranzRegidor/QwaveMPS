"""
1 TLS - Decay in semi-infinite waveguide with decoherence
=========================================================

This is an example of a single two-level system (TLS) in a semi-infinite
waveguide with delayed feedback in the density-matrix formalism.

The setup is similar to the standard non-Markovian feedback example, but now
includes pure dephasing through a collapse operator in the Liouvillian.

All the examples are in units of the TLS total decay rate, gamma. Hence, in
general, gamma=1.

Computes time evolution, population dynamics, and the integrated excitation
number in the output field and feedback loop.
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
# For the feedback geometry, there is only one time-bin channel. The feedback
# delay is set by ``tau`` and the roundtrip phase by ``phase``.

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


#%%
# Choose the initial state and Liouvillian
# ----------------------------------------
#
# The system starts in the excited state and the waveguide is initialized in
# vacuum. The pure states are converted to density-matrix MPS bins using
# ``convert_to_dm``.
#
# We include decoherence through the a lindblad term in Liouvillian, defined using the jump operator c_deph

sys_initial_state_dm = qmps.convert_to_dm(qmps.states.tls_excited())
wg_initial_state_dm = qmps.convert_to_dm(qmps.states.vacuum(input_params.tmax, input_params))

H = qmps.hamiltonian_1tls_feedback(input_params)
d_t = int(np.prod(d_t_total))

gamma_phi = 0.2
c_deph = np.sqrt(gamma_phi) * np.kron(np.kron(np.eye(d_t), qmps.tls_pop()), np.eye(d_t))
L = qmps.liouvillian(H / input_params.delta_t, [c_deph])


#%%
# Time evolution
# --------------

bins_dm = qmps.t_evol_nmar_dm(L, sys_initial_state_dm, wg_initial_state_dm, input_params)


#%%
# Calculate observables
# ---------------------
#
# We evaluate the TLS population and the loop/output field excitation numbers.

tls_pops = qmps.single_time_expectation_dm(bins_dm.system_states, qmps.spre(qmps.tls_pop()))

flux_op = qmps.b_pop(input_params)
flux_super = qmps.spre(flux_op)

transmitted_flux = qmps.single_time_expectation_dm(bins_dm.output_field_states, flux_super)
loop_flux = qmps.single_time_expectation_dm(bins_dm.loop_field_states, flux_super)

net_transmitted = np.cumsum(transmitted_flux) * input_params.delta_t
loop_sum = qmps.loop_integrated_statistics(loop_flux, input_params)
total_quanta = tls_pops + loop_sum + net_transmitted


#%%
# Plot the results
# ----------------

plt.plot(tlist, np.real(tls_pops), linewidth=3, color="k", linestyle="-", label=r"$n_{\rm TLS}$")
plt.plot(tlist, np.real(net_transmitted), linewidth=3, color="orange", linestyle="-", label=r"$N^{\rm out}$")
plt.plot(tlist, np.real(loop_sum), linewidth=3, color="b", linestyle=":", label=r"$N^{\rm loop}$")
plt.plot(tlist, np.real(total_quanta), linewidth=3, color="g", linestyle="-", label="Total")
plt.legend(loc="upper right")
plt.xlabel(r"Time, $\gamma t$")
plt.ylabel("Populations")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xlim([0.0, input_params.tmax])
plt.show()
