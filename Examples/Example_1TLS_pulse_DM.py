"""
1 TLS - Drive with Fock-state pulse and decoherence
===================================================

This example demonstrates density-matrix pulse scattering from a single
two-level system (TLS) driven by a 1-photon pulse in a chiral waveguide.

In addition to the coherent waveguide interaction, the example includes:

1. Pure dephasing with rate :math:`\\gamma_\\phi`
2. Additional loss with rate :math:`\\gamma_0`

The resulting spectrum is compared with the analytic sinc-squared lineshape of
the ideal top-hat pulse in the absence of pure dephasing and additional loss.

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
# We use a chiral right-propagating waveguide and a single-photon Fock pulse.

gamma_l, gamma_r = qmps.coupling("chiral_r", gamma=1)
input_params = qmps.parameters.InputParams(
    delta_t=0.02,
    tmax=15.0,
    d_sys_total=np.array([2]),
    d_t_total=np.array([2, 2]),
    gamma_l=gamma_l,
    gamma_r=gamma_r,
    bond_max=10,
)

tlist = np.arange(0, input_params.tmax + input_params.delta_t / 2, input_params.delta_t)


#%%
# Choose the initial state and pulse
# ----------------------------------
#
# The TLS starts in its ground state and the right-moving input field is a
# top-hat 1-photon pulse.

sys_initial_state_dm = qmps.convert_to_dm(qmps.states.tls_ground())

pulse_time = 2.0
photon_num = 1
pulse_env = qmps.tophat_envelope(pulse_time, input_params)

wg_initial_state_dm = qmps.convert_to_dm(
    qmps.states.fock_pulse(
        pulse_env,
        input_params.tmax,
        photon_num,
        input_params,
        direction="R",
        bond0=2,
    )
)


#%%
# Build the Liouvillian
# ---------------------
#
# The density-matrix dynamics is generated from the Hamiltonian and collapse
# operators describing pure dephasing and additional decay.

H = qmps.hamiltonian_1tls(input_params)
d_t = int(np.prod(input_params.d_t_total))

gamma_phi = 0.12
gamma_decay = 0.05

c_deph = np.sqrt(gamma_phi) * np.kron(qmps.tls_pop(), np.eye(d_t))
c_decay = np.sqrt(gamma_decay) * np.kron(qmps.sigmaminus(), np.eye(d_t))
L = qmps.liouvillian(H / input_params.delta_t, [c_deph, c_decay])


#%%
# Time evolution
# --------------

bins_dm = qmps.t_evol_mar_dm(L, sys_initial_state_dm, wg_initial_state_dm, input_params)


#%%
# Calculate observables
# ---------------------
#
# Compute the TLS population, the transmitted excitation number, and the
# first-order correlation function of the outgoing right-moving field. The
# system and field observables are promoted to superoperators with ``spre``.



tls_pops = qmps.single_time_expectation_dm(bins_dm.system_states, qmps.spre(qmps.tls_pop()))
flux_out_r = qmps.single_time_expectation_dm(
    bins_dm.output_field_states,
    qmps.spre(qmps.b_pop_r(input_params)),
)

g1, corr_tlist = qmps.correlation_2op_2t_dm(
    bins_dm.correlation_bins,
    qmps.b_dag_r(input_params),
    qmps.b_r(input_params),
    input_params,
)


#%%
# Compute the spectrum
# --------------------
#
# We use the time-dependent spectral intensity and integrate over the central
# time variable to obtain a spectrum that can be compared to the ideal analytic
# sinc-squared envelope of a top-hat pulse.

spectrum, wlist = qmps.spectral_intensity(g1, input_params, padding=1064)
sw_dm = np.trapezoid(spectrum, corr_tlist, axis=0)
analytic_sinc = np.abs(np.sinc(wlist / np.pi)) ** 2


#%%
# Plot the results
# ----------------

# Plot the resulting spectrum together with the analytical spectrum obtained for
# the ideal top-hat pulse without decoherence processes.

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

axes[0].plot(tlist, np.real(tls_pops), linewidth=3, color="k", linestyle="-", label=r"$n_{\rm TLS}$")
axes[0].plot(
    tlist,
    np.cumsum(np.real(flux_out_r)) * input_params.delta_t,
    linewidth=3,
    color="orange",
    linestyle="-",
    label=r"$N_{R}^{\rm out}$",
)
axes[0].set_xlabel(r"Time, $\gamma t$")
axes[0].set_ylabel("Populations")
axes[0].grid(True, linestyle="--", alpha=0.6)
axes[0].legend()
axes[0].set_xlim([0.0, input_params.tmax])

axes[1].plot(wlist, np.real(sw_dm), linewidth=3, color="r", linestyle="-", label="DM spectrum")
axes[1].plot(
    wlist,
    analytic_sinc,
    ls=(0, (1, 1)),
    linewidth=3,
    color="k",
    label=r"$|\mathrm{sinc}(\omega)|^2$",
)
axes[1].set_xlabel(r"Frequency [$\gamma$]")
axes[1].set_ylabel("Spectral intensity")
axes[1].grid(True, linestyle="--", alpha=0.6)
axes[1].set_xlim(-4, 4)
axes[1].legend()

fig.tight_layout()
plt.show()
