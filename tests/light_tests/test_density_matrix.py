"""
Light tests for the density-matrix API.
"""

import numpy as np

import QwaveMPS as qmps


def _make_markov_params():
    gamma_l, gamma_r = qmps.coupling("symmetrical", gamma=1)
    return qmps.parameters.InputParams(
        delta_t=0.1,
        tmax=0.6,
        d_sys_total=np.array([2]),
        d_t_total=np.array([2, 2]),
        gamma_l=gamma_l,
        gamma_r=gamma_r,
        bond_max=4,
    )


def _make_feedback_params():
    gamma_l, gamma_r = qmps.coupling("symmetrical", gamma=1)
    return qmps.parameters.InputParams(
        delta_t=0.1,
        tmax=0.6,
        d_sys_total=np.array([2]),
        d_t_total=np.array([2]),
        gamma_l=gamma_l,
        gamma_r=gamma_r,
        bond_max=4,
        tau=0.2,
        phase=np.pi,
    )


def test_convert_to_dm_shapes():
    sys_dm = qmps.convert_to_dm(qmps.states.tls_excited())
    assert sys_dm.shape == (1, 4, 1)

    params = _make_markov_params()
    field_dm = qmps.convert_to_dm(qmps.states.vacuum(params.tmax, params))
    assert len(field_dm) == int(round(params.tmax / params.delta_t, 0))
    assert field_dm[0].shape == (1, 16, 1)


def test_reshape_liouvillian_and_liouvillian_shapes():
    H = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    c_op = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    L = qmps.liouvillian(H, [c_op])
    assert L.shape == (4, 4)

    reshaped = qmps.reshape_liouvillian(np.eye(16, dtype=complex), [2, 2])
    assert reshaped.shape == (4, 4, 4, 4)


def test_single_time_expectation_dm_for_excited_state():
    excited_dm = qmps.convert_to_dm(qmps.states.tls_excited())
    bins = [[np.ones((1, 1)), excited_dm, np.ones((1, 1))]]

    pops = qmps.single_time_expectation_dm(bins, qmps.spre(qmps.tls_pop()))
    assert np.allclose(pops, np.array([1.0]))


def test_t_evol_mar_dm_matches_pure_state_without_losses():
    params = _make_markov_params()
    sys_state = qmps.states.tls_excited()
    sys_dm = qmps.convert_to_dm(sys_state)
    wg_state = qmps.states.vacuum(params.tmax, params)
    wg_dm = qmps.convert_to_dm(wg_state)

    H = qmps.hamiltonian_1tls(params)
    bins = qmps.t_evol_mar(H, sys_state, wg_state, params)
    bins_dm = qmps.t_evol_mar_dm(qmps.liouvillian(H / params.delta_t), sys_dm, wg_dm, params)

    pops = np.real(qmps.single_time_expectation(bins.system_states, qmps.tls_pop()))
    pops_dm = np.real(qmps.single_time_expectation_dm(bins_dm.system_states, qmps.spre(qmps.tls_pop())))
    assert np.allclose(pops_dm, pops, atol=1e-8)


def test_t_evol_nmar_dm_matches_pure_state_without_losses():
    params = _make_feedback_params()
    sys_state = qmps.states.tls_excited()
    sys_dm = qmps.convert_to_dm(qmps.states.tls_excited())
    wg_state = qmps.states.vacuum(params.tmax, params)
    wg_dm = qmps.convert_to_dm(wg_state)
    H = qmps.hamiltonian_1tls_feedback(params)

    bins = qmps.t_evol_nmar(H, sys_state, wg_state, params)
    bins_dm = qmps.t_evol_nmar_dm(qmps.liouvillian(H / params.delta_t), sys_dm, wg_dm, params)

    assert bins_dm.loop_field_states is not None
    assert len(bins_dm.loop_field_states) > 1
    assert len(bins_dm.output_field_states) > 1

    flux_op = qmps.b_pop(params)
    tls_pops = np.real(qmps.single_time_expectation(bins.system_states, qmps.tls_pop()))
    tls_pops_dm = np.real(qmps.single_time_expectation_dm(bins_dm.system_states, qmps.spre(qmps.tls_pop())))

    transmitted_flux = np.real(qmps.single_time_expectation(bins.output_field_states, flux_op))
    transmitted_flux_dm = np.real(
        qmps.single_time_expectation_dm(bins_dm.output_field_states, qmps.spre(flux_op))
    )

    loop_flux = np.real(qmps.single_time_expectation(bins.loop_field_states, flux_op))
    loop_flux_dm = np.real(qmps.single_time_expectation_dm(bins_dm.loop_field_states, qmps.spre(flux_op)))

    assert np.allclose(tls_pops_dm, tls_pops, atol=1e-8)
    assert np.allclose(transmitted_flux_dm, transmitted_flux, atol=1e-8)
    assert np.allclose(loop_flux_dm, loop_flux, atol=1e-8)


def test_dm_correlations_return_expected_shapes():
    params = _make_markov_params()
    sys_dm = qmps.convert_to_dm(qmps.states.tls_excited())
    H = qmps.hamiltonian_1tls(params)
    bins_dm = qmps.t_evol_mar_dm(qmps.liouvillian(H / params.delta_t), sys_dm, None, params)

    g1, tlist = qmps.correlation_2op_2t_dm(
        bins_dm.correlation_bins,
        qmps.b_dag_r(params),
        qmps.b_r(params),
        params,
    )
    g2, _ = qmps.correlation_4op_2t_dm(
        bins_dm.correlation_bins,
        qmps.b_dag_r(params),
        qmps.b_dag_r(params),
        qmps.b_r(params),
        qmps.b_r(params),
        params,
    )

    expected_n = len(bins_dm.correlation_bins) - 1
    assert g1.shape == (expected_n, expected_n)
    assert g2.shape == (expected_n, expected_n)
    assert tlist.shape == (expected_n,)


def test_entanglement_dm_zero_for_trivial_schmidt_vectors():
    ent = qmps.entanglement_dm([np.array([1.0]), np.array([1.0, 0.0])])
    assert np.allclose(ent, np.zeros(2))
