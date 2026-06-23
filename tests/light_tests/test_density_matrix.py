"""
Light tests for the density-matrix API.
"""

import numpy as np
import pytest

import QwaveMPS as qmps
from QwaveMPS.simulation_dm import (
    DIAGNOSTIC_WARNING_THRESHOLD,
    _final_system_diagnostics,
    _prepare_dm_input_bins,
    _rho_diagnostics,
    _svd_tensor_dm,
)


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


def test_prepare_dm_input_bins_is_vacuum_iterator():
    params = _make_markov_params()
    input_field = _prepare_dm_input_bins(None, params)

    first_bin = next(input_field)
    second_bin = next(input_field)

    assert first_bin.shape == (1, 16, 1)
    assert np.array_equal(second_bin, first_bin)


def test_prepare_dm_input_bins_yields_input_then_vacuum():
    params = _make_markov_params()
    occupied_state = np.zeros((1, 4, 1), dtype=complex)
    occupied_state[:, 1, :] = 1
    occupied = qmps.convert_to_dm(occupied_state)
    input_field = _prepare_dm_input_bins([occupied], params)

    assert np.array_equal(next(input_field), occupied)
    assert np.array_equal(
        next(input_field),
        qmps.convert_to_dm(qmps.states.wg_ground(4)),
    )


def test_rho_diagnostics_for_valid_density_matrix():
    rho = np.array([[0.25, 0.0], [0.0, 0.75]], dtype=complex)

    herm_err, trace_err = _rho_diagnostics(rho)

    assert herm_err == 0.0
    assert trace_err == 0.0


def test_final_system_diagnostics_prints_and_warns(capsys):
    rho_tensor = np.array([[[1.2], [0.0], [0.0], [0.0]]], dtype=complex)
    system_bin = [np.ones((1, 1)), rho_tensor, np.ones((1, 1))]

    with pytest.warns(RuntimeWarning, match="final diagnostics exceed"):
        herm_err, trace_err = _final_system_diagnostics(system_bin, np.array([2]), "Test DM")

    output = capsys.readouterr().out
    assert "Hermiticity violation=" in output
    assert "trace-preservation violation=" in output
    assert herm_err == 0.0
    assert trace_err > DIAGNOSTIC_WARNING_THRESHOLD


def test_final_system_diagnostics_supports_multiple_subsystems(capsys):
    pure_state = np.zeros((1, 4, 1), dtype=complex)
    pure_state[:, 1, :] = 1.0
    system_bin = [
        np.ones((1, 1)),
        qmps.convert_to_dm(pure_state),
        np.ones((1, 1)),
    ]

    herm_err, trace_err = _final_system_diagnostics(
        system_bin,
        np.array([2, 2]),
        "Two-TLS DM",
    )

    capsys.readouterr()
    assert herm_err == 0.0
    assert trace_err == 0.0


def test_dm_svd_applies_relative_cutoff():
    tensor = np.diag([1.0, 0.1, 0.01])

    _, singular_values, _ = _svd_tensor_dm(
        tensor,
        bond_max=3,
        d1=1,
        d2=1,
        tol=0.1,
    )

    assert singular_values.shape == (1,)
    assert np.allclose(singular_values, np.array([1.0]))


def test_markov_dm_evolution_uses_parameter_relative_cutoff(monkeypatch):
    import QwaveMPS.simulation_dm as simulation_dm

    params = _make_markov_params()
    params.relative_cutoff = 1e-7
    observed_cutoffs = []
    original_svd = simulation_dm._svd_tensor_dm

    def recording_svd(tensor, bond_max, d1, d2, tol=0):
        observed_cutoffs.append(tol)
        return original_svd(tensor, bond_max, d1, d2, tol)

    monkeypatch.setattr(simulation_dm, "_svd_tensor_dm", recording_svd)
    sys_dm = qmps.convert_to_dm(qmps.states.tls_excited())
    H = qmps.hamiltonian_1tls(params)
    simulation_dm.t_evol_mar_dm(qmps.liouvillian(H / params.delta_t), sys_dm, None, params)

    assert observed_cutoffs
    assert all(cutoff == params.relative_cutoff for cutoff in observed_cutoffs)


def test_reshape_liouvillian_and_liouvillian_shapes():
    H = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    c_op = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    L = qmps.liouvillian(H, [c_op])
    assert L.shape == (4, 4)

    reshaped = qmps.reshape_liouvillian(np.eye(16, dtype=complex), [2, 2])
    assert reshaped.shape == (4, 4, 4, 4)


def test_row_major_superoperators_match_matrix_multiplication():
    op = np.array([[0.2, 1.0j], [-0.4j, 0.7]], dtype=complex)
    rho = np.array([[0.6, 0.1j], [-0.1j, 0.4]], dtype=complex)
    rho_vec = rho.reshape(-1)

    left_result = (qmps.spre(op) @ rho_vec).reshape(rho.shape)
    right_result = (qmps.spost(op) @ rho_vec).reshape(rho.shape)

    assert np.allclose(left_result, op @ rho)
    assert np.allclose(right_result, rho @ op)


def test_lindblad_dissipator_action_and_rate():
    a = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    rho = np.array([[0.3, 0.2j], [-0.2j, 0.7]], dtype=complex)
    gamma = 0.25
    adag = a.conj().T
    expected = gamma * (a @ rho @ adag - 0.5 * (adag @ a @ rho + rho @ adag @ a))

    result = (qmps.lindblad_dissipator(a, gamma) @ rho.reshape(-1)).reshape(rho.shape)

    assert np.allclose(result, expected)


def test_liouvillian_matches_master_equation_action():
    H = np.array([[0.3, 0.2j], [-0.2j, -0.1]], dtype=complex)
    a = np.array([[0.0, 0.4], [0.0, 0.0]], dtype=complex)
    rho = np.array([[0.6, 0.1], [0.1, 0.4]], dtype=complex)
    adag_a = a.conj().T @ a
    expected = -1.0j * (H @ rho - rho @ H)
    expected += a @ rho @ a.conj().T - 0.5 * (adag_a @ rho + rho @ adag_a)

    result = (qmps.liouvillian(H, [a]) @ rho.reshape(-1)).reshape(rho.shape)

    assert np.allclose(result, expected)


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
