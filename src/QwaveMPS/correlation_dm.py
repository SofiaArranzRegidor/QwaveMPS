#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Density-matrix correlation and entanglement helpers for QwaveMPS.
"""

import copy

import numpy as np
from ncon import ncon
from tqdm import tqdm

from QwaveMPS.operators import op_list_check, sigmaminus, sigmaplus
from QwaveMPS.operators_dm import (
    expval_dm,
    expval_twotime_dm,
    reshape_liouvillian,
    spre,
    tensor_to_rho,
    trace_left,
)
from QwaveMPS.simulation_dm import absorb_right_env

__all__ = [
    "correlation_2op_2t_dm",
    "correlation_4op_2t_dm",
    "correlation_2op_1t_dm",
    "correlation_4op_1t_dm",
    "correlations_2t_dm",
    "correlations_1t_dm",
    "entanglement_dm",
]


def _progress(iterable, enabled: bool, total: int, desc: str):
    if not enabled:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="step", dynamic_ncols=True)


def correlation_2op_2t_dm(correlation_bins, a_op_list, b_op_list, params, completion_print_flag: bool = True):
    """
    Compute the density-matrix two-time correlation
    :math:`\\langle A(t) B(t+\\tau) \\rangle`.

    Parameters
    ----------
    correlation_bins : list[np.ndarray]
        Density-matrix correlation bins returned by a time-evolution routine.

    a_op_list : np.ndarray or list[np.ndarray]
        Single operator ``A`` or list of operators.

    b_op_list : np.ndarray or list[np.ndarray]
        Single operator ``B`` or list of operators.

    params : InputParams
        Simulation parameters.

    completion_print_flag : bool, default: True
        If ``True``, show the outer-loop progress bar during the calculation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Correlation data and the associated time axis. For a single operator
        pair, the correlation array has shape ``(n_t, n_tau)``. For a list of
        operator pairs, the result has shape ``(n_ops, n_t, n_tau)``.
    """
    list_flag = op_list_check(a_op_list)
    if list_flag and len(a_op_list) != len(b_op_list):
        raise ValueError("Lengths of operators lists are not equals")

    ops_same_time = []
    ops_two_time = []
    if list_flag:
        for a_op, b_op in zip(a_op_list, b_op_list):
            ops_same_time.append(spre(a_op @ b_op))
            ops_two_time.append(spre(np.kron(a_op, b_op)))
    else:
        ops_same_time.append(spre(a_op_list @ b_op_list))
        ops_two_time.append(spre(np.kron(a_op_list, b_op_list)))

    results, t_list = correlations_2t_dm(
        correlation_bins, ops_same_time, ops_two_time, params, completion_print_flag=completion_print_flag
    )
    return (results if list_flag else results[0]), t_list


def correlation_4op_2t_dm(
    correlation_bins,
    a_op_list,
    b_op_list,
    c_op_list,
    d_op_list,
    params,
    completion_print_flag: bool = True,
):
    """
    Compute the density-matrix four-operator two-time correlation
    :math:`\\langle A(t) B(t+\\tau) C(t+\\tau) D(t) \\rangle`.

    Parameters
    ----------
    correlation_bins : list[np.ndarray]
        Density-matrix correlation bins returned by a time-evolution routine.

    a_op_list, b_op_list, c_op_list, d_op_list : np.ndarray or list[np.ndarray]
        Single operators or matching lists of operators.

    params : InputParams
        Simulation parameters.

    completion_print_flag : bool, default: True
        If ``True``, show the outer-loop progress bar during the calculation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Correlation data and the associated time axis. For a single operator
        quadruple, the correlation array has shape ``(n_t, n_tau)``. For a
        list of operator quadruples, the result has shape
        ``(n_ops, n_t, n_tau)``.
    """
    list_flag = op_list_check(a_op_list)
    if list_flag and not (len(a_op_list) == len(b_op_list) == len(c_op_list) == len(d_op_list)):
        raise ValueError("Lengths of operators lists are not equal")

    ops_same_time = []
    ops_two_time = []
    if list_flag:
        for a_op, b_op, c_op, d_op in zip(a_op_list, b_op_list, c_op_list, d_op_list):
            ops_same_time.append(spre(a_op @ b_op @ c_op @ d_op))
            ops_two_time.append(spre(np.kron(a_op @ d_op, b_op @ c_op)))
    else:
        ops_same_time.append(spre(a_op_list @ b_op_list @ c_op_list @ d_op_list))
        ops_two_time.append(spre(np.kron(a_op_list @ d_op_list, b_op_list @ c_op_list)))

    results, t_list = correlations_2t_dm(
        correlation_bins, ops_same_time, ops_two_time, params, completion_print_flag=completion_print_flag
    )
    return (results if list_flag else results[0]), t_list


def correlation_2op_1t_dm(correlation_bins, a_op_list, b_op_list, params, t, completion_print_flag: bool = True):
    """
    Compute one density-matrix time-slice of
    :math:`\\langle A(t) B(t+\\tau) \\rangle`.

    Parameters
    ----------
    correlation_bins : list[np.ndarray]
        Density-matrix correlation bins returned by a time-evolution routine.

    a_op_list : np.ndarray or list[np.ndarray]
        Single operator ``A`` or list of operators.

    b_op_list : np.ndarray or list[np.ndarray]
        Single operator ``B`` or list of operators.

    params : InputParams
        Simulation parameters.

    t : float
        Time at which the first operator is evaluated.

    completion_print_flag : bool, default: True
        Currently unused. It is accepted for API compatibility with the full
        two-time correlation routines.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Correlation data and the associated time axis. For a single operator
        pair, the correlation array has shape ``(n_tau,)``. For a list of
        operator pairs, the result has shape ``(n_ops, n_tau)``.
    """
    list_flag = op_list_check(a_op_list)
    if list_flag and len(a_op_list) != len(b_op_list):
        raise ValueError("Lengths of operators lists are not equals")

    ops_same_time = []
    ops_two_time = []
    if list_flag:
        for a_op, b_op in zip(a_op_list, b_op_list):
            ops_same_time.append(spre(a_op @ b_op))
            ops_two_time.append(spre(np.kron(a_op, b_op)))
    else:
        ops_same_time.append(spre(a_op_list @ b_op_list))
        ops_two_time.append(spre(np.kron(a_op_list, b_op_list)))

    results, t_list = correlations_1t_dm(
        correlation_bins, ops_same_time, ops_two_time, params, t, completion_print_flag=completion_print_flag
    )
    return (results if list_flag else results[0]), t_list


def correlation_4op_1t_dm(
    correlation_bins,
    a_op_list,
    b_op_list,
    c_op_list,
    d_op_list,
    params,
    t,
    completion_print_flag: bool = True,
):
    """
    Compute one density-matrix time-slice of the four-operator correlation
    :math:`\\langle A(t) B(t+\\tau) C(t+\\tau) D(t) \\rangle`.

    Parameters
    ----------
    correlation_bins : list[np.ndarray]
        Density-matrix correlation bins returned by a time-evolution routine.

    a_op_list, b_op_list, c_op_list, d_op_list : np.ndarray or list[np.ndarray]
        Single operators or matching lists of operators.

    params : InputParams
        Simulation parameters.

    t : float
        Time at which the first pair of operators is evaluated.

    completion_print_flag : bool, default: True
        Currently unused. It is accepted for API compatibility with the full
        two-time correlation routines.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Correlation data and the associated time axis. For a single operator
        quadruple, the correlation array has shape ``(n_tau,)``. For a list of
        operator quadruples, the result has shape ``(n_ops, n_tau)``.
    """
    list_flag = op_list_check(a_op_list)
    if list_flag and not (len(a_op_list) == len(b_op_list) == len(c_op_list) == len(d_op_list)):
        raise ValueError("Lengths of operators lists are not equal")

    ops_same_time = []
    ops_two_time = []
    if list_flag:
        for a_op, b_op, c_op, d_op in zip(a_op_list, b_op_list, c_op_list, d_op_list):
            ops_same_time.append(spre(a_op @ b_op @ c_op @ d_op))
            ops_two_time.append(spre(np.kron(a_op @ d_op, b_op @ c_op)))
    else:
        ops_same_time.append(spre(a_op_list @ b_op_list @ c_op_list @ d_op_list))
        ops_two_time.append(spre(np.kron(a_op_list @ d_op_list, b_op_list @ c_op_list)))

    results, t_list = correlations_1t_dm(
        correlation_bins, ops_same_time, ops_two_time, params, t, completion_print_flag=completion_print_flag
    )
    return (results if list_flag else results[0]), t_list


def correlations_2t_dm(correlation_bins, ops_same_time, ops_two_time, params, completion_print_flag: bool = False):
    """
    Evaluate general two-time correlations in the density-matrix formalism.

    Parameters
    ----------
    correlation_bins : list[np.ndarray]
        Density-matrix correlation bins returned by a time-evolution routine.

    ops_same_time : list[np.ndarray]
        Superoperators used on the equal-time diagonal ``tau = 0``.

    ops_two_time : list[np.ndarray]
        Two-time superoperators used for ``tau > 0``.

    params : InputParams
        Simulation parameters.

    completion_print_flag : bool, default: False
        If ``True``, show the outer-loop progress bar during the calculation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Stack of correlation matrices with shape ``(n_ops, n_t, n_tau)`` and
        the corresponding time axis.
    """
    d_t_total = params.d_t_total
    d_t = int(np.prod(d_t_total) ** 2)

    time_bin_list_copy = copy.deepcopy(correlation_bins)
    right_s = time_bin_list_copy[-1]
    time_bin_list_copy = time_bin_list_copy[:-1]

    for i in range(len(ops_two_time)):
        ops_two_time[i] = reshape_liouvillian(ops_two_time[i], [np.sqrt(d_t), np.sqrt(d_t)])

    correlations = np.array(
        [np.zeros((len(time_bin_list_copy), len(time_bin_list_copy)), dtype=complex) for _ in ops_two_time]
    )

    tr_w = np.eye(int(np.sqrt(d_t)), dtype=np.complex128).reshape(-1)
    rights = [None] * (len(time_bin_list_copy) + 1)
    rights[len(time_bin_list_copy)] = right_s
    for k in range(len(time_bin_list_copy) - 1, -1, -1):
        rights[k] = absorb_right_env(rights[k + 1], time_bin_list_copy[k], tr_w)

    left = np.ones((1, 1))
    M_stack = np.array([np.einsum("a,b,abij->ij", tr_w, tr_w, op) for op in ops_two_time])

    outer_loop = _progress(
        range(len(time_bin_list_copy) - 1),
        enabled=completion_print_flag,
        total=max(len(time_bin_list_copy) - 1, 0),
        desc="Two time correlation",
    )
    for i in outer_loop:
        if i != 0:
            left = ncon([left, time_bin_list_copy[i - 1], tr_w], [[-1, 1], [1, 2, -3], [2]])

        for j in range(len(time_bin_list_copy) - i):
            i_1 = time_bin_list_copy[i]
            i_2 = time_bin_list_copy[i + j]
            right = rights[i + j + 1]

            if j == 0:
                for k in range(len(correlations)):
                    correlations[k][i, j] = expval_dm([left, i_1, right], ops_same_time[k])
            elif j == 1:
                state = ncon([i_1, i_2], [[-1, -2, 1], [1, -3, -4]])
                for k in range(len(correlations)):
                    correlations[k][i, j] = expval_twotime_dm([left, state, right], ops_two_time[k])
            else:
                if j == 2:
                    middle = ncon([time_bin_list_copy[i + j - 1], tr_w], [[-1, 2, -2], [2]])
                else:
                    middle = ncon([middle, time_bin_list_copy[i + j - 1], tr_w], [[-1, 1], [1, 2, -2], [2]])
                T = _collapse_T_with_middle(left, i_1, middle, i_2, right)
                correlations[:, i, j] = np.einsum("ij,kij->k", T, M_stack)

    t_list = np.arange(len(correlation_bins) - 1) * params.delta_t
    return correlations, t_list


def correlations_1t_dm(correlation_bins, ops_same_time, ops_two_time, params, t, completion_print_flag: bool = False):
    """
    Evaluate one time-slice of general density-matrix correlations.

    Parameters
    ----------
    correlation_bins : list[np.ndarray]
        Density-matrix correlation bins returned by a time-evolution routine.

    ops_same_time : list[np.ndarray]
        Superoperators used on the equal-time point ``tau = 0``.

    ops_two_time : list[np.ndarray]
        Two-time superoperators used for ``tau > 0``.

    params : InputParams
        Simulation parameters.

    t : float
        Time at which the first operator is evaluated.

    completion_print_flag : bool, default: False
        Currently unused. It is accepted for API compatibility with
        :func:`correlations_2t_dm`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Stack of correlation vectors with shape ``(n_ops, n_tau)`` and the
        corresponding time axis.
    """
    d_t_total = params.d_t_total
    d_t = int(np.prod(d_t_total) ** 2)

    time_bin_list_copy = copy.deepcopy(correlation_bins)
    right_s = time_bin_list_copy[-1]
    time_bin_list_copy = time_bin_list_copy[:-1]

    for i in range(len(ops_two_time)):
        ops_two_time[i] = reshape_liouvillian(ops_two_time[i], [np.sqrt(d_t), np.sqrt(d_t)])

    correlations = np.array([np.zeros((len(time_bin_list_copy)), dtype=complex) for _ in ops_two_time])

    tr_w = np.eye(int(np.sqrt(d_t)), dtype=np.complex128).reshape(-1)
    rights = []
    for i in range(len(time_bin_list_copy)):
        rights.append(ncon([trace_left(time_bin_list_copy[i:], tr_w), right_s], [[-1, 1], [1, -2]]))
    rights.append(right_s)

    left = np.ones((1, 1))
    M_stack = np.array([np.einsum("a,b,abij->ij", tr_w, tr_w, op) for op in ops_two_time])
    i = int(round(t / params.delta_t, 0))

    for _ in range(i):
        left = ncon([left, time_bin_list_copy[i - 1], tr_w], [[-1, 1], [1, 2, -3], [2]])

    for j in range(len(time_bin_list_copy) - i):
        i_1 = time_bin_list_copy[i]
        i_2 = time_bin_list_copy[i + j]
        right = rights[i + j + 1]

        if j == 0:
            for k in range(len(correlations)):
                correlations[k][j] = expval_dm([left, i_1, right], ops_same_time[k])
        elif j == 1:
            state = ncon([i_1, i_2], [[-1, -2, 1], [1, -3, -4]])
            for k in range(len(correlations)):
                correlations[k][j] = expval_twotime_dm([left, state, right], ops_two_time[k])
        else:
            if j == 2:
                middle = ncon([time_bin_list_copy[i + j - 1], tr_w], [[-1, 2, -2], [2]])
            else:
                middle = ncon([middle, time_bin_list_copy[i + j - 1], tr_w], [[-1, 1], [1, 2, -2], [2]])
            T = _collapse_T_with_middle(left, i_1, middle, i_2, right)
            correlations[:, j] = np.einsum("ij,kij->k", T, M_stack)

    t_list = np.arange(len(correlation_bins) - 1) * params.delta_t
    return correlations, t_list


def entanglement_dm(sch):
    """
    Compute von Neumann entanglement entropies from Schmidt coefficients.

    Parameters
    ----------
    sch : list[np.ndarray]
        Sequence of Schmidt coefficient arrays.

    Returns
    -------
    list[float]
        Entanglement entropies associated with each Schmidt spectrum.
    """
    ent_list = []
    for s in sch:
        sqrd_sch = np.trim_zeros(np.asarray(s) ** 2)
        if len(sqrd_sch) == 0:
            ent_list.append(0.0)
            continue
        ent_list.append(float(-np.sum(sqrd_sch * np.log2(sqrd_sch))))
    return ent_list


def _collapse_T_with_middle(left, i_1, middle, i_2, right):
    tmp = np.tensordot(left, i_1, axes=([1], [0]))
    tmp = np.tensordot(tmp, middle, axes=([2], [0]))
    tmp = np.tensordot(tmp, i_2, axes=([2], [0]))
    return np.tensordot(tmp, right, axes=([3], [0]))[0, :, :, 0]
