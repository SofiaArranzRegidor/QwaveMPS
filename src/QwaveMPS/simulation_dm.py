#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Density-matrix time evolution routines for QwaveMPS.
"""

import numpy as np
from ncon import ncon
from scipy.linalg import expm, svd
from tqdm import tqdm

from QwaveMPS.operators import swap
from QwaveMPS.parameters import Bins
from QwaveMPS.states import vacuum as vacuum_state
from QwaveMPS.states import wg_ground

from QwaveMPS.operators_dm import (
    absorb_left_env,
    absorb_right_env,
    convert_to_dm,
    reshape_liouvillian,
    trace_left,
)

__all__ = ["t_evol_mar_dm", "t_evol_nmar_dm"]


def _svd_tensor_dm(tensor: np.ndarray, bond_max: int, d1: int, d2: int, tol: float = 0):
    chiL = tensor.shape[0]
    chiR = tensor.shape[-1]

    reshaped = tensor.reshape(chiL * d1, d2 * chiR)
    U, S, Vh = svd(reshaped, full_matrices=False)

    if len(S) == 0:
        raise RuntimeError("SVD returned empty singular values.")

    if tol > 0:
        s0 = np.sum(S)
        if s0 == 0:
            chi_cutoff = 1
        else:
            chi_cutoff = max(1, int(np.sum(S / s0 >= tol)))
    else:
        chi_cutoff = len(S)

    chi = min(bond_max, chi_cutoff)
    S = S[:chi]
    U = U[:, :chi].reshape(chiL, d1, chi)
    Vt = Vh[:chi, :].reshape(chi, d2, chiR)
    return U, S, Vt


def _prepare_dm_input_bins(i_n0, params):
    if i_n0 is not None:
        return i_n0
    return convert_to_dm(vacuum_state(params.tmax, params))


def _progress(iterable, total: int, desc: str):
    return tqdm(iterable, total=total, desc=desc, unit="step", dynamic_ncols=True)


def t_evol_mar_dm(L: np.ndarray, i_s0: np.ndarray, i_n0, params):
    delta_t = params.delta_t
    tmax = params.tmax
    bond = params.bond_max
    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total

    i_ns = _prepare_dm_input_bins(i_n0, params)
    d_t = int(np.prod(d_t_total) ** 2)
    d_sys = int(np.prod(d_sys_total) ** 2)

    sbins = []
    tbins = []
    tbins_in = []
    cor_list = []
    schmidt = []

    vacuum_dm = convert_to_dm(wg_ground(int(np.prod(d_t_total))))
    sbins.append([np.ones((1, 1)), i_s0, np.ones((1, 1))])
    tbins.append([np.ones((1, 1)), vacuum_dm, np.ones((1, 1))])
    tbins_in.append([np.ones((1, 1)), vacuum_dm, np.ones((1, 1))])
    schmidt.append(np.zeros(1))

    n = int(round(tmax / delta_t, 0))
    U_swap = swap(d_sys, d_t)
    U = reshape_liouvillian(expm(L * delta_t), [np.sqrt(d_sys), np.sqrt(d_t)])

    tr_w = np.eye(int(np.sqrt(d_t)), dtype=np.complex128).reshape(-1)
    tr_s = np.eye(int(np.sqrt(d_sys)), dtype=np.complex128).reshape(-1)

    left_iter = np.ones((1, 1))
    R_suffix = [None] * (len(i_ns) + 1)
    R_suffix[len(i_ns)] = np.ones((1, 1), dtype=np.complex128)
    for k in range(len(i_ns) - 1, -1, -1):
        R_suffix[k] = absorb_right_env(R_suffix[k + 1], i_ns[k], tr_w)

    i_s = i_s0
    for k in _progress(range(n), total=n, desc="t_evol_mar_dm"):
        i_nk = i_ns[k]

        phi1 = ncon([i_s, i_nk], [[-1, -2, 1], [1, -3, -4]])
        i_s, stemp, i_nk = _svd_tensor_dm(phi1, bond, d_sys, d_t)
        i_nk = stemp[:, None, None] * i_nk
        tbins_in.append(
            [ncon([left_iter, i_s, tr_s], [[-1, 1], [1, 2, -2], [2]]), i_nk, R_suffix[k + 1]]
        )

        phi1 = ncon([i_s, i_nk, U], [[-1, 2, 3], [3, 4, -4], [-2, -3, 2, 4]])
        i_s, stemp, i_n = _svd_tensor_dm(phi1, bond, d_sys, d_t)
        i_s = i_s * stemp[None, None, :]

        right_with_in = absorb_right_env(R_suffix[k + 1], i_n, tr_w)
        sbins.append([left_iter, i_s, right_with_in])
        tbins.append(
            [ncon([left_iter, i_s, tr_s], [[-1, 1], [1, 2, -2], [2]]), i_n, R_suffix[k + 1]]
        )

        phi2 = ncon([i_s, i_n, U_swap], [[-1, 5, 2], [2, 6, -4], [-2, -3, 5, 6]])
        i_n, stemp, i_st = _svd_tensor_dm(phi2, bond, d_t, d_sys)
        i_s = stemp[:, None, None] * i_st

        schmidt.append(stemp)
        cor_list.append(i_n)
        left_iter = ncon([left_iter, i_n, tr_w], [[-1, 1], [1, 2, -2], [2]])

    cor_list[-1] = i_n * stemp[None, None, :]
    cor_list.append(ncon([i_st, tr_s], [[-1, 2, -3], [2]]))

    return Bins(
        system_states=sbins,
        output_field_states=tbins,
        input_field_states=tbins_in,
        correlation_bins=cor_list,
        schmidt=schmidt,
    )


def t_evol_nmar_dm(L: np.ndarray, i_s0: np.ndarray, i_n0, params):
    delta_t = params.delta_t
    tmax = params.tmax
    bond = params.bond_max
    d_t_total = params.d_t_total
    d_sys_total = params.d_sys_total
    tau = params.tau

    i_ns = _prepare_dm_input_bins(i_n0, params)
    d_t = int(np.prod(d_t_total) ** 2)
    d_sys = int(np.prod(d_sys_total) ** 2)

    sbins = []
    tbins = []
    tbins_in = []
    taubins = []
    cor_list = []
    schmidt = []
    schmidt_tau = []

    sbins.append([np.ones((1, 1)), i_s0, np.ones((1, 1))])
    tbins.append([np.ones((1, 1)), i_ns[0], trace_left(i_ns[1:])])
    tbins_in.append([np.ones((1, 1)), i_ns[0], trace_left(i_ns[1:])])
    taubins.append([np.ones((1, 1)), i_ns[0], trace_left(i_ns[1:])])
    schmidt.append(np.zeros(1))
    schmidt_tau.append(np.zeros(1))

    n = int(round(tmax / delta_t, 0))
    l = int(round(tau / delta_t, 0))

    swap_t_t = swap(d_t, d_t)
    U_swap = swap(d_sys, d_t)
    U = reshape_liouvillian(expm(L * delta_t), [np.sqrt(d_t), np.sqrt(d_sys), np.sqrt(d_t)])

    nbins = [i_ns[0] for _ in range(l)]
    i_stemp = i_s0

    tr_w = np.eye(int(np.sqrt(d_t)), dtype=np.complex128).reshape(-1)
    tr_s = np.eye(int(np.sqrt(d_sys)), dtype=np.complex128).reshape(-1)

    R_suffix = [None] * (n + 1)
    R_suffix[n] = np.ones((1, 1), dtype=np.complex128)
    for k in range(n - 1, -1, -1):
        R_suffix[k] = absorb_right_env(R_suffix[k + 1], i_ns[k], tr_w)

    left_iter = np.ones((1, 1))
    bin_sys_corr_list_tau = []
    bin_sys_corr_list = []

    for k in _progress(range(n), total=n, desc="t_evol_nmar_dm"):
        i_tau = nbins[k]
        for i in range(k, k + l - 1):
            i_n = nbins[i + 1]
            swaps = ncon([i_tau, i_n, swap_t_t], [[-1, 5, 2], [2, 6, -4], [-2, -3, 5, 6]])
            i_n2, stemp, i_t = _svd_tensor_dm(swaps, bond, d_t, d_t)
            i_tau = ncon([np.diag(stemp), i_t], [[-1, 1], [1, -3, -4]])
            nbins[i] = i_n2

        i_1 = ncon([i_tau, i_stemp], [[-1, -2, 1], [1, -3, -4]])
        i_t, stemp, i_stemp = _svd_tensor_dm(i_1, bond, d_t, d_sys)
        i_s = stemp[:, None, None] * i_stemp

        i_nk = i_ns[k]
        phi1 = ncon([i_s, i_nk], [[-1, -2, 1], [1, -3, -4]])
        i_s, stemp, i_nk = _svd_tensor_dm(phi1, bond, d_sys, d_t)
        i_nk = stemp[:, None, None] * i_nk

        right = R_suffix[k + 1]
        bins_tr = trace_left(nbins[k:-1], tr_w)
        left_total = ncon([left_iter, bins_tr], [[-1, 1], [1, -2]])
        left = absorb_left_env(left_total, i_t, tr_w)
        tbins_in.append([left, i_nk, right])

        phi1 = ncon(
            [i_t, i_s, i_nk, U],
            [[-1, 3, 1], [1, 4, 2], [2, 5, -5], [-2, -3, -4, 3, 4, 5]],
        )
        i_t, stemp, i_2 = _svd_tensor_dm(phi1, bond, d_t, d_t * d_sys)
        i_2 = stemp[:, None, None] * i_2
        i_stemp, stemp, i_n = _svd_tensor_dm(i_2, bond, d_sys, d_t)
        i_s = i_stemp * stemp[None, None, :]

        left = absorb_left_env(left_total, i_t, tr_w)
        bin_sys_corr_list.append([left, ncon([i_s, i_n], [[-1, -2, 1], [1, -3, -4]]), right])

        right = absorb_right_env(right, i_n, tr_w)
        bin_sys_corr_list_tau.append([left_total, ncon([i_t, i_s], [[-1, -2, 1], [1, -3, -4]]), right])
        sbins.append([left, i_s, right])

        phi2 = ncon([i_s, i_n, U_swap], [[-1, 3, 2], [2, 4, -4], [-2, -3, 3, 4]])
        i_n, stemp, i_stemp = _svd_tensor_dm(phi2, bond, d_t, d_sys)

        i_n = i_n * stemp[None, None, :]
        cont = ncon([i_t, i_n], [[-1, -2, 1], [1, -3, -4]])
        i_t, stemp, i_n = _svd_tensor_dm(cont, bond, d_t, d_t)
        i_tau = i_t * stemp[None, None, :]

        left = absorb_left_env(left_total, i_tau, tr_w)
        right = absorb_right_env(R_suffix[k + 1], i_stemp, tr_w)
        tbins.append([left, i_n, right])

        right = absorb_right_env(right, i_n, tr_w)
        taubins.append([left_total, i_tau, right])
        schmidt.append(stemp)

        nbins[k + l - 1] = i_tau
        nbins.append(i_n)

        for i in range(k + l - 1, k, -1):
            i_n = nbins[i - 1]
            swaps = ncon([i_n, i_tau, swap_t_t], [[-1, 5, 2], [2, 6, -4], [-2, -3, 5, 6]])
            i_t, stemp, i_n2 = _svd_tensor_dm(swaps, bond, d_t, d_t)
            i_tau = i_t * stemp[None, None, :]
            nbins[i] = i_n2

        schmidt_tau.append(stemp)
        nbins[k] = i_t
        nbins[k + 1] = stemp[:, None, None] * i_n2
        left_iter = absorb_left_env(left_iter, nbins[k], tr_w)
        cor_list.append(i_t)

    cor_list[-1] = i_t * stemp[None, None, :]
    nbins_con = trace_left(nbins[n:], tr_w)
    out = ncon([nbins_con, i_stemp, tr_s], [[-1, 3], [3, 4, -2], [4]])
    cor_list.append(out)

    bins = Bins(
        system_states=sbins,
        loop_field_states=tbins,
        output_field_states=taubins,
        input_field_states=tbins_in,
        correlation_bins=cor_list,
        schmidt=schmidt,
        schmidt_tau=schmidt_tau,
    )
    bins.bin_sys_corr_list_tau = bin_sys_corr_list_tau
    bins.bin_sys_corr_list = bin_sys_corr_list
    return bins
