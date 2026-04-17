#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Density-matrix and superoperator utilities for QwaveMPS.
"""

import numpy as np
from ncon import ncon

from QwaveMPS.operators import op_list_check

__all__ = [
    "reshape_liouvillian",
    "single_time_expectation_dm",
    "expval_dm",
    "expvals_dm",
    "expval_twotime_dm",
    "expvals_twotime_dm",
    "absorb_right_env",
    "absorb_left_env",
    "trace_left",
    "tensor_to_rho",
    "spre",
    "spost",
    "lindblad_dissipator",
    "liouvillian",
    "convert_to_dm",
]


def reshape_liouvillian(Lmat: np.ndarray, dims) -> np.ndarray:
    """
    Reshape a Liouvillian matrix into a flattened site tensor.
    """
    dims = np.asarray(dims, dtype=int)
    if np.any(dims <= 0):
        raise ValueError("All subsystem dimensions must be positive.")
    n = len(dims)

    d = int(np.prod(dims))
    if Lmat.shape != (d * d, d * d):
        raise ValueError(f"Expected Lmat shape {(d*d, d*d)}, got {Lmat.shape}")

    L4 = Lmat.reshape(d, d, d, d)
    L = L4.reshape(*dims, *dims, *dims, *dims)

    perm = []
    for i in range(n):
        perm.append(n + i)
        perm.append(i)
    for i in range(n):
        perm.append(3 * n + i)
        perm.append(2 * n + i)

    Lp = np.transpose(L, perm)
    out_shape = (dims * dims).tolist()
    in_shape = (dims * dims).tolist()
    return Lp.reshape(*out_shape, *in_shape)


def expval_dm(bin_in, op_super):
    left, sys, right = bin_in
    d_sys_phys = sys.shape[1]
    tr_s = np.eye(int(np.sqrt(d_sys_phys)), dtype=np.complex128).reshape(-1)
    return ncon(
        [left, sys, op_super, tr_s, right],
        [[-1, 1], [1, 2, 4], [2, 3], [3], [4, -2]],
    )[0, 0]


def expvals_dm(bin_in, op_super):
    return np.array([expval_dm(b, op_super) for b in bin_in])


def expval_twotime_dm(bin_in, op_super):
    left, sys, right = bin_in

    d_sys_phys_1 = sys.shape[1]
    d_sys_phys_2 = sys.shape[2]
    tr_s1 = np.eye(int(np.sqrt(d_sys_phys_1)), dtype=np.complex128).reshape(-1)
    tr_s2 = np.eye(int(np.sqrt(d_sys_phys_2)), dtype=np.complex128).reshape(-1)

    return ncon(
        [left, sys, op_super, tr_s1, tr_s2, right],
        [[-1, 1], [1, 2, 3, 4], [5, 6, 2, 3], [5], [6], [4, -2]],
    )[0, 0]


def expvals_twotime_dm(bin_in, op_super):
    return np.array([expval_twotime_dm(b, op_super) for b in bin_in])


def single_time_expectation_dm(normalized_bins, ops_list):
    """
    Density-matrix version of single-time expectation values.
    """
    if op_list_check(ops_list):
        return np.array([expvals_dm(normalized_bins, op) for op in ops_list])
    return expvals_dm(normalized_bins, ops_list)


def absorb_right_env(R, A, tr_w):
    return ncon([A, tr_w, R], [[-1, 1, 2], [1], [2, -2]])


def absorb_left_env(left, bin_tensor, tr_w):
    return ncon([left, tr_w, bin_tensor], [[-1, 1], [2], [1, 2, -2]])


def trace_left(lefts, tr_w=None):
    if len(lefts) == 0:
        return np.ones((1, 1))

    d_w_phys = lefts[0].shape[1]
    if tr_w is None:
        tr_w = np.eye(int(np.sqrt(d_w_phys)), dtype=np.complex128).reshape(-1)

    left = ncon([lefts[0], tr_w], [[-1, 1, -2], [1]])
    for i in range(1, len(lefts)):
        left = ncon([left, tr_w, lefts[i]], [[-1, 1], [2], [1, 2, -2]])
    return left


def tensor_to_rho(rho_tensor, dims, bond_axes=(0, -1)):
    dims = np.asarray(dims, dtype=int)
    if np.any(dims <= 0):
        raise ValueError("All subsystem dimensions must be positive.")

    n = len(dims)
    d_total = int(np.prod(dims))
    x = rho_tensor

    if bond_axes is not None and len(bond_axes) > 0:
        norm_axes = []
        for ax in bond_axes:
            norm_axes.append(ax if ax >= 0 else x.ndim + ax)
        x = np.squeeze(x, axis=tuple(sorted(set(norm_axes))))

    expected_shape = tuple((dims * dims).tolist())
    if x.shape != expected_shape:
        raise ValueError(
            f"After squeezing bond axes, expected physical shape {expected_shape}, got {x.shape}."
        )

    rho2n = x.reshape(*sum(([int(d), int(d)] for d in dims), []))
    perm = list(range(0, 2 * n, 2)) + list(range(1, 2 * n, 2))
    rho_kb = np.transpose(rho2n, perm)
    return rho_kb.reshape(d_total, d_total)


def spre(op):
    return np.kron(np.identity(np.shape(op)[0]), op)


def spost(op):
    return np.kron(op, np.identity(np.shape(op)[0]))


def lindblad_dissipator(a):
    ad_a = np.transpose(np.conj(a)) @ a
    return spre(a) @ spost(np.transpose(np.conj(a))) - 0.5 * spre(ad_a) - 0.5 * spost(ad_a)


def liouvillian(H, c_ops=None):
    L = -1.0j * (spre(H) - spost(np.transpose(H)))
    if c_ops is not None:
        L += sum(lindblad_dissipator(c_op) for c_op in c_ops)
    return L


def mps_site_to_mpo_site(A: np.ndarray) -> np.ndarray:
    chiL, d, chiR = A.shape
    W6 = A[:, None, :, None, :, None] * np.conj(A)[None, :, None, :, None, :]
    return W6.reshape(chiL * chiL, d, d, chiR * chiR)


def mpo_site_flatten_phys(W: np.ndarray) -> np.ndarray:
    chiL2, d1, d2, chiR2 = W.shape
    if d1 != d2:
        raise ValueError("Expected equal physical dimensions when flattening MPO site.")
    return W.reshape(chiL2, d1 * d2, chiR2)


def convert_to_dm(inputs):
    if isinstance(inputs, list):
        return [mpo_site_flatten_phys(mps_site_to_mpo_site(W)) for W in inputs]
    return mpo_site_flatten_phys(mps_site_to_mpo_site(inputs))
