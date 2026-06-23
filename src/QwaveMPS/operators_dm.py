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
    Reshape a Liouvillian matrix into an interleaved site tensor.

    Parameters
    ----------
    Lmat : np.ndarray
        Liouvillian written as a matrix acting on the vectorized density matrix.

    dims : sequence of int
        Physical Hilbert-space dimensions of the subsystems before vectorization.

    Returns
    -------
    np.ndarray
        Tensor form of the Liouvillian with input and output indices grouped
        site by site, ready for tensor-network contractions.
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
    """
    Compute the expectation value of a single-time superoperator.

    Parameters
    ----------
    bin_in : list[np.ndarray]
        Density-matrix MPS bin written as ``[left, system, right]``.

    op_super : np.ndarray
        Superoperator acting on the vectorized physical density matrix.

    Returns
    -------
    complex
        Expectation value of ``op_super`` for the supplied bin.
    """
    left, sys, right = bin_in
    d_sys_phys = sys.shape[1]
    tr_s = np.eye(int(np.sqrt(d_sys_phys)), dtype=np.complex128).reshape(-1)
    return ncon(
        [left, sys, op_super, tr_s, right],
        [[-1, 1], [1, 2, 4], [2, 3], [3], [4, -2]],
    )[0, 0]


def expvals_dm(bin_in, op_super):
    """
    Evaluate a single superoperator on a sequence of bins.

    Parameters
    ----------
    bin_in : list[list[np.ndarray]]
        Sequence of density-matrix bins.

    op_super : np.ndarray
        Superoperator evaluated on each bin.

    Returns
    -------
    np.ndarray
        Array of expectation values, one per bin.
    """
    return np.array([expval_dm(b, op_super) for b in bin_in])


def expval_twotime_dm(bin_in, op_super):
    """
    Compute a two-time expectation value in the density-matrix formalism.

    Parameters
    ----------
    bin_in : list[np.ndarray]
        Two-site effective state written as ``[left, system, right]`` where the
        system tensor contains two physical time arguments.

    op_super : np.ndarray
        Two-time superoperator acting on the vectorized two-bin state.

    Returns
    -------
    complex
        Two-time expectation value of ``op_super``.
    """
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
    """
    Evaluate a two-time superoperator on a sequence of bins.

    Parameters
    ----------
    bin_in : list[list[np.ndarray]]
        Sequence of effective two-time bins.

    op_super : np.ndarray
        Two-time superoperator evaluated on each bin.

    Returns
    -------
    np.ndarray
        Array of two-time expectation values.
    """
    return np.array([expval_twotime_dm(b, op_super) for b in bin_in])


def single_time_expectation_dm(normalized_bins, ops_list):
    """
    Compute single-time expectation values for density-matrix bins.

    Parameters
    ----------
    normalized_bins : list[list[np.ndarray]]
        Density-matrix bins on which the observables are evaluated.

    ops_list : np.ndarray or list[np.ndarray]
        Single superoperator or list of superoperators.

    Returns
    -------
    np.ndarray
        Expectation values for each time bin. If ``ops_list`` is a list, the
        result has one row per operator.
    """
    if op_list_check(ops_list):
        return np.array([expvals_dm(normalized_bins, op) for op in ops_list])
    return expvals_dm(normalized_bins, ops_list)


def absorb_right_env(R, A, tr_w):
    """
    Absorb one physical site into a right environment tensor.

    Parameters
    ----------
    R : np.ndarray
        Right environment tensor.

    A : np.ndarray
        Site tensor to absorb.

    tr_w : np.ndarray
        Flattened trace vector for the physical Hilbert space.

    Returns
    -------
    np.ndarray
        Updated right environment.
    """
    return ncon([A, tr_w, R], [[-1, 1, 2], [1], [2, -2]])


def absorb_left_env(left, bin_tensor, tr_w):
    """
    Absorb one physical site into a left environment tensor.

    Parameters
    ----------
    left : np.ndarray
        Left environment tensor.

    bin_tensor : np.ndarray
        Site tensor to absorb.

    tr_w : np.ndarray
        Flattened trace vector for the physical Hilbert space.

    Returns
    -------
    np.ndarray
        Updated left environment.
    """
    return ncon([left, tr_w, bin_tensor], [[-1, 1], [2], [1, 2, -2]])


def trace_left(lefts, tr_w=None):
    """
    Trace out a sequence of density-matrix bins from the left.

    Parameters
    ----------
    lefts : list[np.ndarray]
        Sequence of site tensors to contract into a left environment.

    tr_w : np.ndarray, default: None
        Flattened trace vector for the physical Hilbert space. If ``None``, it
        is inferred from the first tensor in ``lefts``.

    Returns
    -------
    np.ndarray
        Left environment obtained after tracing out the supplied tensors.
    """
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
    """
    Convert a flattened density-matrix tensor back to matrix form.

    Parameters
    ----------
    rho_tensor : np.ndarray
        Tensor whose physical legs correspond to vectorized subsystem density
        matrices.

    dims : sequence of int
        Physical Hilbert-space dimensions of the subsystems before
        vectorization.

    bond_axes : tuple[int, ...], default: (0, -1)
        Bond axes to squeeze before reshaping the physical indices into a
        density matrix.

    Returns
    -------
    np.ndarray
        Density matrix in the standard Kronecker-product basis.
    """
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
    """
    Construct the row-major left-action superoperator ``rho -> op @ rho``.

    Parameters
    ----------
    op : np.ndarray
        Operator acting on the physical Hilbert space.

    Returns
    -------
    np.ndarray
        Superoperator satisfying
        ``vec(op @ rho) = (op kron I) vec(rho)`` for row-major ``vec``.
    """
    d = op.shape[0]
    return np.kron(op, np.eye(d, dtype=op.dtype))


def spost(op):
    """
    Construct the row-major right-action superoperator ``rho -> rho @ op``.

    Parameters
    ----------
    op : np.ndarray
        Operator acting on the physical Hilbert space.

    Returns
    -------
    np.ndarray
        Superoperator satisfying
        ``vec(rho @ op) = (I kron op.T) vec(rho)`` for row-major ``vec``.
    """
    d = op.shape[0]
    return np.kron(np.eye(d, dtype=op.dtype), op.T)


def lindblad_dissipator(a, gamma=1.0):
    """
    Construct the Lindblad dissipator for one collapse operator.

    Parameters
    ----------
    a : np.ndarray
        Collapse operator.

    gamma : float, default: 1
        Rate multiplying the dissipator.

    Returns
    -------
    np.ndarray
        Dissipative contribution to the Liouvillian generated by ``a``.
    """
    adag = a.conj().T
    adag_a = adag @ a
    return gamma * (
        spre(a) @ spost(adag)
        - 0.5 * spre(adag_a)
        - 0.5 * spost(adag_a)
    )


def liouvillian(H, c_ops=None):
    """
    Build the Liouvillian superoperator for a Hamiltonian and collapse terms.

    Parameters
    ----------
    H : np.ndarray
        Hamiltonian acting on the physical Hilbert space.

    c_ops : list[np.ndarray], default: None
        Collapse operators defining dissipative channels.

    Returns
    -------
    np.ndarray
        Liouvillian in matrix form acting on vectorized density matrices.
    """
    L = -1.0j * (spre(H) - spost(H))
    if c_ops is not None:
        L += sum(lindblad_dissipator(c_op) for c_op in c_ops)
    return L


def mps_site_to_mpo_site(A: np.ndarray) -> np.ndarray:
    """
    Convert a pure-state MPS site into the corresponding MPO site.

    Parameters
    ----------
    A : np.ndarray
        MPS site tensor with shape ``(chi_left, d, chi_right)``.

    Returns
    -------
    np.ndarray
        MPO site tensor with separate bra and ket physical indices.
    """
    chiL, d, chiR = A.shape
    W6 = A[:, None, :, None, :, None] * np.conj(A)[None, :, None, :, None, :]
    return W6.reshape(chiL * chiL, d, d, chiR * chiR)


def mpo_site_flatten_phys(W: np.ndarray) -> np.ndarray:
    """
    Flatten the physical bra and ket legs of an MPO site.

    Parameters
    ----------
    W : np.ndarray
        MPO site tensor with shape ``(chi_left, d, d, chi_right)``.

    Returns
    -------
    np.ndarray
        Site tensor with the physical indices flattened into a single
        vectorized density-matrix leg.
    """
    chiL2, d1, d2, chiR2 = W.shape
    if d1 != d2:
        raise ValueError("Expected equal physical dimensions when flattening MPO site.")
    return W.reshape(chiL2, d1 * d2, chiR2)


def convert_to_dm(inputs):
    """
    Convert pure-state MPS tensors to density-matrix tensors.

    Parameters
    ----------
    inputs : np.ndarray or list[np.ndarray]
        Single MPS site tensor or list of MPS site tensors.

    Returns
    -------
    np.ndarray or list[np.ndarray]
        Density-matrix representation with flattened physical legs. The return
        type matches the input structure.
    """
    if isinstance(inputs, list):
        return [mpo_site_flatten_phys(mps_site_to_mpo_site(W)) for W in inputs]
    return mpo_site_flatten_phys(mps_site_to_mpo_site(inputs))
