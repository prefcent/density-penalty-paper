"""The torus kernel applied by fast Fourier transform.

Same kernel as ``common.torus_kernel(scale)`` -- K_ij = (scale * c_ij + d0)^(-beta) with
c_ij the periodic Euclidean distance and a zero diagonal -- applied as a 2D circular
convolution, so the N x N matrix is never formed. It is passed to prefcent through the
public ``OperatorKernel`` interface. Its identity is the SHA-256 of the generating 2D row,
n and a layout tag; the row determines the operator, so the identity is declared trusted.
"""
from __future__ import annotations

import hashlib

import numpy as np

import prefcent as pc


def torus_fft_kernel(n: int, cell_m: float, scale: float, d0: float, beta: float) -> pc.OperatorKernel:
    i = np.arange(n)
    w = np.minimum(i, n - i) * cell_m
    dist = np.sqrt(w[:, None] ** 2 + w[None, :] ** 2)
    row = (dist * scale + d0) ** (-beta)
    row[0, 0] = 0.0                                  # no self-interaction
    rf = np.fft.rfft2(row)
    h = hashlib.sha256(row.tobytes())
    h.update(np.asarray([n], dtype=np.int64).tobytes())
    h.update(b"layout:torus2d-circulant")
    ident = {"kind": "consumer", "value": f"torus2d-fft:{h.hexdigest()}", "trusted": True}

    def matvec(x: np.ndarray) -> np.ndarray:
        y = np.fft.irfft2(rf * np.fft.rfft2(x.reshape(n, n)), s=(n, n))
        return np.ascontiguousarray(y.reshape(-1), dtype=np.float64)

    return pc.OperatorKernel.from_callables(matvec, None, n * n, is_symmetric=True,
                                            self_interaction=False, identity=ident)
