# schatten_norm_cpu.py

from cil.optimisation.functions import Function
from numba import njit, prange
import numpy as np

# -----------------------------------------------------------
# 1) NUMBA-ACCELERATED HELPERS (updated Hessian & gradient)
# -----------------------------------------------------------

@njit(parallel=True)
def cpu_nuc_norm_fair(x, eps):
    acc = 0.0
    nx, ny, nz, _, _ = x.shape
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                block = x[i, j, k]
                u, s, vt = np.linalg.svd(block, full_matrices=False)
                for idx in range(s.size):
                    σ = s[idx]
                    acc += eps * (σ / eps - np.log(1.0 + σ / eps))
    return acc

@njit(parallel=True)
def cpu_nuc_norm_fair_no_sum(x, eps, out):
    nx, ny, nz, _, _ = x.shape
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                block = x[i, j, k]
                u, s, vt = np.linalg.svd(block, full_matrices=False)
                acc = 0.0
                for idx in range(s.size):
                    σ = s[idx]
                    acc += eps * (σ / eps - np.log(1.0 + σ / eps))
                out[i, j, k] = acc

@njit(parallel=True)
def cpu_nuc_norm_charbonnier(x, eps):
    acc = 0.0
    nx, ny, nz, _, _ = x.shape
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                block = x[i, j, k]
                u, s, vt = np.linalg.svd(block, full_matrices=False)
                for idx in range(s.size):
                    σ = s[idx]
                    acc += np.sqrt(σ * σ + eps * eps) - eps
    return acc

@njit(parallel=True)
def cpu_nuc_norm_charbonnier_no_sum(x, eps, out):
    nx, ny, nz, _, _ = x.shape
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                block = x[i, j, k]
                u, s, vt = np.linalg.svd(block, full_matrices=False)
                acc = 0.0
                for idx in range(s.size):
                    σ = s[idx]
                    acc += np.sqrt(σ * σ + eps * eps) - eps
                out[i, j, k] = acc

@njit(parallel=True)
def cpu_nuc_norm_gradient_fair(x, eps, out_grad):
    """
    Gradient of ∑ g(σ_i) with g(σ)=eps*(σ/eps − log(1+σ/eps)).
    Correct derivative w.r.t M is U diag(g'(σ)/σ) Vᵀ.
    Here g'(σ)=σ/(ε+σ) ⇒ g'(σ)/σ=1/(ε+σ).
    """
    nx, ny, nz, M, d = x.shape
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                block = x[i, j, k]
                u, s, vt = np.linalg.svd(block, full_matrices=False)
                r = s.size
                diag_vals = np.zeros(r, dtype=block.dtype)
                for idx in range(r):
                    σ = s[idx]
                    diag_vals[idx] = 1.0 / (eps + σ)  # = g'(σ)/σ
                temp = np.zeros((M, r), dtype=block.dtype)
                for p in range(M):
                    for q in range(r):
                        temp[p, q] = u[p, q] * diag_vals[q]
                grad_block = temp @ vt
                for p in range(M):
                    for q in range(d):
                        out_grad[i, j, k, p, q] = grad_block[p, q]

@njit(parallel=True)
def cpu_nuc_norm_hessian_fair(x, eps, out_hess):
    """
    Hessian‐diag of ∑ g(σ_i) w.r.t. entries of M.
    We need varphi''(μ)+2·varphi'(μ) at μ=σ².
    For fair: g'(σ)=σ/(ε+σ), g''(σ)=ε/(ε+σ)².
    ⇒ term1 = g''/(4σ²) = ε/(4σ² (ε+σ)²)
    ⇒ term2 = ((4σ²−1)/(4σ³))·g'(σ) = (4σ²−1)/(4σ³)·(σ/(ε+σ)) = (4σ²−1)/(4σ² (ε+σ))
    Sum = ε/(4σ² (ε+σ)²) + (4σ²−1)/(4σ² (ε+σ)).
    """
    nx, ny, nz, M, d = x.shape
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                block = x[i, j, k]
                u, s, vt = np.linalg.svd(block, full_matrices=False)
                r = s.size
                diag_vals = np.zeros(r, dtype=block.dtype)
                for idx in range(r):
                    σ = s[idx]
                    if σ == 0.0:
                        diag_vals[idx] = 0.0
                    else:
                        # g'' = ε/(ε+σ)²
                        gpp = eps / ((eps + σ) * (eps + σ))
                        # g' = σ/(ε+σ)
                        gp = σ / (eps + σ)
                        term1 = gpp / (4.0 * σ * σ)
                        term2 = ((4.0 * σ * σ - 1.0) / (4.0 * σ * σ * σ)) * gp
                        diag_vals[idx] = term1 + term2
                temp = np.zeros((M, r), dtype=block.dtype)
                for p in range(M):
                    for q in range(r):
                        temp[p, q] = u[p, q] * diag_vals[q]
                hess_block = temp @ vt
                for p in range(M):
                    for q in range(d):
                        out_hess[i, j, k, p, q] = hess_block[p, q]

@njit(parallel=True)
def cpu_nuc_norm_gradient_charbonnier(x, eps, out_grad):
    """
    Gradient of ∑ g(σ) with g(σ)=√(σ²+ε²)−ε.
    g'(σ)=σ/√(σ²+ε²), so g'(σ)/σ = 1/√(σ²+ε²).
    """
    nx, ny, nz, M, d = x.shape
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                block = x[i, j, k]
                u, s, vt = np.linalg.svd(block, full_matrices=False)
                r = s.size
                diag_vals = np.zeros(r, dtype=block.dtype)
                for idx in range(r):
                    σ = s[idx]
                    diag_vals[idx] = 1.0 / np.sqrt(σ * σ + eps * eps)  # = g'(σ)/σ
                temp = np.zeros((M, r), dtype=block.dtype)
                for p in range(M):
                    for q in range(r):
                        temp[p, q] = u[p, q] * diag_vals[q]
                grad_block = temp @ vt
                for p in range(M):
                    for q in range(d):
                        out_grad[i, j, k, p, q] = grad_block[p, q]

@njit(parallel=True)
def cpu_nuc_norm_hessian_charbonnier(x, eps, out_hess):
    """
    Hessian‐diag of ∑ g(σ) with g(σ)=√(σ²+ε²)−ε.
    g'(σ)=σ/√(σ²+ε²), g''(σ)=ε²/(σ²+ε²)^(3/2).
    term1 = g''/(4σ²) = [ε²/(σ²+ε²)^(3/2)] / (4σ²)
    term2 = ((4σ²−1)/(4σ³))·g'(σ) = (4σ²−1)/(4σ³)·(σ/√(σ²+ε²))
    """
    nx, ny, nz, M, d = x.shape
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                block = x[i, j, k]
                u, s, vt = np.linalg.svd(block, full_matrices=False)
                r = s.size
                diag_vals = np.zeros(r, dtype=block.dtype)
                for idx in range(r):
                    σ = s[idx]
                    if σ == 0.0:
                        diag_vals[idx] = 0.0
                    else:
                        gpp = (eps * eps) / ((σ * σ + eps * eps) ** (1.5))
                        gp = σ / np.sqrt(σ * σ + eps * eps)
                        term1 = gpp / (4.0 * σ * σ)
                        term2 = ((4.0 * σ * σ - 1.0) / (4.0 * σ * σ * σ)) * gp
                        diag_vals[idx] = term1 + term2
                temp = np.zeros((M, r), dtype=block.dtype)
                for p in range(M):
                    for q in range(r):
                        temp[p, q] = u[p, q] * diag_vals[q]
                hess_block = temp @ vt
                for p in range(M):
                    for q in range(d):
                        out_hess[i, j, k, p, q] = hess_block[p, q]

@njit(parallel=True)
def cpu_nuc_norm(x):
    acc = 0.0
    nx, ny, nz, _, _ = x.shape
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                block = x[i, j, k]
                _, s, _ = np.linalg.svd(block, full_matrices=False)
                for idx in range(s.size):
                    acc += s[idx]
    return acc

@njit(parallel=True)
def cpu_nuc_norm_no_sum(x, out):
    nx, ny, nz, _, _ = x.shape
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                block = x[i, j, k]
                _, s, _ = np.linalg.svd(block, full_matrices=False)
                acc = 0.0
                for idx in range(s.size):
                    acc += s[idx]
                out[i, j, k] = acc

@njit(parallel=True)
def cpu_nuc_norm_proximal(x, tau, out):
    nx, ny, nz, M, d = x.shape
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                block = x[i, j, k]
                u, s, vt = np.linalg.svd(block, full_matrices=False)
                r = s.size
                diag_vals = np.zeros(r, dtype=block.dtype)
                for idx in range(r):
                    val = s[idx] - tau
                    diag_vals[idx] = val if val > 0.0 else 0.0
                temp = np.zeros((M, r), dtype=block.dtype)
                for p in range(M):
                    for q in range(r):
                        temp[p, q] = u[p, q] * diag_vals[q]
                prox_block = temp @ vt
                for p in range(M):
                    for q in range(d):
                        out[i, j, k, p, q] = prox_block[p, q]

@njit(parallel=True)
def cpu_nuc_norm_convex_conjugate(x):
    acc = 0.0
    nx, ny, nz, _, _ = x.shape
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                block = x[i, j, k]
                _, s, _ = np.linalg.svd(block, full_matrices=False)
                for idx in range(s.size):
                    σ = s[idx]
                    acc += (σ if σ < 1.0 else 1.0)
    return acc


# -----------------------------------------------------------
# 2) CLASS DEFINITION (CPU-only)
# -----------------------------------------------------------

class CPUVectorialTotalVariation(Function):
    """
    CPU-only fallback implementing Vectorial TV via numba‐accelerated small‐SVD loops.
    Available smoothing: 'fair', 'charbonnier', or None (pure nuclear norm).
    """

    def __init__(self, eps, smoothing_function='fair'):
        self.eps = eps
        self.smoothing_function = smoothing_function

    def __call__(self, x):
        if self.smoothing_function == 'fair':
            return cpu_nuc_norm_fair(x, self.eps)
        elif self.smoothing_function == 'charbonnier':
            return cpu_nuc_norm_charbonnier(x, self.eps)
        else:
            return cpu_nuc_norm(x)

    def call_no_sum(self, x, out=None):
        if out is None:
            out = np.zeros(x.shape[:3], dtype=x.dtype)
        if self.smoothing_function == 'fair':
            cpu_nuc_norm_fair_no_sum(x, self.eps, out)
        elif self.smoothing_function == 'charbonnier':
            cpu_nuc_norm_charbonnier_no_sum(x, self.eps, out)
        else:
            cpu_nuc_norm_no_sum(x, out)
        return out

    def gradient(self, x, out_grad=None):
        if out_grad is None:
            out_grad = np.zeros(x.shape, dtype=x.dtype)
        if self.smoothing_function == 'fair':
            cpu_nuc_norm_gradient_fair(x, self.eps, out_grad)
        elif self.smoothing_function == 'charbonnier':
            cpu_nuc_norm_gradient_charbonnier(x, self.eps, out_grad)
        else:
            raise ValueError("Gradient only defined if smoothing_function is 'fair' or 'charbonnier'")
        return out_grad

    def hessian(self, x, out_hess=None):
        if out_hess is None:
            out_hess = np.zeros(x.shape, dtype=x.dtype)
        if self.smoothing_function == 'fair':
            cpu_nuc_norm_hessian_fair(x, self.eps, out_hess)
        elif self.smoothing_function == 'charbonnier':
            cpu_nuc_norm_hessian_charbonnier(x, self.eps, out_hess)
        else:
            raise ValueError("Hessian only defined if smoothing_function is 'fair' or 'charbonnier'")
        return out_hess

    def proximal(self, x, tau, out_prox=None):
        if out_prox is None:
            out_prox = np.zeros(x.shape, dtype=x.dtype)
        cpu_nuc_norm_proximal(x, tau, out_prox)
        return out_prox

    def convex_conjugate(self, x):
        return cpu_nuc_norm_convex_conjugate(x)
