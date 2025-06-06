# schatten_norm_cpu.py

from cil.optimisation.functions import Function
from numba import njit, prange
import numpy as np

# -----------------------------------------------------------
# 1) NUMBA-ACCELERATED HELPERS (Corrected Gradient and New Hessian)
# -----------------------------------------------------------

# --- Helper functions for smoothing terms ---
@njit
def h_fair(s, eps):
    return eps * (s / (eps + 1e-9) - np.log(1.0 + s / (eps + 1e-9)))

@njit
def h_prime_fair(s, eps): # h'(s)
    return s / (eps + s + 1e-9)

@njit
def h_double_prime_fair(s, eps): # h''(s)
    return eps / (eps + s + 1e-9)**2

@njit
def h_charbonnier(s, eps):
    return np.sqrt(s**2 + eps**2) - eps

@njit
def h_prime_charbonnier(s, eps): # h'(s)
    return s / np.sqrt(s**2 + eps**2 + 1e-9)

@njit
def h_double_prime_charbonnier(s, eps): # h''(s)
    return eps**2 / (s**2 + eps**2 + 1e-9)**1.5

# --- Value Calculation ---
@njit(parallel=True)
def cpu_nuc_norm_sum(x, h_func, eps):
    acc = 0.0
    nx, ny, nz, _, _ = x.shape
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                block = x[i, j, k]
                u, s, vt = np.linalg.svd(block, full_matrices=False)
                for idx in range(s.size):
                    acc += h_func(s[idx], eps)
    return acc

# --- Corrected Gradient Calculation ---
@njit(parallel=True)
def cpu_gradient(x, h_prime_func, eps, out_grad):
    """
    Corrected Gradient of ∑ h(σ_i).
    Derivative w.r.t matrix A is U @ diag(h'(σ)) @ V^T.
    """
    nx, ny, nz, M, d = x.shape
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                block = x[i, j, k]
                u, s, vt = np.linalg.svd(block, full_matrices=False)
                r = s.size
                
                # Directly compute h'(s_k)
                diag_vals = h_prime_func(s, eps)
                
                # Reconstruct: U @ diag(h'(s)) @ V^T
                temp = u * diag_vals # equivalent to u @ diag(diag_vals) for r=M
                grad_block = temp @ vt
                
                # Store result
                out_grad[i, j, k] = grad_block

# --- New Hessian Components Calculation ---
@njit(parallel=True)
def cpu_hessian_components(x, h_double_prime_func, eps, out_hess_coeffs, out_rank_one_fields):
    """
    Calculates the building blocks for the diagonal preconditioner.
    1. Hessian coefficients: h''(s_k)
    2. Basis matrices: u_k v_k^T for each singular mode k
    """
    nx, ny, nz, M, d = x.shape
    for i in prange(nx):
        for j in prange(ny):
            for k in prange(nz):
                block = x[i, j, k]
                u, s, vt = np.linalg.svd(block, full_matrices=False)
                r = s.size

                # 1. Calculate and store Hessian coefficients h''(s_k)
                hess_coeffs_k = h_double_prime_func(s, eps)
                out_hess_coeffs[i, j, k, :r] = hess_coeffs_k

                # 2. Calculate and store rank-1 matrices u_k v_k^T
                for k_idx in range(r):
                    uk = u[:, k_idx:k_idx+1]      # Shape (M, 1)
                    vk_t = vt[k_idx:k_idx+1, :]   # Shape (1, d)
                    
                    # Store outer product u_k @ v_k^T
                    out_rank_one_fields[i, j, k, k_idx, :, :] = uk @ vk_t


# --- Proximal Calculation (Unchanged) ---
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
                
                temp = u * diag_vals # u @ diag(...)
                prox_block = temp @ vt
                out[i, j, k] = prox_block


# -----------------------------------------------------------
# 2) CLASS DEFINITION (CPU-only) - Updated
# -----------------------------------------------------------

class CPUVectorialTotalVariation(Function):
    """
    CPU-only fallback implementing Vectorial TV via numba‐accelerated small‐SVD loops.
    Available smoothing: 'fair', 'charbonnier', or None (pure nuclear norm).
    """

    def __init__(self, eps, smoothing_function='fair'):
        super(CPUVectorialTotalVariation, self).__init__()
        self.eps = eps
        self.smoothing_function = smoothing_function

        # Select the correct functions based on smoothing type
        if self.smoothing_function == 'fair':
            self.h_func = h_fair
            self.h_prime_func = h_prime_fair
            self.h_double_prime_func = h_double_prime_fair
        elif self.smoothing_function == 'charbonnier':
            self.h_func = h_charbonnier
            self.h_prime_func = h_prime_charbonnier
            self.h_double_prime_func = h_double_prime_charbonnier
        else:
            # Fallback for pure nuclear norm (no smoothing)
            # h(s)=s, h'(s)=1, h''(s)=0
            self.h_func = lambda s, eps: s
            self.h_prime_func = lambda s, eps: np.ones_like(s)
            self.h_double_prime_func = lambda s, eps: np.zeros_like(s)

    def __call__(self, x):
        return cpu_nuc_norm_sum(x, self.h_func, self.eps)

    def gradient(self, x, out=None):
        if out is None:
            out = np.zeros_like(x)
        cpu_gradient(x, self.h_prime_func, self.eps, out)
        return out

    def hessian_components(self, x):
        """
        Calculates and returns the components needed for the diagonal preconditioner.

        Args:
            x (np.ndarray): The field of Jacobian matrices, shape (nx, ny, nz, M, d).

        Returns:
            tuple: A tuple containing:
                - out_hess_coeffs (np.ndarray): The Hessian coefficients h''(s_k).
                  Shape: (nx, ny, nz, r), where r=min(M,d).
                - out_rank_one_fields (np.ndarray): The basis matrices u_k v_k^T.
                  Shape: (nx, ny, nz, r, M, d).
        """
        # Determine the rank r = min(M,d)
        r = min(x.shape[-2], x.shape[-1])
        
        # Allocate output arrays
        out_hess_coeffs = np.zeros(x.shape[:-2] + (r,), dtype=x.dtype)
        out_rank_one_fields = np.zeros(x.shape[:-2] + (r,) + x.shape[-2:], dtype=x.dtype)
        
        cpu_hessian_components(
            x,
            self.h_double_prime_func,
            self.eps,
            out_hess_coeffs,
            out_rank_one_fields
        )
        return out_hess_coeffs, out_rank_one_fields

    def proximal(self, x, tau, out=None):
        if out is None:
            out = np.zeros_like(x)
        # Note: Proximal operator is for the non-smoothed L1 norm of singular values
        cpu_nuc_norm_proximal(x, tau, out)
        return out
