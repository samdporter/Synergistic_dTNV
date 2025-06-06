#schatten_norm_gpu.py
from cil.optimisation.functions import Function

import torch
from torch import vmap
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pseudo_inverse_torch(H):
    """Inverse except when element is zero."""
    return torch.where(H != 0, 1.0 / H, torch.zeros_like(H))

# function to add Identitu matrix to hermitian matrix
# in order to avoid numerical issues
def add_identity_torch(H, eps=1e-6):
    """
    Adds a small multiple of the identity matrix to the input matrix H.
    This is useful for numerical stability, especially when computing inverses.
    
    Input:
        H: torch.Tensor of shape (n, n) - the input matrix
        eps: float - the small value to multiply with the identity matrix
    Output:
        torch.Tensor of shape (n, n) - the modified matrix
    """
    return H + eps * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eigenvalues_2x2_torch(H):
    """
    Input:  H of shape (2,2), dtype float32 or float64, on `device`.
    Output: tensor of shape (2,) containing the two nonnegative eigenvalues of H.
    (We enforce nonnegativity by clamping any tiny negative round‐off to zero.)
    """
    # H = [[a, b],
    #       [c, d]]
    a = H[0,0]
    b = H[0,1]
    c = H[1,0]
    d = H[1,1]

    # trace/2 and det
    half_trace = (a + d) * 0.5
    det      = a*d - b*c

    # discriminant = (trace/2)^2 − det
    disc = half_trace*half_trace - det
    # clamp any tiny negative to zero
    disc = torch.where(disc < 0.0, torch.zeros_like(disc), disc)
    root = torch.sqrt(disc)

    λ1 = half_trace + root
    λ2 = half_trace - root

    # If λ1 or λ2 is slightly negative due to fp error, clamp to zero
    λ1 = torch.where(λ1 < 0.0, torch.zeros_like(λ1), λ1)
    λ2 = torch.where(λ2 < 0.0, torch.zeros_like(λ2), λ2)
    return torch.stack([λ1, λ2], dim=0)


def eigenvectors_2x2_torch(H, eigenvalues):
    """
    Input:
      H:             a (2,2) symmetric matrix
      eigenvalues:   a length‐2 tensor [λ1, λ2] from eigenvalues_2x2_torch(H)
    Output:
      a (2,2) tensor whose columns are the (unit‐norm) eigenvectors of H.
      Column 0 corresponds to eigenvalue eigenvalues[0], column 1 to eigenvalues[1].
    """

    a = H[0,0]
    b = H[0,1]
    c = H[1,0]
    d = H[1,1]
    λ1, λ2 = eigenvalues[0], eigenvalues[1]

    # We want:
    #   If |b| > 0, pick e = [b, λ - a]^T
    #   else if |c| > 0, pick e = [λ - d, c]^T
    #   else pick e = [1, 0]^T  (or [0,1] for the second eigenvector)
    #
    # Then normalize each.
    #
    # Implement with torch.where to avoid Python control flow:

    b_nonzero = (b.abs() > 1e-9)
    c_nonzero = (c.abs() > 1e-9)

    # First eigenvector (for λ1)
    e1_candidate1 = torch.stack([b,        λ1 - a], dim=0)  # if b_nonzero
    e1_candidate2 = torch.stack([λ1 - d,   c      ], dim=0)  # if (not b_nonzero but c_nonzero)
    e1_default    = torch.tensor([1.0, 0.0], device=device, dtype=H.dtype)

    e1 = torch.where(
        b_nonzero,
        e1_candidate1,
        torch.where(c_nonzero, e1_candidate2, e1_default)
    )
    e1_norm = torch.linalg.norm(e1)
    e1 = e1 / torch.where(e1_norm > 1e-9, e1_norm, torch.tensor(1.0, device=e1.device))


    # Second eigenvector (for λ2)
    e2_candidate1 = torch.stack([b,        λ2 - a], dim=0)
    e2_candidate2 = torch.stack([λ2 - d,   c      ], dim=0)
    e2_default    = torch.tensor([0.0, 1.0], device=device, dtype=H.dtype)

    e2 = torch.where(
        b_nonzero,
        e2_candidate1,
        torch.where(c_nonzero, e2_candidate2, e2_default)
    )
    e2_norm = torch.linalg.norm(e2)
    e2 = e2 / torch.where(e2_norm > 1e-9, e2_norm, torch.tensor(1.0, device=e2.device))


    return torch.stack([e1, e2], dim=1)  # shape (2,2): col 0=e1, col 1=e2


def eigenvalues_3x3_torch(H):
    """
    Input: H of shape (3,3), symmetric.
    Output: (3,) tensor of its eigenvalues, all clamped ≥ 0.
    """
    assert H.shape == (3,3)
    vals = torch.linalg.eigvalsh(H)
    vals = torch.where(vals < 0.0, torch.zeros_like(vals), vals)
    return vals


def eigenvectors_3x3_torch(H, eigenvalues):
    """
    Input:
      H:            a (3,3) symmetric matrix
      eigenvalues:  length‐3 tensor from eigenvalues_3x3_torch(H)
    Output:
      (3,3) tensor whose columns are the corresponding eigenvectors (unit norm).
    """
    _, vecs = torch.linalg.eigh(H)
    return vecs


def l1_norm_torch(x):
    return torch.sum(torch.abs(x))

def l1_norm_prox_torch(x, tau):
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0)

def l2_norm_torch(x):
    return torch.sqrt(torch.sum(x ** 2))

def l2_norm_prox_torch(x, tau):
    n = torch.maximum(l2_norm_torch(x), torch.tensor(1.0, device=x.device))
    factor = torch.clamp(n - tau, min=0.0) / n
    return x * factor

def charbonnier_torch(x, eps):
    return torch.sqrt(x ** 2 + eps ** 2) - eps

def charbonnier_grad_torch(x, eps):
    return x / torch.sqrt(x ** 2 + eps ** 2)

def charbonnier_hessian_diag_torch(x, eps):
    return eps ** 2 / (x ** 2 + eps ** 2) ** 1.5

def charbonnier_inv_hessian_diag_torch(x, eps):
    return (x ** 2 + eps ** 2) ** 1.5 / (eps**2)

def fair_torch(x, eps):
    return eps * (torch.abs(x) / (eps) - torch.log1p(torch.abs(x) / (eps)))

def fair_grad_torch(x, eps):
    return x / (eps + torch.abs(x))

def fair_hessian_diag_torch(x, eps):
    return eps / (eps + torch.abs(x)) ** 2

def fair_inv_hessian_diag_torch(x, eps):
    return (eps + torch.abs(x)) ** 2 / (eps)

def perona_malik_torch(x, eps):
    return (eps / 2) * (1 - torch.exp(-x ** 2 / (eps ** 2)))

def perona_malik_grad_torch(x, eps):
    return x * torch.exp(-x ** 2 / (eps ** 2)) / (eps ** 2)

def perona_malik_hessian_diag_torch(x, eps):
    return (eps ** 2 - 2 * x ** 2) * torch.exp(-x ** 2 / (eps ** 2)) / (eps ** 3)

def perona_malik_inv_hessian_diag_torch(x, eps):
    return (eps ** 3) * torch.exp(x ** 2 / (eps ** 2)) / (eps ** 2 - 2 * x ** 2)

def nothing_torch(x, eps=0):
    return x

def nothing_grad_torch(x, eps=0):
    return torch.ones_like(x)

def nothing_hessian_diag_torch(x, eps=0):
    return torch.zeros_like(x)

def norm_torch(H, func, smoothing_func, order, eps, tail=None):
    if order == 0:
        M = H
        Hsym = M.T @ M
        Hsym = add_identity_torch(Hsym, eps/1e3)
    else:
        M = H
        Hsym = M @ M.T
        Hsym = add_identity_torch(Hsym, eps/1e3)

    if Hsym.shape[-2:] == (2, 2):
        eig = eigenvalues_2x2_torch(Hsym)
    elif Hsym.shape[-2:] == (3, 3):
        eig = eigenvalues_3x3_torch(Hsym)
    else:
        raise ValueError("Only 2×2 or 3×3 blocks supported")

    sigma = torch.sqrt(eig)
    
    # *** VMAP-SAFE TAILING LOGIC ***
    if tail is not None:
        # Sort values and indices to identify the smallest `tail` values
        sorted_sigma, sort_indices = torch.sort(sigma)
        
        # Create a mask that is 1 for the smallest `tail` elements, and 0 otherwise
        mask_sorted = torch.cat([
            torch.ones(tail, device=sigma.device, dtype=sigma.dtype),
            torch.zeros(sigma.shape[0] - tail, device=sigma.device, dtype=sigma.dtype)
        ])
        
        # "Unsort" the mask to align with the original `sigma` tensor
        unsort_indices = torch.argsort(sort_indices)
        mask = mask_sorted[unsort_indices]
    else:
        mask = torch.ones_like(sigma)

    # Apply smoothing only to the masked entries
    s_smoothed = smoothing_func(sigma * mask, eps)
    
    # For non-tailed values, mask is 0, so smoothing_func(0, eps) is applied.
    # We want these to contribute their original value to the norm, so we add them back.
    # The norm of h(s) should be sum(h(s_tailed)) + sum(s_untiled)
    s_to_norm = s_smoothed + sigma * (1 - mask)

    return func(s_to_norm)


def norm_func_torch_xxt(M, func, tau, tail=None):
    H = M @ M.T
    # Add small multiple of identity for numerical stability
    H = add_identity_torch(H, eps=1e-6)
    if H.shape[-2:] == (2, 2):
        S2 = eigenvalues_2x2_torch(H)
        U = eigenvectors_2x2_torch(H, S2)
    elif H.shape[-2:] == (3, 3):
        S2 = eigenvalues_3x3_torch(H)
        U = eigenvectors_3x3_torch(H, S2)
    else:
        raise ValueError(f"Matrix size {H.shape} not supported")

    S = torch.sqrt(S2)
    S_func = func(S, tau) # Calculate processed values for all s

    if tail is not None:
        sorted_S, sort_indices = torch.sort(S)
        mask_sorted = torch.cat([
            torch.ones(tail, device=S.device, dtype=S.dtype),
            torch.zeros(S.shape[0] - tail, device=S.device, dtype=S.dtype)
        ])
        unsort_indices = torch.argsort(sort_indices)
        mask = mask_sorted[unsort_indices]
        
        S_final = S * (1 - mask) + S_func * mask
    else:
        S_final = S_func
    
    # Corrected reconstruction: U @ diag(f(s)/s) @ U.T @ M
    # This is equivalent to U @ diag(f(s)) @ V.T, where V.T = diag(1/s)@U.T@M
    S_inv = pseudo_inverse_torch(S)
    return U @ torch.diag(S_final * S_inv) @ U.T @ M

def norm_func_torch_xtx(M, func, tau, tail=None):
    H = M.T @ M
    # Add small multiple of identity for numerical stability
    H = add_identity_torch(H, eps=1e-6)
    if H.shape[-2:] == (2, 2):
        S2 = eigenvalues_2x2_torch(H)
        V = eigenvectors_2x2_torch(H, S2)
    elif H.shape[-2:] == (3, 3):
        S2 = eigenvalues_3x3_torch(H)
        V = eigenvectors_3x3_torch(H, S2)
    else:
        raise ValueError(f"Matrix size {H.shape} not supported")

    S = torch.sqrt(S2)
    S_func = func(S, tau)

    if tail is not None:
        sorted_S, sort_indices = torch.sort(S)
        mask_sorted = torch.cat([
            torch.ones(tail, device=S.device, dtype=S.dtype),
            torch.zeros(S.shape[0] - tail, device=S.device, dtype=S.dtype)
        ])
        unsort_indices = torch.argsort(sort_indices)
        mask = mask_sorted[unsort_indices]
        
        S_final = S * (1 - mask) + S_func * mask
    else:
        S_final = S_func

    # Corrected reconstruction: M @ V @ diag(f(s)/s) @ V.T
    # This is equivalent to U @ diag(f(s)) @ V.T, where U = M@V@diag(1/s)
    S_inv = pseudo_inverse_torch(S)
    return M @ V @ torch.diag(S_final * S_inv) @ V.T


def norm_func_torch(M, func, tau, order=0, tail=None):
    if order == 0:
        return norm_func_torch_xtx(M, func, tau, tail)
    elif order == 1:
        return norm_func_torch_xxt(M, func, tau, tail)
    else:
        raise ValueError("Invalid order")

def vectorised_norm(A, func, smoothing_func, order=0, eps=0, tail=None):
    def single_block(block):
        return norm_torch(block, func, smoothing_func, order, eps, tail)
    return vmap(vmap(vmap(single_block, in_dims=0), in_dims=0), in_dims=0)(A)

def vectorised_norm_func(A, func, tau, order=0, tail=None):
    def single_block(block):
        return norm_func_torch(block, func, tau, order, tail)
    return vmap(vmap(vmap(single_block, in_dims=0), in_dims=0), in_dims=0)(A)

class GPUVectorialTotalVariation(Function):
    """
    GPU implementation of the vectorial total variation function.
    """
    def __init__(self, eps=None, norm='nuclear',
                 smoothing_function=None, numpy_out=True,
                 tail=None):
        super(GPUVectorialTotalVariation, self).__init__()
        if eps is not None:
            self.eps = torch.tensor(eps, device=device)
        else:
            self.eps = torch.tensor(0.0, device=device)
        self.norm = norm
        self.smoothing_function = smoothing_function
        self.numpy_out = numpy_out
        self.tail = tail

    def direct(self, x):
        order = 1 if x.shape[-2] <= x.shape[-1] else 0
        if self.norm == 'nuclear':
            norm_func = l1_norm_torch
        elif self.norm == 'frobenius':
            norm_func = l2_norm_torch
        else:
            raise ValueError('Norm not defined')
        if self.smoothing_function == 'fair':
            smoothing_func = fair_torch
        elif self.smoothing_function == 'charbonnier':
            smoothing_func = charbonnier_torch
        elif self.smoothing_function == 'perona_malik':
            smoothing_func = perona_malik_torch
        else:
            smoothing_func = nothing_torch
        out = vectorised_norm(x, norm_func, smoothing_func, order, self.eps, self.tail)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)
        val = self.direct(x).sum()
        return val.cpu().numpy() if self.numpy_out else val

    def proximal(self, x, tau):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)
        order = 1 if x.shape[-2] <= x.shape[-1] else 0
        if self.norm == 'nuclear':
            prox_func = l1_norm_prox_torch
        else:
            raise ValueError('Proximal for this norm not defined')
        out = vectorised_norm_func(x, prox_func, tau, order, self.tail)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def gradient(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)
        order = 1 if x.shape[-2] <= x.shape[-1] else 0
        if self.smoothing_function == 'fair':
            grad_func = fair_grad_torch
        elif self.smoothing_function == 'charbonnier':
            grad_func = charbonnier_grad_torch
        elif self.smoothing_function == 'perona_malik':
            grad_func = perona_malik_grad_torch
        else:
            raise ValueError('Smoothing function not defined for gradient')
        out = vectorised_norm_func(x, grad_func, self.eps, order, self.tail)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def hessian_components(self, x):
        """
        Calculates the components needed for the diagonal preconditioner.
        This is the ONLY method that performs a full SVD, as it is
        mathematically required to get both U and V for the Hessian formula.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)

        if self.smoothing_function == 'fair':
            hessian_diag_func = fair_hessian_diag_torch
        elif self.smoothing_function == 'charbonnier':
            hessian_diag_func = charbonnier_hessian_diag_torch
        elif self.smoothing_function == 'perona_malik':
            hessian_diag_func = perona_malik_hessian_diag_torch
        else:
            hessian_diag_func = nothing_hessian_diag_torch

        # --- Step 1: Perform SVD on the entire field of matrices ---
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)

        # --- Step 2: Calculate Hessian coefficients h''(s_k) ---
        hess_coeffs = hessian_diag_func(S, self.eps)

        if self.tail is not None:
            mask = torch.zeros_like(S)
            num_singular_values = S.shape[-1]
            # Select smallest `tail` values. SVD returns descending, so tail is at the end.
            start_index = max(0, num_singular_values - self.tail)
            mask[..., start_index:] = 1.0
            hess_coeffs = hess_coeffs * mask

        # --- Step 3: Construct the field of rank-1 basis matrices u_k v_k^T ---
        U_perm = U.permute(*range(U.ndim - 2), -1, -2)
        U_unsqueezed = U_perm.unsqueeze(-1)
        Vh_unsqueezed = Vh.unsqueeze(-2)
        
        rank_one_fields = U_unsqueezed @ Vh_unsqueezed
        
        return hess_coeffs, rank_one_fields
