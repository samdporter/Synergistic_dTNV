# schatten_norm_gpu_slow.py

from cil.optimisation.functions import Function

import torch
from torch import vmap
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pseudo_inverse_torch(H):
    """Inverse except when element is zero."""
    return torch.where(H != 0, 1.0 / H, torch.zeros_like(H))


def l1_norm_torch(x):
    return torch.sum(torch.abs(x), dim=-1)


def l1_norm_prox_torch(x, eps):
    return torch.sign(x) * torch.clamp(torch.abs(x) - eps, min=0)


def l2_norm_torch(x):
    return torch.sqrt(torch.sum(x ** 2, dim=-1))


def l2_norm_prox_torch(x, eps):
    # Unsqueeze eps for broadcasting
    eps_unsqueezed = eps.unsqueeze(-1)
    
    # Calculate norms along the last dimension
    norms = torch.linalg.norm(x, dim=-1, keepdim=True)
    # Avoid division by zero
    norms = torch.maximum(norms, torch.tensor(1e-9, device=x.device))
    
    # Calculate scaling factor
    factor = torch.clamp(norms - eps_unsqueezed, min=0.0) / norms
    return x * factor


def charbonnier_torch(x, eps):
    return torch.sqrt(x ** 2 + eps ** 2) - eps


def charbonnier_grad_torch(x, eps):
    # Add small epsilon to denominator for stability
    return x / torch.sqrt(x ** 2 + eps ** 2 )


def charbonnier_hessian_diag_torch(x, eps):
    # Returns g''(σ) for Charbonnier: eps²/(σ² + eps²)^(3/2)
    return eps ** 2 / (x ** 2 + eps ** 2 ) ** 1.5


def charbonnier_inv_hessian_diag_torch(x, eps):
    return (x ** 2 + eps ** 2 ) ** 1.5 / (eps ** 2 )


def fair_torch(x, eps):
    return eps * (torch.abs(x) / (eps ) - torch.log1p(torch.abs(x) / (eps )))


def fair_grad_torch(x, eps):
    return x / (eps + torch.abs(x) )


def fair_hessian_diag_torch(x, eps):
    # Returns g''(σ) for Fair: eps/(eps + |σ|)^2
    return eps / (eps + torch.abs(x) ) ** 2


def fair_inv_hessian_diag_torch(x, eps):
    return (eps + torch.abs(x) ) ** 2 / (eps )


def perona_malik_torch(x, eps):
    return (eps / 2) * (1 - torch.exp(-x ** 2 / (eps ** 2 )))


def perona_malik_grad_torch(x, eps):
    return x * torch.exp(-x ** 2 / (eps ** 2 )) / (eps ** 2 )


def perona_malik_hessian_diag_torch(x, eps):
    # Returns g''(σ) for Perona-Malik: (eps² − 2σ²) e^(−σ²/eps²)/eps³
    return (eps ** 2 - 2 * x ** 2) * torch.exp(-x ** 2 / (eps ** 2 )) / (eps ** 3 )


def perona_malik_inv_hessian_diag_torch(x, eps):
    return (eps ** 3 ) * torch.exp(x ** 2 / (eps ** 2 )) / (eps ** 2 - 2 * x ** 2 )


def nothing_torch(x, eps=0):
    return x


def nothing_grad_torch(x, eps=0):
    return torch.ones_like(x)


def nothing_hessian_diag_torch(x, eps=0):
    return torch.zeros_like(x)


class GPUVectorialTotalVariation(Function):
    """
    GPU implementation of the vectorial total variation function.
    """
    def __init__(self, eps=None, norm='nuclear',
                 smoothing_function=None, numpy_out=True,
                 tail=None):

        if eps is not None:
            self.eps = torch.tensor(eps, device=device)
        else:
            self.eps = torch.tensor(0.0, device=device)
        self.norm = norm
        self.smoothing_function = smoothing_function
        self.numpy_out = numpy_out
        self.tail = tail

    def direct(self, x):
        
        # --- Select appropriate functions ---
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

        # --- Efficient Batched Calculation ---
        # 1. Compute singular values for all voxels
        S = torch.linalg.svdvals(x) # S shape: (nx, ny, nz, min(M,d))
        
        # 2. Apply tailing if specified
        if self.tail is not None:
            num_singular_values = S.shape[-1]
            start_index = max(0, num_singular_values - self.tail)
            S_tailed = S[..., start_index:]
        else:
            S_tailed = S

        # 3. Apply smoothing function h(s)
        s_smoothed = smoothing_func(S_tailed, self.eps)

        # 4. Apply norm function (e.g., sum for nuclear norm)
        out = norm_func(s_smoothed)
        
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)

        val = self.direct(x).sum()
        return val.cpu().numpy() if self.numpy_out else val

    def proximal(self, x, eps):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)

        if self.norm == 'nuclear':
            prox_func = l1_norm_prox_torch
        elif self.norm == 'frobenius':
            prox_func = l2_norm_prox_torch
        else:
            raise ValueError('Norm not defined')

        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        
        # Apply proximal operator h_prox(s)
        S_prox_values = prox_func(S, eps)

        # If tailing, only apply prox to the tail values.
        if self.tail is not None:
            mask = torch.zeros_like(S)
            num_singular_values = S.shape[-1]
            start_index = max(0, num_singular_values - self.tail)
            mask[..., start_index:] = 1.0
            
            # Combine original head with processed tail
            S_final = S * (1 - mask) + S_prox_values * mask
        else:
            S_final = S_prox_values
        
        out = torch.matmul(U, torch.matmul(torch.diag_embed(S_final), Vh))

        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def gradient(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)

        if self.smoothing_function == 'fair':
            grad_func = fair_grad_torch
        elif self.smoothing_function == 'charbonnier':
            grad_func = charbonnier_grad_torch
        elif self.smoothing_function == 'perona_malik':
            grad_func = perona_malik_grad_torch
        else: # Default to non-smoothed
            # The gradient of h(s)=s is h'(s)=1
            grad_func = nothing_grad_torch
        
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)

        # Apply gradient function h'(s)
        S_grad_values = grad_func(S, self.eps)

        # If tailing, the gradient for non-tailed values is 0
        if self.tail is not None:
            mask = torch.zeros_like(S)
            num_singular_values = S.shape[-1]
            start_index = max(0, num_singular_values - self.tail)
            mask[..., start_index:] = 1.0
            S_grad_values = S_grad_values * mask
        
        # Reconstruct the gradient matrix: U diag(h'(s)) V^T
        out = torch.matmul(U, torch.matmul(torch.diag_embed(S_grad_values), Vh))
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def hessian_components(self, x):
        """
        Calculates the components needed for the diagonal preconditioner.
        Returns the building blocks for the calling class to use.

        Returns:
            tuple: A tuple containing:
                - hess_coeffs (torch.Tensor): h''(s_k), shape (..., num_singular_values).
                - rank_one_fields (torch.Tensor): u_k v_k^T, shape (..., num_singular_values, M, d).
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
        # U shape: (..., M, r), S shape: (..., r), Vh shape: (..., r, d)

        # --- Step 2: Calculate Hessian coefficients h''(s_k) ---
        hess_coeffs = hessian_diag_func(S, self.eps) # Shape: (..., r)

        if self.tail is not None:
            mask = torch.zeros_like(S)
            num_singular_values = S.shape[-1]
            start_index = max(0, num_singular_values - self.tail)
            mask[..., start_index:] = 1.0
            hess_coeffs = hess_coeffs * mask

        # --- Step 3: Construct the field of rank-1 basis matrices u_k v_k^T ---
        # Target shape: (..., r, M, d)
        
        # U shape is (..., M, r). We need k to be an outer dimension.
        U_perm = U.permute(*range(U.ndim - 2), -1, -2) # Swap last two dims -> (..., r, M)
        
        # Unsqueeze to prepare for batched matrix multiplication (outer product)
        # U_perm becomes (..., r, M, 1)
        # Vh becomes     (..., r, 1, d)
        U_unsqueezed = U_perm.unsqueeze(-1)
        Vh_unsqueezed = Vh.unsqueeze(-2)

        # This performs a batch of r outer products for each voxel
        rank_one_fields = U_unsqueezed @ Vh_unsqueezed # Shape: (..., r, M, d)
        
        return hess_coeffs, rank_one_fields
