from cil.optimisation.functions import Function

import torch
from torch import vmap
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pseudo_inverse_torch(H):
    """Inverse except when element is zero."""
    return torch.where(H != 0, 1.0 / H, torch.zeros_like(H))

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

    b_nonzero = (b.abs() > 0.0)
    c_nonzero = (c.abs() > 0.0)

    # First eigenvector (for λ1)
    e1_candidate1 = torch.stack([b,        λ1 - a], dim=0)  # if b_nonzero
    e1_candidate2 = torch.stack([λ1 - d,   c      ], dim=0)  # if (not b_nonzero but c_nonzero)
    e1_default    = torch.tensor([1.0, 0.0], device=device, dtype=H.dtype)

    e1 = torch.where(
        b_nonzero,
        e1_candidate1,
        torch.where(c_nonzero, e1_candidate2, e1_default)
    )
    e1 = e1 / e1.norm()

    # Second eigenvector (for λ2)
    e2_candidate1 = torch.stack([b,        λ2 - a], dim=0)
    e2_candidate2 = torch.stack([λ2 - d,   c      ], dim=0)
    e2_default    = torch.tensor([0.0, 1.0], device=device, dtype=H.dtype)

    e2 = torch.where(
        b_nonzero,
        e2_candidate1,
        torch.where(c_nonzero, e2_candidate2, e2_default)
    )
    e2 = e2 / e2.norm()

    return torch.stack([e1, e2], dim=1)  # shape (2,2): col 0=e1, col 1=e2



def eigenvalues_3x3_torch(H):
    """
    Input: H of shape (3,3), symmetric.
    Output: (3,) tensor of its eigenvalues, all clamped ≥ 0.
    We simply call torch.linalg.eigvalsh and clamp to zero.
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
    We simply let torch.linalg.eigh produce the full set, which is 
    guaranteed continuous (and torch.linalg.eigh is already vectorizable).
    """
    # torch.linalg.eigh returns (eigenvalues, eigenvectors), but
    # since we already computed eigenvalues above (and clamped),
    # it suffices to call torch.linalg.eigh(H) again:
    _, vecs = torch.linalg.eigh(H)
    # ensure the same sign convention?  Typically not needed; eigh already picks a basis.
    return vecs  # shape (3,3): each column is an eigenvector


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
    return eps ** 2 / (x ** 2 + eps ** 2) ** (3 / 2)

def charbonnier_inv_hessian_diag_torch(x, eps):
    return (x ** 2 + eps ** 2) ** (3 / 2) / eps ** 2

def fair_torch(x, eps):
    return eps * (torch.abs(x) / eps - torch.log1p(torch.abs(x) / eps))

def fair_grad_torch(x, eps):
    return x / (eps + torch.abs(x))

def fair_hessian_diag_torch(x, eps):
    return eps / (eps + torch.abs(x)) ** 2

def fair_inv_hessian_diag_torch(x, eps):
    return (eps + torch.abs(x)) ** 2 / eps

def perona_malik_torch(x, eps):
    return (eps / 2) * (1 - torch.exp(-x ** 2 / eps ** 2))

def perona_malik_grad_torch(x, eps):
    return x * torch.exp(-x ** 2 / eps ** 2) / (eps ** 2)

def perona_malik_hessian_diag_torch(x, eps):
    return (eps ** 2 - 2 * x ** 2) * torch.exp(-x ** 2 / eps ** 2) / (eps ** 3)

def perona_malik_inv_hessian_diag_torch(x, eps):
    return eps ** 3 * torch.exp(x ** 2 / eps ** 2) / (eps ** 2 - 2 * x ** 2)

def nothing_torch(x, eps=0):
    return x

def nothing_grad_torch(x, eps=0):
    return torch.ones_like(x)

def norm_torch(H, func, smoothing_func, order, eps, tail=None):
    """
    Compute one block's contribution to the (smoothed) Schatten‐p norm
    and apply 'func' (which might be the l1‐norm or a gradient, etc.)
    H is either (2×2) or (3×3) symmetric.  'order' tells us whether
    to form H = M^T M (order=0) or M M^T (order=1).  We return a vector
    of length = min(dim), containing g(σ_i) or g'(σ_i), etc.
    """
    if order == 0:
        M = H
        Hsym = M.T @ M
    else:
        M = H
        Hsym = M @ M.T

    # Eigenvalues of Hsym
    if Hsym.shape[-2:] == (2, 2):
        eig = eigenvalues_2x2_torch(Hsym)
    elif Hsym.shape[-2:] == (3, 3):
        eig_full = eigenvalues_3x3_torch(Hsym)  # length 3
        # Only keep the top 2 if M is 3×2 or 2×3, etc.  But since order=1 or 0
        # we typically have rank ≤2.
        # We assume eig_full is nonnegative and sorted ascending.  We take the top r.
        eig = eig_full[-2:]
    else:
        raise ValueError("Only 2×2 or 3×3 blocks supported")

    sigma = torch.sqrt(eig)         # singular values
    if tail is not None:
        # Mask out all but the smallest 'tail' singular values
        sorted_sigma, idx = torch.sort(sigma)
        mask = torch.zeros_like(sigma)
        mask[idx[:tail]] = 1.0
    else:
        mask = torch.ones_like(sigma)

    # Apply smoothing only on the masked entries
    s = sigma * mask
    sm = smoothing_func(s, eps)     # shape = (2,) or (3,)
    # Then apply 'func' (could be sum(abs(.)) or the gradient or Hessian‐diag)
    return func(sm)

def norm_func_torch_xxt(M, func, tau, tail=None):
    H = M @ M.T
    if H.shape[-2:] == (2, 2):
        S2 = eigenvalues_2x2_torch(H)
        U = eigenvectors_2x2_torch(H, S2)
    elif H.shape[-2:] == (3, 3):
        S2 = eigenvalues_3x3_torch(H)
        U = eigenvectors_3x3_torch(H, S2)
    else:
        raise ValueError(f"Matrix size {H.shape} not supported")

    S = torch.sqrt(S2)                      # singular values
    if tail is not None:
        # keep only the tail smallest singular values
        S_sorted, idx = torch.sort(S)
        mask = torch.zeros_like(S)
        mask[idx[:tail]] = 1.0
    else:
        mask = torch.ones_like(S)

    S_inv = pseudo_inverse_torch(S)
    S_func = func(S * mask, tau)

    # Reconstruct prox: U diag(S_func) diag(S_inv) U^T M
    return U @ torch.diag(S_func) @ torch.diag(S_inv) @ U.T @ M

def norm_func_torch_xtx(M, func, tau, tail=None):
    H = M.T @ M
    if H.shape[-2:] == (2, 2):
        S2 = eigenvalues_2x2_torch(H)
        V = eigenvectors_2x2_torch(H, S2)
    elif H.shape[-2:] == (3, 3):
        S2 = eigenvalues_3x3_torch(H)
        V = eigenvectors_3x3_torch(H, S2)
    else:
        raise ValueError(f"Matrix size {H.shape} not supported")

    S = torch.sqrt(S2)
    if tail is not None:
        S_sorted, idx = torch.sort(S)
        mask = torch.zeros_like(S)
        mask[idx[:tail]] = 1.0
    else:
        mask = torch.ones_like(S)

    S_inv = pseudo_inverse_torch(S)
    S_func = func(S * mask, tau)

    return M @ V @ torch.diag(S_inv) @ torch.diag(S_func) @ V.T

def norm_func_torch(M, func, tau, order=0, tail=None):
    if order == 0:
        return norm_func_torch_xtx(M, func, tau, tail)
    elif order == 1:
        return norm_func_torch_xxt(M, func, tau, tail)
    else:
        raise ValueError("Invalid order")

def vectorised_norm(A, func, smoothing_func, order=0, eps=0, tail=None):
    """
    Apply norm_torch(A[i,j,k,...], func, smoothing_func, order, eps, tail)
    across the batch dimensions using vmap.
    The ellipses "..." stand for the last two dims of A, which are the M×d block.
    """
    def single_block(block):
        return norm_torch(block, func, smoothing_func, order, eps, tail)
    # We assume A has shape (nx,ny,nz,M,d).  We want to map over nx, ny, nz.
    return vmap(vmap(vmap(single_block, in_dims=0), in_dims=0), in_dims=0)(A)

def vectorised_norm_func(A, func, tau, order=0, tail=None):
    """
    Similar vmap wrapper, but func is a proximal or gradient applied to singular values.
    """
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

        if eps is not None:
            self.eps = torch.tensor(eps, device=device)
        else:
            self.eps = torch.tensor(0.0, device=device)
        self.norm = norm
        self.smoothing_function = smoothing_function
        self.numpy_out = numpy_out
        self.tail = tail

    def direct(self, x):
        # x shape: (nx,ny,nz,M,d)
        # Decide order based on block shape
        order = 1 if x.shape[-2] <= x.shape[-1] else 0

        # Choose which “func” to apply to singular values
        if self.norm == 'nuclear':
            norm_func = l1_norm_torch
        elif self.norm == 'frobenius':
            norm_func = l2_norm_torch
        else:
            raise ValueError('Norm not defined')

        # Choose smoothing function
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
        # x may be numpy or torch
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)

        val = self.direct(x).sum()  # sum over all voxels
        return val.cpu().numpy() if self.numpy_out else val

    def proximal(self, x, tau):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)

        order = 1 if x.shape[-2] <= x.shape[-1] else 0

        if self.norm == 'nuclear':
            prox_func = l1_norm_prox_torch
        elif self.norm == 'frobenius':
            prox_func = l2_norm_prox_torch
        else:
            raise ValueError('Norm not defined')

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
            raise ValueError('Smoothing function not defined')

        out = vectorised_norm_func(x, grad_func, self.eps, order, self.tail)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def hessian_diag(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)

        order = 1 if x.shape[-2] <= x.shape[-1] else 0

        if self.smoothing_function == 'fair':
            hess_func = fair_hessian_diag_torch
        elif self.smoothing_function == 'charbonnier':
            hess_func = charbonnier_hessian_diag_torch
        elif self.smoothing_function == 'perona_malik':
            hess_func = perona_malik_hessian_diag_torch
        else:
            raise ValueError('Smoothing function not defined')

        out = vectorised_norm_func(x, hess_func, self.eps, order, self.tail)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def inv_hessian_diag(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        else:
            x = x.to(device, dtype=torch.float32)

        order = 1 if x.shape[-2] <= x.shape[-1] else 0

        if self.smoothing_function == 'fair':
            inv_hess_func = fair_inv_hessian_diag_torch
        elif self.smoothing_function == 'charbonnier':
            inv_hess_func = charbonnier_inv_hessian_diag_torch
        elif self.smoothing_function == 'perona_malik':
            inv_hess_func = perona_malik_inv_hessian_diag_torch
        else:
            raise ValueError('Smoothing function not defined')

        out = vectorised_norm_func(x, inv_hess_func, self.eps, order, self.tail)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
