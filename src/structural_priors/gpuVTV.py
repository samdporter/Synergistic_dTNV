import torch
from torch import vmap
import numpy as np
from numbers import Number

from .Function import Function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pseudo_inverse_torch(H):
    """Inverse except when element is zero."""
    return torch.where(H != 0, 1/H, torch.tensor(0.0, device=device)) 

def eigenvalues_2x2_torch(H):
    assert H.shape == (2, 2)
    a, b, c, d = H[..., 0, 0], H[..., 0, 1], H[..., 1, 0], H[..., 1, 1]
    av_trace = (a + d)/2
    determinant = a * d - b * c
    
    discriminant = av_trace**2 - determinant
    discriminant = torch.where(discriminant < 0, 0, discriminant)
    
    eigenvalue1 = av_trace + torch.sqrt(discriminant)
    eigenvalue2 = av_trace - torch.sqrt(discriminant)
    
    # Set very small eigenvalues to zero
    eigenvalue1 = torch.where(eigenvalue1 < 0, 0, eigenvalue1)
    eigenvalue2 = torch.where(eigenvalue2 < 0, 0, eigenvalue2)
    
    return torch.stack([eigenvalue1, eigenvalue2], dim=-1)

def eigenvectors_2x2_torch(H, eigenvalues):
    # Extract elements from the matrix
    a = H[0, 0]
    b = H[0, 1]
    c = H[1, 0]
    d = H[1, 1]

    # Handle the case where b or c (off-diagonal elements) might be zero
    b_nonzero_condition = (b.abs() > 0)
    c_nonzero_condition = (c.abs() > 0)

    # Compute the first eigenvector
    e1 = torch.where(
        b_nonzero_condition,
        torch.stack([b, eigenvalues[0] - a]),
        torch.where(
            c_nonzero_condition,
            torch.stack([eigenvalues[0] - d, c]),
            torch.tensor([1, 0], dtype=torch.float32, device=device)
        )
    )

    # Compute the second eigenvector
    e2 = torch.where(
        b_nonzero_condition,
        torch.stack([b, eigenvalues[1] - a]),
        torch.where(
            c_nonzero_condition,
            torch.stack([eigenvalues[1] - d, c]),
            torch.tensor([0, 1], dtype=torch.float32, device=device)
        )
    )

    # Normalize the eigenvectors
    e1norm = torch.norm(e1)
    e2norm = torch.norm(e2)

    return torch.stack([e1 / e1norm, e2 / e2norm], dim=-1)

def cardano_cubic_roots_torch(a, b, c, d):
    # Coefficients for the depressed cubic equation
    p = (3 * a * c - b ** 2) / (3 * a ** 2)
    q = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)

    # Discriminant of the cubic equation
    discriminant = (q / 2) ** 2 + (p / 3) ** 3
    
    discriminant = discriminant
    discriminant = torch.where(discriminant < 0, torch.zeros_like(discriminant), discriminant)

    r = torch.sqrt(torch.abs(-(p / 3) ** 3))
    theta = torch.acos(torch.clamp(-q / (2 * r), -1.0, 1.0))  # Clamp the argument of acos to avoid NaN

    # Calculate the roots using Cardano's formula
    root1 = 2 * torch.pow(r, 1/3) * torch.cos(theta / 3) - b / (3 * a)
    root2 = 2 * torch.pow(r, 1/3) * torch.cos((theta + 2 * np.pi) / 3) - b / (3 * a)
    root3 = 2 * torch.pow(r, 1/3) * torch.cos((theta + 4 * np.pi) / 3) - b / (3 * a)

    roots = torch.stack([root1, root2, root3], dim=-1)
    return torch.sort(roots).values

def eigenvalues_3x3_torch(H):
    assert H.shape == (3, 3)
    a = torch.tensor(1.0, dtype=H.dtype, device=H.device)
    b = -torch.trace(H)
    c = 0.5 * (torch.trace(H)**2 - torch.trace(H @ H))
    d = -torch.det(H)
    eigenvalues = cardano_cubic_roots_torch(a, b, c, d)
    # Replace NaNs with zeros
    #eigenvalues = torch.where(torch.isnan(eigenvalues), torch.zeros_like(eigenvalues), eigenvalues)
    # if very small eigenvalues are smaller than 0, set them to zero
    eigenvalues = torch.where(eigenvalues < 0, torch.zeros_like(eigenvalues), eigenvalues)
    return eigenvalues

def compute_special_eigenvector(mu):
    v_i = torch.stack([torch.ones_like(mu), -mu, torch.zeros_like(mu)], dim=-1)
    norm = torch.sqrt(1 + mu ** 2)
    v_i = v_i / norm.unsqueeze(-1)
    return v_i

def eigenvectors_3x3_torch(A, eigenvalues):
    I = torch.eye(3, dtype=A.dtype, device=A.device)
    eigenvectors = []
    default_eigenvectors = torch.eye(3, dtype=A.dtype, device=A.device)

    for i, eigenvalue in enumerate(eigenvalues):
        M = A - eigenvalue * I

        # Compute the norms of the columns of M
        norms = torch.norm(M, dim=0)
        min_norm = torch.min(norms)

        # Mask for the minimum norm column
        min_norm_mask = norms == min_norm

        # Special eigenvector calculation
        e1 = M[:, 0]
        e2 = M[:, 1]

        mu = (e2[0] / (e2[1] + torch.finfo(e2.dtype).eps))  # Avoid division by zero
        special_eigenvector = compute_special_eigenvector(mu)

        # Compute the cross product of the first two columns of M
        cross_product = torch.cross(e1, e2)
        cross_norm = torch.norm(cross_product)

        # Check if e1 and e2 are multiples of each other
        multiples_condition = cross_norm < 1e-6

        # Use torch.where to select the correct eigenvector
        eigenvector = torch.where(multiples_condition.unsqueeze(0), special_eigenvector.unsqueeze(0), cross_product.unsqueeze(0) / (cross_norm + 1e-8)).squeeze(0)

        # Default eigenvector corresponding to the column with the minimum norm
        default_eigenvector = torch.sum(default_eigenvectors * min_norm_mask.unsqueeze(0), dim=1)

        # Handle the case where all values are zero
        min_norm_zero_mask = min_norm < 1e-6
        eigenvector = torch.where(min_norm_zero_mask.unsqueeze(-1), default_eigenvector, eigenvector)

        eigenvectors.append(eigenvector)

    return torch.stack(eigenvectors, dim=-1)

def l1_norm_torch(x):
    return torch.sum(torch.abs(x))

def l1_norm_prox_torch(x, tau):
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0)

def l2_norm_torch(x):
    return torch.sqrt(torch.sum(x**2))

def l2_norm_prox_torch(x, tau):
    return x / torch.maximum(1, l2_norm_torch(x) / tau)

def charbonnier_torch(x, eps):
    return torch.sqrt(x**2 + eps**2) - eps

def charbonnier_grad_torch(x, eps):
    return x / torch.sqrt(x**2 + eps**2)

def charbonnier_hessian_diag_torch(x, eps):
    return eps**2 / (x**2 + eps**2)**(3/2)

def charbinier_inv_hessian_diag_torch(x, eps):
    return (x**2 + eps**2)**(3/2) / eps**2

def fair_torch(x, eps):
    return eps * (torch.abs(x) / eps - torch.log(1 + torch.abs(x) / eps))

def fair_grad_torch(x, eps):
    return x / (eps + torch.abs(x))

def fair_hessian_diag_torch(x, eps):
    return eps / (eps + torch.abs(x))**2

def fair_inv_hessian_diag_torch(x, eps):
    return (eps + torch.abs(x))**2 / eps

def perona_malik_torch(x, eps):
    return eps/2 * (1 - torch.exp(-x**2 / eps**2))

def perona_malik_grad_torch(x, eps):
    return x * torch.exp(-x**2 / eps**2) / eps**2

def perona_malik_hessian_diag_torch(x, eps):
    return (eps**2 - 2*x**2) * torch.exp(- x**2 / eps**2) / eps**3

def perona_malik_inv_hessian_diag_torch(x, eps):
    return eps**3 * torch.exp(x**2 / eps**2) / (eps**2 - 2*x**2)

def nothing_torch(x, eps=0):
    return x

def nothing_grad_torch(x, eps=0):
    return torch.ones_like(x)

def norm_torch(M, func, smoothing_func, order, eps):

    if order == 0:
        H = M.T @ M
    elif order == 1:
        H = M @ M.T
    else:
        raise ValueError("Invalid order")
    if H.shape == (2, 2):
        eigenvalues = eigenvalues_2x2_torch(H)
    elif H.shape == (3, 3):
        raise ValueError("3x3 matrix not working as intended")
        eigenvalues = eigenvalues_3x3_torch(H)[1:]

    singularvalues = torch.sqrt(eigenvalues)
    singularvalues = smoothing_func(singularvalues, eps)
    return func(singularvalues)

def norm_func_torch_xxt(X, func, tau):

    H = X@X.T

    if H.shape == (2,2):
        S_square = eigenvalues_2x2_torch(H)
        U = eigenvectors_2x2_torch(H, S_square)
    elif H.shape == (3,3):
        raise ValueError("3x3 matrix not working as intended")
        S_square = eigenvalues_3x3_torch(H)
        U = eigenvectors_3x3_torch(H, S_square)
    else:
        raise ValueError(f"Matrix size {H.shape} not supported")

    S = torch.sqrt(S_square)
    S_inv = pseudo_inverse_torch(S)
    S_func = func(S, tau)

    # Reconstruct the result matrix
    return U @ torch.diag(S_func) @ torch.diag(S_inv) @ U.T @ X

def norm_func_torch_xtx(X, func, tau):

    H = X.T@X

    if H.shape == (2,2):
        S_square = eigenvalues_2x2_torch(H)
        V = eigenvectors_2x2_torch(H, S_square)
    elif H.shape == (3,3):
        raise ValueError("3x3 matrix not working as intended")
        S_square = eigenvalues_3x3_torch(H)
        V = eigenvectors_3x3_torch(H, S_square)
    else:
        raise ValueError(f"Matrix size {H.shape} not supported")

    S = torch.sqrt(S_square)
    S_inv = pseudo_inverse_torch(S)
    S_func = func(S, tau)

    # Reconstruct the result matrix
    return X @ V @ torch.diag(S_inv) @ torch.diag(S_func)  @ V.T

def norm_func_torch(X, func, tau, order=0):

    if order == 0:
        return norm_func_torch_xtx(X, func, tau)
    elif order == 1:
        return norm_func_torch_xxt(X, func, tau)
    else:
        raise ValueError("Invalid order")

def vectorised_norm(A, func, smoothing_func, order=0, eps=0):
    def ordered_norm(A):
        return norm_torch(A, func, smoothing_func, order, eps)

    # Apply vmap across the last two dimensions
    return vmap(vmap(vmap(ordered_norm, in_dims=0), in_dims=0), in_dims=0)(A)

def vectorised_norm_func(A, func, tau, order=0):

    def norm_prox_element(M):
        return norm_func_torch(M, func, tau, order)

    return vmap(vmap(vmap(norm_prox_element, in_dims=0), in_dims=0), in_dims=0)(A)

class GPUVectorialTotalVariation(Function):
    """ 
    GPU implementation of the vectorial total variation function.
    """
    def __init__(self, eps=None, norm = 'nuclear',
                 smoothing_function=None, numpy_out=False):        

        """Initializes the GPUVectorialTotalVariation class.
        """    
        if eps is not None:    
            self.eps = torch.tensor(eps)
        else:
            self.eps = torch.tensor(0)
        self.norm = norm
        self.smoothing_function = smoothing_function
        self.numpy_out = numpy_out

    def direct(self, x):

        # order defined by smallest of the last two dimensions
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

        return vectorised_norm(x, norm_func, smoothing_func, order, self.eps)

    def __call__(self, x):
        
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device)
        else:
            x.to(device)

        return torch.sum(self.direct(x)).cpu().numpy() if self.numpy_out else torch.sum(self.direct(x))

    def proximal(self, x, tau):

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device)
        else:
            x.to(device)

        # order defined by smallest of the last two dimensions
        order =1 if x.shape[-2] <= x.shape[-1] else 0

        if self.norm == 'nuclear':
            norm_func = l1_norm_prox_torch
        elif self.norm == 'frobenius':
            norm_func = l2_norm_prox_torch
        else:
            raise ValueError('Norm not defined')

        return vectorised_norm_func(x, norm_func, tau, order)
    
    def gradient(self, x):

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, device=device)
        else:
            x.to(device)

        # order defined by smallest of the last two dimensions
        order =1 if x.shape[-2] <= x.shape[-1] else 0
        
        if self.smoothing_function == 'fair':
            smoothing_func = fair_grad_torch
        elif self.smoothing_function == 'charbonnier':
            smoothing_func = charbonnier_grad_torch
        elif self.smoothing_function == 'perona_malik':
            smoothing_func = perona_malik_grad_torch
        else:
            raise ValueError('Smoothing function not defined')

        return vectorised_norm_func(x, smoothing_func, self.eps, order)

    def hessian_diag(self, x):

        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
            
        x.to(device)

        # order defined by smallest of the last two dimensions
        order =1 if x.shape[-2] <= x.shape[-1] else 0
        
        if self.smoothing_function == 'fair':
            smoothing_func = fair_hessian_diag_torch
        elif self.smoothing_function == 'charbonnier':
            smoothing_func = charbonnier_hessian_diag_torch
        elif self.smoothing_function == 'perona_malik':
            smoothing_func = perona_malik_hessian_diag_torch
        else:
            raise ValueError('Smoothing function not defined')

        return vectorised_norm_func(x, smoothing_func, self.eps, order)
    
    def inv_hessian_diag(self, x):

        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
            
        x.to(device)

        # order defined by smallest of the last two dimensions
        order =1 if x.shape[-2] <= x.shape[-1] else 0
        
        if self.smoothing_function == 'fair':
            smoothing_func = fair_inv_hessian_diag_torch
        elif self.smoothing_function == 'charbonnier':
            smoothing_func = charbinier_inv_hessian_diag_torch
        elif self.smoothing_function == 'perona_malik':
            smoothing_func = perona_malik_inv_hessian_diag_torch
        else:
            raise ValueError('Smoothing function not defined')

        return vectorised_norm_func(x, smoothing_func, self.eps, order)