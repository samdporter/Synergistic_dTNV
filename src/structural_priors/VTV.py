from cil.optimisation.functions import Function
try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    device = 'cpu'
import numpy as np

from sirf.STIR import ImageData


from .schatten_norm_cpu import CPUVectorialTotalVariation
from.Gradients import Jacobian

class BlockDataContainerToArray:
    def __init__(self, domain_geometry, gpu=True):
        self.domain_geometry = domain_geometry
        self.gpu = gpu

    def direct(self, x, out=None):
        # Ensure input has the correct attribute
        if not hasattr(x, "containers"):
            raise ValueError("Input x must be a block data container with a 'containers' attribute.")
        
        # Convert containers to a single stacked array (GPU or CPU)
        arrays = (d.as_array() for d in x.containers)  # Generator to reduce memory overhead
        if self.gpu:
            ret = torch.stack([torch.tensor(arr, device=device) for arr in arrays], dim=-1)
        else:
            ret = np.stack(list(arrays), axis=-1)  # Convert generator to list for numpy stack

        # Write to output if provided
        if out is not None:
            out.fill(ret)
        return ret

    def adjoint(self, x, out=None):
        # Convert input to a NumPy array if needed
        if self.gpu and isinstance(x, torch.Tensor):
            x_arr = x.cpu().numpy()
        else:
            x_arr = np.asarray(x)

        # Fill the domain geometry with the corresponding slices
        res = self.domain_geometry.clone()
        for i, r in enumerate(res.containers):
            r.fill(x_arr[..., i])

        # Write to output if provided
        if out is not None:
            out.fill(res)
        return res

class WeightedVectorialTotalVariation(Function):
    """
    Basically an OperatorCompositionFunction but without requiring as_arrays et al.
    so that we can keep everything on the GPU
    """

    def __init__(self, geometry, weights, delta, smoothing='fair', norm = 'nuclear', 
                 gpu=True, anatomical=None, stable = True, tail_singular_values=None,
                 hessian='diagonal'):
        """
        Initializes the WeightedVectorialTotalVariation class.
        """        

        voxel_sizes = geometry.containers[0].voxel_sizes()
        if isinstance(anatomical, ImageData):
            anatomical = anatomical.as_array()
        self.jacobian = Jacobian(
            voxel_sizes, anatomical=anatomical, 
            gpu=gpu, numpy_out=not gpu,
            method="central",)
        self.smoothing = smoothing
        self.hessian = hessian

        self.bdc2a = BlockDataContainerToArray(geometry, gpu=gpu)
        self.weights = self.bdc2a.direct(weights)
        self.gpu = gpu
            
        if gpu:
            self.inv_weights = torch.reciprocal(self.weights)
            self.inv_weights = torch.nan_to_num(self.inv_weights, nan=0.0, neginf=0.0, posinf=0.0)
            if tail_singular_values is not None:
                print(f"tail_singular_values = {tail_singular_values}")
            if stable:
                from .schatten_norm_gpu_slow import GPUVectorialTotalVariation
                self.vtv = GPUVectorialTotalVariation(eps=delta, norm=norm, smoothing_function=smoothing, tail=tail_singular_values)
            else:
                from .schatten_norm_gpu import GPUVectorialTotalVariation
                self.vtv = GPUVectorialTotalVariation(eps=delta, norm=norm, smoothing_function=smoothing, tail=tail_singular_values)
        else:
            self.inv_weights = np.reciprocal(self.weights, where=self.weights != 0)
            if tail_singular_values is not None:
                raise ValueError("tail_singular_values is only implemented for GPU")
            self.vtv = CPUVectorialTotalVariation(delta, smoothing_function=smoothing)

    def __call__(self, x):
        weights = self.weights.unsqueeze(-1)  # ensure proper broadcasting
        return self.vtv(
            weights * self.jacobian.direct(
                self.bdc2a.direct(x)
                )
            )

    
    def call_no_sum(self, x):
        # expand weghts by one extra axis on end
        weights = self.weights.unsqueeze(-1)
        return self.bdc2a.adjoint(
            self.vtv.call_no_sum(
                weights*self.jacobian.direct(
                    self.bdc2a.direct(x)
                    )
                )
            )
    
    
    def gradient(self, x, out=None):
        # expand weghts by one extra axis on end
        weights = self.weights.unsqueeze(-1)
        ret =  self.bdc2a.adjoint(
            self.jacobian.adjoint(
                weights*self.vtv.gradient(
                    weights*self.jacobian.direct(
                        self.bdc2a.direct(x)
                        )
                    )
                )
            )
        if out is not None:
            out.fill(ret)
            return out
        else:
            return ret
        
    def _scalar_upper_hessian_diag_arr(self, x):
        
        # Expand weights for proper broadcasting.
        if hasattr(self.weights, "unsqueeze"):
            weights = self.weights.unsqueeze(-1)
        else:
            weights = np.expand_dims(self.weights, axis=-1)
        
        x_arr = self.bdc2a.direct(x)
        # Compute u = weights * jacobian.direct(bdc2a.direct(x))
        u = weights * self.jacobian.direct(x_arr)
        # Compute the diagonal Hessian approximation for vtv evaluated at u.
        vtv_diag = self.vtv.hessian_diag(u)
        # Backproject the diagonal using the adjoints.
        return self.jacobian.adjoint(vtv_diag).abs()
    
    # --- WeightedVectorialTotalVariation: Hessian diagonal -------------
    def _hess_diag_core(self, x_arr):
        sens   = self.jacobian.sensitivity(x_arr)          # (..., m, 3)
        s2     = sens.pow(2) if self.gpu else sens**2

        w      = self.weights.unsqueeze(-1)                # (..., m, 1)
        u      = w * self.jacobian.direct(x_arr)
        d2phi  = self.vtv.hessian_diag(u)                  # (..., m, 3)

        diag   = (d2phi * s2).sum(dim=-1) if self.gpu \
                else (d2phi * s2).sum(axis=-1)
        return diag.abs()
    
    def _scalar_upper_hessian_diag_arr(self, x):
        # global Lipschitz constant of J
        L = self.jacobian.calculate_norm() ** 2      # scalar
        w = self.weights.unsqueeze(-1)
        x_arr = self.bdc2a.direct(x)
        u = w * self.jacobian.direct(x_arr)           # (..., m, 3)
        vtv_diag = self.vtv.hessian_diag(u).sum(dim=-1 if self.gpu else -1)  # (..., m)
        return (L * vtv_diag).abs()

    def hessian_diag_arr(self, x):
        if self.hessian == 'scalar':
            return self._scalar_upper_hessian_diag_arr(x)
        # default: diagonal
        x_arr = self.bdc2a.direct(x)
        return self._hess_diag_core(x_arr)

    def hessian_diag(self, x, out=None):
        """
        Block‑data‑container version of the Hessian diagonal.
        Result lives in the same geometry as `x`.
        """
        diag_arr = self.hessian_diag_arr(x)
        # multiply by w² before mapping back
        diag_arr = (self.weights ** 2) * diag_arr
        result   = self.bdc2a.adjoint(diag_arr)
        if out is not None:
            out.fill(result)
            return out
        return result

    def inv_hessian_diag(self, x, out=None, epsilon=0.0):
        """
        Inverse of the diagonal Hessian approximation.
        """
        diag_arr = self.hessian_diag_arr(x) + epsilon
        if self.gpu:
            inv_arr = torch.reciprocal(diag_arr)
            torch.nan_to_num(inv_arr, nan=0.0, posinf=0.0, neginf=0.0,
                            out=inv_arr)
        else:
            inv_arr = np.reciprocal(diag_arr, where=diag_arr != 0)

        inv_arr = (self.inv_weights ** 2) * inv_arr
        result  = self.bdc2a.adjoint(inv_arr)
        if out is not None:
            out.fill(result)
            return out
        return result
        
    def proximal(self, x, tau, out=None):
        # expand weghts by one extra axis on end
        weights = self.weights.unsqueeze(-1)
        ret = self.bdc2a.adjoint(
            self.jacobian.adjoint(
                weights*self.vtv.proximal(
                    weights*self.jacobian.direct(
                        self.bdc2a.direct(x)
                    ), tau
                )
            )
        )
        if out is not None:
            out.fill(ret)
            return out
        else:
            return ret