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
                 gpu=True, anatomical=None, stable = True):
        """
        Initializes the WeightedVectorialTotalVariation class.
        """        

        voxel_sizes = geometry.containers[0].voxel_sizes()
        if isinstance(anatomical, ImageData):
            anatomical = anatomical.as_array()
        self.jacobian = Jacobian(voxel_sizes, weights, anatomical=anatomical, gpu=gpu, numpy_out=not gpu)
        self.smoothing = smoothing
        if gpu:
            if stable:
                from .schatten_norm_gpu_slow import GPUVectorialTotalVariation
                self.vtv = GPUVectorialTotalVariation(eps=delta, norm=norm, smoothing_function=smoothing)
            else:
                from .schatten_norm_gpu import GPUVectorialTotalVariation
                self.vtv = GPUVectorialTotalVariation(eps=delta, norm=norm, smoothing_function=smoothing)
        else:
            self.vtv = CPUVectorialTotalVariation(delta, smoothing_function=smoothing)

        self.bdc2a = BlockDataContainerToArray(geometry, gpu=gpu)

    def __call__(self, x):
        return self.vtv(self.jacobian.direct(self.bdc2a.direct(x)))
    
    def call_no_sum(self, x):
        return self.bdc2a.adjoint(self.vtv.call_no_sum(self.jacobian.direct(self.bdc2a.direct(x))))
    
    def gradient(self, x, out=None):
        ret =  self.bdc2a.adjoint(self.jacobian.adjoint(self.vtv.gradient(self.jacobian.direct(self.bdc2a.direct(x)))))
        if out is not None:
            out.fill(ret)
        return ret
    
    def hessian_diag(self, x, out=None):
        ret = self.bdc2a.adjoint(self.jacobian.adjoint(self.vtv.hessian_diag(self.jacobian.direct(self.bdc2a.direct(x)))))
        if out is not None:
            out.fill(ret)
        return ret.abs()
    
    def inv_hessian_diag(self, x, out=None):
        ret = self.bdc2a.adjoint(self.jacobian.adjoint(self.vtv.inv_hessian_diag(self.jacobian.direct(self.bdc2a.direct(x)))))
        if out is not None:
            out.fill(ret)
        return ret.abs()
    
    def proximal(self, x, tau, out=None):
        ret = self.bdc2a.adjoint(self.jacobian.adjoint(self.vtv.proximal(self.jacobian.direct(self.bdc2a.direct(x)), tau)))
        if out is not None:
            out.fill(ret)
        return ret