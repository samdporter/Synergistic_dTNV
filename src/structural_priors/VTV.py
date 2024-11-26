from cil.optimisation.functions import Function
import array_api_compat.numpy as np
import array_api_compat.torch as torch
import torch

from sirf.STIR import ImageData


from .schatten_norm_cpu import CPUVectorialTotalVariation
from.Gradients import Jacobian

class BlockDataContainerToArray():

    def __init__(self, domain_geometry, gpu=True):

        self.domain_geometry = domain_geometry
        self.gpu = gpu

        if self.gpu:
            # use torch as xp
            self.xp = torch
        else:
            self.xp = np

    def direct(self, x, out=None):

        if not hasattr(x, "containers"):
            raise ValueError("Input x must be a block data container with a 'containers' attribute.")
        ret = self.xp.stack([self.xp.asarray(d.as_array()) for d in x.containers], axis=-1)
        if out is not None:
            out.fill(ret)
        return ret

    def adjoint(self, x, out=None):

        if self.gpu and isinstance(x, torch.Tensor):
            x_arr = x.cpu().numpy()
        else:
            x_arr = np.asarray(x)
        res = self.domain_geometry.clone()
        for i, r in enumerate(res.containers):
            r.fill(x_arr[..., i])
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
        self.jacobian = Jacobian(voxel_sizes, weights, anatomical=anatomical, gpu=gpu)
        self.smoothing = smoothing
        if gpu:
            if stable:
                from .schatten_norm_gpu_slow import GPUVectorialTotalVariation
                self.vtv = GPUVectorialTotalVariation(eps=delta, norm=norm, smoothing_function=smoothing)
            else:
                from .schatten_norm_gpu import GPUVectorialTotalVariation
                self.vtv = GPUVectorialTotalVariation(eps=delta, norm=norm, smoothing_function=smoothing)
        else:
            if smoothing != 'fair':
                raise ValueError("Only fair smoothing is supported for CPU")
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
        return ret
    
    def inv_hessian_diag(self, x, out=None):
        ret = self.bdc2a.adjoint(self.jacobian.adjoint(self.vtv.inv_hessian_diag(self.jacobian.direct(self.bdc2a.direct(x)))))
        if out is not None:
            out.fill(ret)
        return ret
    
    def proximal(self, x, tau, out=None):
        ret = self.bdc2a.adjoint(self.jacobian.adjoint(self.vtv.proximal(self.jacobian.direct(self.bdc2a.direct(x)), tau)))
        if out is not None:
            out.fill(ret)
        return ret