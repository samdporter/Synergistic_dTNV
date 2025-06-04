from cil.optimisation.functions import Function
try:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    device = 'cpu'
import numpy as np
from sirf.STIR import ImageData
from .schatten_norm_cpu import CPUVectorialTotalVariation
from .Gradients import Jacobian

class BlockDataContainerToArray:
    def __init__(self, domain_geometry, gpu=True):
        self.domain_geometry = domain_geometry
        self.gpu = gpu

    def direct(self, x, out=None):
        if not hasattr(x, "containers"):
            raise ValueError("Input x must be a block data container with a 'containers' attribute.")
        arrays = [d.as_array() for d in x.containers]
        if self.gpu:
            tens = [torch.tensor(arr, device=device) for arr in arrays]
            ret = torch.stack(tens, dim=-1)
        else:
            ret = np.stack(arrays, axis=-1)
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
    GPU/CPU vectorial total variation with optional gradient normalization.
    """

    def __init__(
            self, geometry, weights,
            delta, smoothing='fair', norm='nuclear',
            gpu=True, anatomical=None, stable=True,
            diagonal = False, both_directions=False,
            tail_singular_values=None,
            hessian='diagonal',
            normalise_gradients=False,  # per-voxel L2
            global_normalise_gradients=False,  # per-image L2
            norm_eps=1e-12):
        voxel_sizes = geometry.containers[0].voxel_sizes()
        if isinstance(anatomical, ImageData):
            anatomical = anatomical.as_array()

        # Jacobian operator: maps N×M images → N×M×d (stack of finite diffs)
        self.jacobian = Jacobian(
            voxel_sizes, anatomical=anatomical,
            gpu=gpu, numpy_out=not gpu,
            method="forward",
            diagonal=diagonal,
            both_directions=both_directions,
        )

        self.smoothing = smoothing
        self.hessian = hessian
        self.bdc2a = BlockDataContainerToArray(geometry, gpu=gpu)

        # Pull out the weights as an array/tensor of shape (Nx,Ny,Nz,M)
        self.weights = self.bdc2a.direct(weights)  # shape (..., M)
        self.gpu = gpu

        # Inverse‐weight is used in inv_hessian_diag
        if gpu:
            self.inv_weights = torch.reciprocal(self.weights)
            self.inv_weights = torch.nan_to_num(
                self.inv_weights, nan=0.0, neginf=0.0, posinf=0.0
            )
        else:
            self.inv_weights = np.reciprocal(self.weights, where=self.weights != 0)

        # Normalization flags
        self.normalise_gradients = normalise_gradients
        self.global_normalise_gradients = global_normalise_gradients
        self.norm_eps = norm_eps

        # Choose GPU or CPU backend for the actual vectorial TV
        if gpu:
            if tail_singular_values is not None:
                print(f"tail_singular_values = {tail_singular_values}")
            if stable:
                from .schatten_norm_gpu_slow import GPUVectorialTotalVariation as GpuVTV
            else:
                from .schatten_norm_gpu import GPUVectorialTotalVariation as GpuVTV
            self.vtv = GpuVTV(eps=delta, norm=norm, smoothing_function=smoothing, tail=tail_singular_values)
        else:
            if tail_singular_values is not None:
                raise ValueError("tail_singular_values is only implemented for GPU")
            self.vtv = CPUVectorialTotalVariation(delta, smoothing_function=smoothing)

    def _normalize_per_voxel(self, J):
        # J shape: (..., M, d)
        # Compute L2 norm of each M×d block, then normalize
        if self.gpu:
            norms = torch.linalg.norm(J, dim=-1, keepdim=True)  # (..., M, 1)
        else:
            norms = np.linalg.norm(J, axis=-1, keepdims=True)   # (..., M, 1)

        factor = norms / (norms + self.norm_eps)
        return J / (norms + self.norm_eps) * factor

    def _normalize_global(self, J):
        # J shape: (..., M, d)
        # Compute a single L2 over all voxels and directions, per modality
        if self.gpu:
            norm_factors = torch.sqrt((J ** 2).sum(dim=(0, 1, 2, -1))).clamp_min(self.norm_eps)
            norm_factors = norm_factors.view(1, 1, 1, -1, 1)  # broadcast to (..., M, d)
            return J / norm_factors, norm_factors
        else:
            norm_factors = np.linalg.norm(J, axis=(0, 1, 2, -1)).clip(min=self.norm_eps)
            norm_factors = norm_factors.reshape((1, 1, 1, -1, 1))
            return J / norm_factors, norm_factors

    def __call__(self, x):
        # 1) Pull x (BDC) → array/tensor of shape (..., M)
        x_arr = self.bdc2a.direct(x)                        # shape (nx, ny, nz, M)
        # 2) Compute the Jacobian: J_raw shape (nx, ny, nz, M, d)
        J_raw = self.jacobian.direct(x_arr)

        # 3) Optional gradient normalization
        if self.global_normalise_gradients:
            J, _ = self._normalize_global(J_raw)
        elif self.normalise_gradients:
            J = self._normalize_per_voxel(J_raw)
        else:
            J = J_raw

        # 4) Multiply by weights (shape (..., M) → expand to (..., M, 1))
        w = self.weights.unsqueeze(-1)                      # (..., M, 1)
        U = w * J                                            # (..., M, d)

        # 5) Call GPU/CPU VTV on U (returns per‐voxel TV components)
        return self.vtv(U)

    def call_no_sum(self, x):
        # Same as __call__, but returns the un-summed per‐voxel quantities
        x_arr = self.bdc2a.direct(x)
        J_raw = self.jacobian.direct(x_arr)
        if self.global_normalise_gradients:
            J, _ = self._normalize_global(J_raw)
        elif self.normalise_gradients:
            J = self._normalize_per_voxel(J_raw)
        else:
            J = J_raw

        w = self.weights.unsqueeze(-1)
        U = w * J
        out_U = self.vtv.call_no_sum(U)   # shape: (nx,ny,nz,M)
        return self.bdc2a.adjoint(out_U)

    def gradient(self, x, out=None):
        # 1) Pull x → array/tensor
        x_arr = self.bdc2a.direct(x)                        # (nx, ny, nz, M)
        # 2) Compute J_raw (nx, ny, nz, M, d)
        J_raw = self.jacobian.direct(x_arr)

        # 3) Normalize if needed
        if self.global_normalise_gradients:
            J, norm_factors = self._normalize_global(J_raw)
        elif self.normalise_gradients:
            norms = torch.linalg.norm(J_raw, dim=-1, keepdim=True).clamp_min(self.norm_eps) if self.gpu \
                else np.linalg.norm(J_raw, axis=-1, keepdims=True).clip(min=self.norm_eps)
            J = J_raw / norms
        else:
            J = J_raw

        # 4) Multiply by weights
        w = self.weights.unsqueeze(-1)          # (..., M, 1)
        U = w * J                                # (..., M, d)

        # 5) Compute inner gradient: shape (..., M, d)
        inner = w * self.vtv.gradient(U)        # the vtv.gradient already accounts for smoothing etc.

        # 6) Undo any normalization
        if self.normalise_gradients:
            inner = inner / norms
        elif self.global_normalise_gradients:
            inner = inner / norm_factors

        # 7) Push back to image‐space: J^* (−divergence)
        # In our code, Jacobian.adjoint performs exactly −divg
        ret = self.jacobian.adjoint(inner)      # shape (nx,ny,nz,M)
        ret = self.bdc2a.adjoint(ret)
        if out is not None:
            out.fill(ret)
            return out
        return ret

    def _hess_diag_core(self, x_arr):
        # x_arr: (nx,ny,nz,M)
        # We want diag(H) = diag( J^*  [D^2 \Phi / D\mathcal J^2]  J ).
        # Steps:
        #  a) Compute the “sensitivity” of J w.r.t x: sens = J’s Jacobian‐adjoint identity
        #     In practice, sens = Jacobian.sensitivity(x_arr), which yields the per-voxel
        #     “sum of squares of finite‐difference coefficients,” shape (nx,ny,nz,M).
        sens = self.jacobian.sensitivity(x_arr)     # shape (nx,ny,nz,M)
        #  b) Square it:
        s2 = sens ** 2 if self.gpu else sens ** 2

        #  c) Compute J_raw and normalize if needed
        J_raw = self.jacobian.direct(x_arr)         # (nx,ny,nz,M,d)
        if self.global_normalise_gradients:
            J, norm_factors = self._normalize_global(J_raw)
        elif self.normalise_gradients:
            norms = torch.linalg.norm(J_raw, dim=-1, keepdim=True).clamp_min(self.norm_eps) if self.gpu \
                else np.linalg.norm(J_raw, axis=-1, keepdims=True).clip(min=self.norm_eps)
            J = J_raw / norms
        else:
            J = J_raw

        #  d) Build U = weights * J
        w = self.weights.unsqueeze(-1)              # (..., M, 1)
        U = w * J                                    # (..., M, d)

        #  e) Ask VTV for the per‐voxel second‐derivative diagonal (in \mathcal J‐space), shape (..., M, d)
        d2phi = self.vtv.hessian_diag(U)            # (..., M, d)

        #  f) Undo normalization
        if self.normalise_gradients:
            d2phi = d2phi / (norms ** 2)
        elif self.global_normalise_gradients:
            d2phi = d2phi / (norm_factors ** 2)

        #  g) Multiply by s2 (the “sensitivity^2”) and sum over the last axis (directions)
        #     That yields shape (nx,ny,nz,M).
        diag = (d2phi * s2).sum(dim=-1) if self.gpu \
            else (d2phi * s2).sum(axis=-1)


        return diag.abs()

    def hessian_diag_arr(self, x):
        x_arr = self.bdc2a.direct(x)                  # (nx,ny,nz,M)
        return self._hess_diag_core(x_arr)            # (nx,ny,nz,M)

    def hessian_diag(self, x, out=None):
        # We already have diag(\mathcal J‐space).  To get diag in “image‐space,”
        # we multiply by weights^2 (because we inserted one w in front when building U).
        diag_arr = self.hessian_diag_arr(x)           # (nx,ny,nz,M)
        diag_arr = (self.weights ** 2) * diag_arr     # (nx,ny,nz,M)
        result = self.bdc2a.adjoint(diag_arr)          # push back to BDC
        if out is not None:
            out.fill(result)
            return out
        return result

    def inv_hessian_diag(self, x, out=None, epsilon=0.0):
        # Just invert the array, then multiply by inv_weights^2, then push back
        diag_arr = self.hessian_diag_arr(x) + epsilon
        if self.gpu:
            inv_arr = torch.reciprocal(diag_arr)
            torch.nan_to_num(inv_arr, nan=0.0, posinf=0.0, neginf=0.0, out=inv_arr)
        else:
            inv_arr = np.reciprocal(diag_arr, where=diag_arr != 0)
        inv_arr = (self.inv_weights ** 2) * inv_arr
        result = self.bdc2a.adjoint(inv_arr)
        if out is not None:
            out.fill(result)
            return out
        return result

    def proximal(self, x, tau, out=None):
        x_arr = self.bdc2a.direct(x)                  # (nx,ny,nz,M)
        J_raw = self.jacobian.direct(x_arr)           # (nx,ny,nz,M,d)

        if self.global_normalise_gradients:
            J, norm_factors = self._normalize_global(J_raw)
        elif self.normalise_gradients:
            norms = torch.linalg.norm(J_raw, dim=-1, keepdim=True).clamp_min(self.norm_eps) if self.gpu \
                else np.linalg.norm(J_raw, axis=-1, keepdims=True).clip(min=self.norm_eps)
            J = J_raw / norms
        else:
            J = J_raw

        w = self.weights.unsqueeze(-1)               # (...,M,1)
        U = w * J                                     # (...,M,d)

        proxU = self.vtv.proximal(U, tau)             # (...,M,d)

        if self.normalise_gradients:
            proxU = proxU / norms
        elif self.global_normalise_gradients:
            proxU = proxU / norm_factors

        # Push back to image space:
        ret = self.jacobian.adjoint(proxU * w)        # (nx,ny,nz,M)
        ret = self.bdc2a.adjoint(ret)
        if out is not None:
            out.fill(ret)
            return out
        return ret
