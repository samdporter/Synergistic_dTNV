from cil.optimisation.functions import Function

class LogDetGramPrior(Function):
    """
    Log-det Gram matrix prior:
        R(x) = sum_voxels log det(I + Ĵᵀ Ĵ)
    where Ĵ is the stack of normalised gradients from each image.
    """

    def __init__(self, geometry, gpu=True, anatomical=None, norm_eps=1e-12):
        self.gpu = gpu
        self.norm_eps = norm_eps
        voxel_sizes = geometry.containers[0].voxel_sizes()
        self.jacobian = Jacobian(
            voxel_sizes, anatomical=anatomical,
            gpu=gpu, numpy_out=not gpu,
            method="forward",
            diagonal=True, both_directions=True
        )
        self.bdc2a = BlockDataContainerToArray(geometry, gpu=gpu)

    def _compute_Jhat_and_norms(self, x):
        J_raw = self.jacobian.direct(self.bdc2a.direct(x))  # shape: (..., N, d)
        norms = torch.linalg.norm(J_raw, dim=-1, keepdim=True).clamp_min(self.norm_eps)
        J_hat = J_raw / norms
        return J_hat, norms, J_raw

    def __call__(self, x):
        J_hat, _, _ = self._compute_Jhat_and_norms(x)
        G = torch.matmul(J_hat, J_hat.transpose(-2, -1))  # (..., N, N)
        I = torch.eye(G.shape[-1], device=G.device).expand_as(G)
        sign, logabsdet = torch.linalg.slogdet(I + G)
        logdet = logabsdet  # if you are sure det > 0 (we shuould be)
        return logdet.sum()

    def gradient(self, x, out=None):
        J_hat, norms, _ = self._compute_Jhat_and_norms(x)
        G = torch.matmul(J_hat, J_hat.transpose(-2, -1))
        I = torch.eye(G.shape[-1], device=G.device).expand_as(G)
        G_inv = torch.linalg.inv(I + G)
        grad_hat = 2 * torch.matmul(G_inv, J_hat)  # (..., N, d)

        J_hat_J_hatT = torch.matmul(J_hat.unsqueeze(-1), J_hat.unsqueeze(-2))  # (..., d, d)
        I_d = torch.eye(J_hat.shape[-1], device=J_hat.device).expand(J_hat.shape[:-1] + (J_hat.shape[-1], J_hat.shape[-1]))
        proj = I_d - J_hat_J_hatT
        grad = torch.matmul(proj, grad_hat.unsqueeze(-1)).squeeze(-1) / norms

        result = self.bdc2a.adjoint(self.jacobian.adjoint(grad))
        if out is not None:
            out.fill(result)
            return out
        return result

    def hessian_diag_arr(self, x):
        x_arr = self.bdc2a.direct(x)
        J_hat, norms, _ = self._compute_Jhat_and_norms(x)

        sens = self.jacobian.sensitivity(x_arr)
        s2 = sens.pow(2) if self.gpu else sens**2

        # Very rough diagonal approximation:
        # Use squared gradient of R, assuming near-linear local behavior
        G = torch.matmul(J_hat, J_hat.transpose(-2, -1))
        I = torch.eye(G.shape[-1], device=G.device).expand_as(G)
        G_inv = torch.linalg.inv(I + G)
        grad_hat = 2 * torch.matmul(G_inv, J_hat)  # (..., N, d)

        J_hat_J_hatT = torch.matmul(J_hat.unsqueeze(-1), J_hat.unsqueeze(-2))  # (..., d, d)
        I_d = torch.eye(J_hat.shape[-1], device=J_hat.device).expand(J_hat.shape[:-1] + (J_hat.shape[-1], J_hat.shape[-1]))
        proj = I_d - J_hat_J_hatT
        grad = torch.matmul(proj, grad_hat.unsqueeze(-1)).squeeze(-1) / norms

        diag = (grad.pow(2) * s2).sum(dim=-1) if self.gpu else (grad**2 * s2).sum(axis=-1)
        return diag.abs()

    def hessian_diag(self, x, out=None):
        diag_arr = self.hessian_diag_arr(x)
        result = self.bdc2a.adjoint(diag_arr)
        if out is not None:
            out.fill(result)
            return out
        return result

    def inv_hessian_diag(self, x, out=None, epsilon=0.0):
        diag_arr = self.hessian_diag_arr(x) + epsilon
        inv_arr = torch.reciprocal(diag_arr) if self.gpu else np.reciprocal(diag_arr, where=diag_arr != 0)
        result = self.bdc2a.adjoint(inv_arr)
        if out is not None:
            out.fill(result)
            return out
        return result

from cil.optimisation.functions import Function
import torch

class MutualInformationGradientPrior(Function):
    def __init__(self, geometry, sigma=1.0, use_autograd=True, norm_eps=1e-12, gpu=True, anatomical=None, max_points=10000):
        from .Gradients import Jacobian

        self.sigma = sigma
        self.norm_eps = norm_eps
        self.use_autograd = use_autograd
        self.gpu = gpu
        self.max_points = max_points
        self.device = 'cuda' if gpu else 'cpu'

        voxel_sizes = geometry.containers[0].voxel_sizes()
        self.jacobian = Jacobian(
            voxel_sizes, anatomical=anatomical,
            gpu=gpu, numpy_out=not gpu,
            method="forward",
            diagonal=True, both_directions=True
        )
        self.bdc2a = BlockDataContainerToArray(geometry, gpu=gpu)
        self._log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device))

    def _get_joint_gradients(self, x):
        J_all = self.jacobian.direct(self.bdc2a.direct(x)).view(-1, 6)
        if J_all.shape[0] > self.max_points:
            idx = torch.randperm(J_all.shape[0], device=J_all.device)[:self.max_points]
            J = J_all[idx]
        else:
            idx = torch.arange(J_all.shape[0], device=J_all.device)
            J = J_all
        return J, idx, J_all.shape

    def __call__(self, x):
        J, _, _ = self._get_joint_gradients(x)
        J = J.detach()
        J.requires_grad_(self.use_autograd)
        return self._compute_mi(J)

    def _compute_mi(self, J):
        N = J.shape[0]
        sigma = self.sigma

        def log_kde(dists, dim, d):
            logs = -0.5 * dists / sigma**2
            logs += -0.5 * d * self._log_2pi - d * torch.log(torch.tensor(sigma, device=J.device))
            return torch.logsumexp(logs, dim=dim) - torch.log(torch.tensor(N, dtype=J.dtype, device=J.device))

        D_joint = torch.cdist(J, J, p=2).pow(2)
        log_joint = log_kde(D_joint, 1, 6)

        x1, x2 = J[:, :3], J[:, 3:]
        D1 = torch.cdist(x1, x1, p=2).pow(2)
        D2 = torch.cdist(x2, x2, p=2).pow(2)
        log_p1 = log_kde(D1, 1, 3)
        log_p2 = log_kde(D2, 1, 3)

        mi = log_joint - log_p1 - log_p2
        return -mi.sum()

    def _manual_gradient(self, J):
        N, D = J.shape
        sigma = self.sigma
        log_2pi = self._log_2pi

        diff = J.unsqueeze(1) - J.unsqueeze(0)
        sq_dists = (diff ** 2).sum(-1)
        log_kernel_joint = -0.5 * sq_dists / sigma**2 - 0.5 * D * log_2pi - D * torch.log(torch.tensor(sigma, device=J.device))
        w_joint = torch.softmax(log_kernel_joint, dim=1)
        grad_joint = (w_joint.unsqueeze(-1) * diff).sum(dim=1) / sigma**2

        x1, x2 = J[:, :3], J[:, 3:]

        def marginal_grad(x):
            d = x.shape[1]
            diff = x.unsqueeze(1) - x.unsqueeze(0)
            sq_d = (diff ** 2).sum(-1)
            log_k = -0.5 * sq_d / sigma**2 - 0.5 * d * log_2pi - d * torch.log(torch.tensor(sigma, device=x.device))
            w = torch.softmax(log_k, dim=1)
            return (w.unsqueeze(-1) * diff).sum(dim=1) / sigma**2

        grad_x1 = marginal_grad(x1)
        grad_x2 = marginal_grad(x2)

        return torch.cat([grad_x1, grad_x2], dim=1) - grad_joint

    def gradient(self, x, out=None):
        J, idx, full_shape = self._get_joint_gradients(x)
        if self.use_autograd:
            J.requires_grad_(True)
            loss = self._compute_mi(J)
            loss.backward()
            grad_flat = J.grad
        else:
            grad_flat = self._manual_gradient(J)

        # Scatter back to full shape
        grad_full = torch.zeros((full_shape[0], 6), dtype=grad_flat.dtype, device=grad_flat.device)
        grad_full[idx] = grad_flat
        grad = grad_full.view(*self.jacobian.direct(self.bdc2a.direct(x)).shape)

        result = self.bdc2a.adjoint(self.jacobian.adjoint(grad))
        if out is not None:
            out.fill(result)
            return out
        return result

    def hessian_diag_arr(self, x):
        grad = self.gradient(x)
        g_arr = self.bdc2a.direct(grad)
        return (g_arr ** 2).sum(dim=-1)

    def hessian_diag(self, x, out=None):
        diag_arr = self.hessian_diag_arr(x)
        result = self.bdc2a.adjoint(diag_arr)
        if out is not None:
            out.fill(result)
            return out
        return result

    def inv_hessian_diag(self, x, out=None, epsilon=0.0):
        diag_arr = self.hessian_diag_arr(x) + epsilon
        inv_arr = torch.reciprocal(diag_arr)
        result = self.bdc2a.adjoint(inv_arr)
        if out is not None:
            out.fill(result)
            return out
        return result

class MutualInformationImagePrior(Function):
    def __init__(self, geometry, sigma=1.0, use_autograd=True, gpu=True, max_points=10000):
        self.sigma = sigma
        self.use_autograd = use_autograd
        self.gpu = gpu
        self.device = torch.device('cuda' if gpu else 'cpu')
        self.max_points = max_points

        self.bdc2a = BlockDataContainerToArray(geometry, gpu=gpu)
        self._log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device))

    def _get_joint_vectors(self, x):
        data = self.bdc2a.direct(x)
        vecs = data.view(-1, data.shape[-1])
        if vecs.shape[0] > self.max_points:
            idx = torch.randperm(vecs.shape[0], device=vecs.device)[:self.max_points]
            vecs = vecs[idx]
        return vecs

    def __call__(self, x):
        J = self._get_joint_vectors(x).detach()
        J.requires_grad_(self.use_autograd)
        return self._compute_mi(J)

    def _compute_mi(self, J):
        N, D = J.shape
        sigma = self.sigma

        def log_kde(dists, dim, d):
            logs = -0.5 * dists / sigma**2
            logs += -0.5 * d * self._log_2pi - d * torch.log(torch.tensor(sigma, device=self.device))
            return torch.logsumexp(logs, dim=dim) - torch.log(torch.tensor(N, dtype=J.dtype, device=self.device))

        D_joint = torch.cdist(J, J, p=2).pow(2)
        log_joint = log_kde(D_joint, 1, D)

        # Marginals
        x1, x2 = J[:, 0].unsqueeze(1), J[:, 1].unsqueeze(1)
        D1 = torch.cdist(x1, x1, p=2).pow(2)
        D2 = torch.cdist(x2, x2, p=2).pow(2)
        log_p1 = log_kde(D1, 1, 1)
        log_p2 = log_kde(D2, 1, 1)

        return -(log_joint - log_p1 - log_p2).sum()

    def gradient(self, x, out=None):
        data = self.bdc2a.direct(x)
        vecs = data.view(-1, data.shape[-1])
        if vecs.shape[0] > self.max_points:
            idx = torch.randperm(vecs.shape[0], device=vecs.device)[:self.max_points]
            J = vecs[idx]
            back = torch.zeros_like(vecs)
        else:
            J = vecs
            idx = torch.arange(vecs.shape[0], device=vecs.device)
            back = torch.zeros_like(J)

        if self.use_autograd:
            J.requires_grad_(True)
            loss = self._compute_mi(J)
            loss.backward()
            grad = J.grad
        else:
            grad = self._manual_gradient(J)

        back[idx] = grad
        grad_full = back.view_as(data)
        result = self.bdc2a.adjoint(grad_full)
        if out is not None:
            out.fill(result)
            return out
        return result

    def _manual_gradient(self, J):
        N, D = J.shape
        sigma = self.sigma

        diff = J.unsqueeze(1) - J.unsqueeze(0)
        sq_dists = (diff ** 2).sum(-1)
        log_kernel_joint = -0.5 * sq_dists / sigma**2 - 0.5 * D * self._log_2pi - D * torch.log(torch.tensor(sigma, device=J.device))
        w_joint = torch.softmax(log_kernel_joint, dim=1)
        grad_joint = (w_joint.unsqueeze(-1) * diff).sum(dim=1) / sigma**2

        grad_p1 = self._marginal_grad(J[:, 0:1])
        grad_p2 = self._marginal_grad(J[:, 1:2])

        return torch.cat([grad_p1, grad_p2], dim=1) - grad_joint

    def _marginal_grad(self, x):
        d = x.shape[1]
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        sq_d = (diff ** 2).sum(-1)
        log_k = -0.5 * sq_d / self.sigma**2 - 0.5 * d * self._log_2pi - d * torch.log(torch.tensor(self.sigma, device=x.device))
        w = torch.softmax(log_k, dim=1)
        return (w.unsqueeze(-1) * diff).sum(dim=1) / self.sigma**2

    def hessian_diag_arr(self, x):
        g = self.gradient(x)
        g_arr = self.bdc2a.direct(g)
        return (g_arr ** 2).sum(dim=-1)

    def hessian_diag(self, x, out=None):
        diag = self.hessian_diag_arr(x)
        if diag.ndim == 3:
            result = self.bdc2a.adjoint(diag)
        else:
            split = torch.unbind(diag, dim=-1)
            result = self.bdc2a.adjoint(torch.stack(split, dim=-1))
        if out is not None:
            out.fill(result)
            return out
        return result

    def inv_hessian_diag(self, x, out=None, epsilon=0.0):
        diag = self.hessian_diag_arr(x) + epsilon
        inv = torch.reciprocal(diag)
        if diag.ndim == 3:
            result = self.bdc2a.adjoint(inv)
        else:
            split = torch.unbind(inv, dim=-1)
            result = self.bdc2a.adjoint(torch.stack(split, dim=-1))
        if out is not None:
            out.fill(result)
            return out
        return result
