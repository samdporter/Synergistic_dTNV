import numpy as np
from numba import njit, prange, jit
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inner_product(x, y):
    return torch.sum(x * y).item()

class Operator():

    def __init__():
        raise NotImplementedError

    def __call__(self, x):
        return self.direct(x)

    def forward(self, x):
        return self.direct(x)

    def backward(self, x):
        return self.adjoint(x)
    
    def adjoint(self, x):
        raise NotImplementedError
    
    def direct(self, x):
        raise NotImplementedError
    
    def calculate_norm(self, x, num_iterations=10):
        if not hasattr(self, 'norm'):
            self.norm = power_iteration(self, x, num_iterations=num_iterations)
            return self.norm
        else:
            return self.norm
        
    def get_adjoint(self):
        return AdjointOperator(self)
        
    def __mul__(self, other):
        return ScaledOperator(self, other)
    
    def __rmul__(self, other):
        return ScaledOperator(self, other)
    
    def __div__(self, other):
        return ScaledOperator(self, 1/other)
    
    def __truediv__(self, other):
        return ScaledOperator(self, 1/other)

    def __rdiv__(self, other):
        return ScaledOperator(self, other)
    
class AdjointOperator(Operator):
    
    def __init__(self, operator):
        self.operator = operator

    def direct(self, x):
        return self.operator.adjoint(x)

    def adjoint(self, x):
        return self.operator.direct(x)
    
class ScaledOperator(Operator):

    def __init__(self, operator, scale):
        self.operator = operator
        self.scale = scale

    def direct(self, x):
        return self.scale * self.operator.direct(x)

    def adjoint(self, x):
        return 1/self.scale * self.operator.adjoint(x)
    
class CompositionOperator(Operator):
    
    def __init__(self, ops):
        self.ops = ops
        
    def direct(self, x):
        res = x.copy()
        for op in self.ops:
            res = op.direct(res)
        return res
    
    def adjoint(self,x):
        res = x.copy()
        for op in self.ops[::-1]:
            res = op.adjoint(res)
        return res

class Jacobian:
    def __init__(self, voxel_sizes=(1, 1, 1), bnd_cond='Periodic', method='forward',
                 anatomical=None, numpy_out=True, diagonal=False, both_directions=False,
                 gpu=True):

        self.gpu = gpu
        self.voxel_sizes = voxel_sizes
        self.method = method
        self.numpy_out = numpy_out
        self.anatomical = anatomical
        self.diagonal = diagonal
        self.both_directions = both_directions

        if isinstance(anatomical, list):
            self.grad = [self._initialise_gradient(a, voxel_sizes, method, bnd_cond, gpu, diagonal, both_directions)
                         for a in anatomical]
            print('Multiple anatomical images detected. This will fail if different number of images are passed to direct and adjoint methods')
        else:
            self.grad = self._initialise_gradient(anatomical, voxel_sizes, method, bnd_cond, gpu, diagonal, both_directions)

    def _initialise_gradient(self, anatomical, voxel_sizes, method, bnd_cond, gpu, diagonal, both_directions):
        if anatomical is None:
            return Gradient(voxel_sizes=voxel_sizes, method=method, bnd_cond=bnd_cond,
                            gpu=gpu, numpy_out=False, diagonal=diagonal, both_directions=both_directions)
        if gpu:
            anatomical = torch.tensor(anatomical, device=device) if isinstance(anatomical, np.ndarray) else anatomical.to(device)
        return DirectionalGradient(
            anatomical,
            voxel_sizes=voxel_sizes,
            method=method,
            bnd_cond=bnd_cond,
            diagonal=diagonal,
            both_directions=both_directions,
            gpu=gpu,
            numpy_out=False
        )


    
    def direct(self, images):
            
        # if gpu is enabled, convert images to torch tensor
        if self.gpu:
            images = torch.tensor(images, device=device) if not isinstance(images, torch.Tensor) else images.to(device)

        num_images = images.shape[-1]
        # if weights is a list of arrays, we need to expand dims
        if isinstance(self.grad, list):
            jac_list = [self.grad[idx].direct(images[..., idx]) for idx in range(num_images)]
        else:
            jac_list = [self.grad.direct(images[..., idx]) for idx in range(num_images)]

        if self.gpu:
            return torch.stack(jac_list, dim=-2).cpu().numpy() if self.numpy_out else torch.stack(jac_list, dim=-2)
        else:
            return np.stack(jac_list, axis=-2)
        
    def adjoint(self, jacobians):
        
        if self.gpu:
            if not isinstance(jacobians, torch.Tensor):
                jacobians = torch.tensor(jacobians, device=device)
            else:
                jacobians = jacobians.to(device)
                
        num_images = jacobians.shape[-2]
        adjoint_list = []
        for idx in range(num_images):
            if isinstance(self.grad, list):
                adjoint_list.append(self.grad[idx].adjoint(jacobians[..., idx,:]))
            else:
                adjoint_list.append(self.grad.adjoint(jacobians[..., idx,:]))
        if self.gpu:
            return torch.stack(adjoint_list, dim=-1).cpu().numpy() if self.numpy_out else torch.stack(adjoint_list, dim=-1)
        else:
            res =  np.stack(adjoint_list, axis=-1)
        return res

    # --- Jacobian.sensitivity ------------------------------------------
    def sensitivity(self, images):
        """
        Return voxel-wise scaling factor for each gradient channel,
        matching the stencil used in Gradient.
        
        For finite difference (f(x+h) - f(x))/h, the sensitivity is 1/h
        where h is the physical step size: h = sqrt(Σ(shift_i * voxel_size_i)²)
        
        Output shape: (z, y, x, n_images, d)
        """
        vs = self.voxel_sizes

        directions = [(1,0,0), (0,1,0), (0,0,1)]
        if self.diagonal:
            directions += [(1,1,0), (1,0,1), (0,1,1)]

        modes = ['forward', 'backward'] if self.both_directions else [self.method]
        n_channels = len(directions) * len(modes)

        shape = (*images.shape[:-1], images.shape[-1], n_channels)
        if self.gpu:
            S = torch.ones(shape, device=images.device)
        else:
            S = np.ones(shape)

        idx = 0
        for shift in directions:
            # Physical step size: Δℓ = sqrt(Σ(α_i * Δx_i)²)
            physical_step = np.sqrt(sum((s * v)**2 for s, v in zip(shift, vs)))
            
            # Sensitivity is 1/Δℓ (inverse of step size)
            scale = 1.0 / physical_step
            
            for _ in modes:
                S[..., :, idx] *= scale
                idx += 1

        return S
        
    # --- Jacobian.calculate_norm ---------------------------------------
    def calculate_norm(self):
        """‖J‖₂ for the chosen stencil."""
        if not hasattr(self, '_norm'):
            vs = self.voxel_sizes
            directions = [(1,0,0), (0,1,0), (0,0,1)]
            if self.diagonal:
                directions += [(1,1,0), (1,0,1), (0,1,1)]
            num_shifts = 2 if self.both_directions else 1

            self._norm = 2.0 * np.sqrt(
                sum(num_shifts * sum((s / v)**2 for s, v in zip(shift, vs)) for shift in directions)
            )
        return self._norm
  
class Gradient:
    def __init__(self, voxel_sizes, method='forward', bnd_cond='Periodic', 
                 numpy_out=False, gpu=True, diagonal=False, both_directions=False):
        self.voxel_sizes = voxel_sizes
        self.method = method
        self.bnd_cond = bnd_cond
        self.numpy_out = numpy_out
        self.gpu = gpu
        self.diagonal = diagonal
        self.both_directions = both_directions

        self.directions = [(1,0,0), (0,1,0), (0,0,1)]
        if self.diagonal:
            self.directions += [(1,1,0), (1,0,1), (0,1,1)]
            
    def diff_along(self, x, shift, mode='forward'):
        shift = torch.tensor(shift)
        dims = (0, 1, 2)

        if mode not in ('forward', 'backward'):
            raise ValueError("Unsupported mode")

        if self.bnd_cond == 'Periodic':
            shift_amt = -shift if mode == 'forward' else shift
            shifted = torch.roll(x, shifts=tuple(int(s.item()) for s in shift_amt), dims=dims)
            return shifted - x if mode == 'forward' else x - shifted
        elif self.bnd_cond == 'Neumann':
            raise NotImplementedError("Neumann boundary condition not implemented for GPU mode.")

    def direct(self, x):
        if not self.gpu:
            raise NotImplementedError("CPU mode not supported in diagonal/both mode.")
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=device)
        else:
            x = x.to(device)

        results = []
        for shift in self.directions:
            modes = ['forward', 'backward'] if self.both_directions else [self.method]
            for mode in modes:
                out = self.diff_along(x, shift, mode)
                vnorm = np.sqrt(sum((s * v)**2 for s, v in zip(shift, self.voxel_sizes)))
                out /= vnorm
                results.append(out)

        result = torch.stack(results, dim=-1)
        return result.cpu().numpy() if self.numpy_out else result

    def adjoint(self, x):
        if not self.gpu:
            raise NotImplementedError("CPU mode not supported in diagonal/both mode.")
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=device)
        else:
            x = x.to(device)

        result = torch.zeros_like(x[..., 0])
        d_idx = 0
        for shift in self.directions:
            modes = ['forward', 'backward'] if self.both_directions else [self.method]
            for mode in modes:
                adj_mode = 'backward' if mode == 'forward' else 'forward'
                comp = self.diff_along(x[..., d_idx], shift, adj_mode)
                vnorm = np.sqrt(sum((s * v)**2 for s, v in zip(shift, self.voxel_sizes)))
                result -= comp / vnorm
                d_idx += 1

        return result.cpu().numpy() if self.numpy_out else result

    def calculate_norm(self):
        if not hasattr(self, '_norm') or self._norm is None:
            vs = self.voxel_sizes
            shifts = self.directions
            if self.both_directions:
                shifts = shifts * 2
            self._norm = 2.0 * np.sqrt(sum(sum((s / v)**2 for s, v in zip(shift, vs)) for shift in shifts))
        return self._norm
    
class DirectionalGradient(Operator):

    def __init__(self, anatomical, voxel_sizes, gamma=1, eta=None,
                  method='forward', bnd_cond='Neumann', numpy_out=False,
                  gpu=False, diagonal=False, both_directions=False
                  ) -> None:

        self.anatomical = anatomical
        self.voxel_size = voxel_sizes
        self.gamma = gamma
        self.method = method
        self.bnd_cond = bnd_cond
        self.numpy_out = numpy_out
        self.gpu = gpu
        self.gradient = Gradient(
            voxel_sizes=self.voxel_size, 
            method=self.method, 
            bnd_cond=self.bnd_cond, 
            numpy_out=False, gpu=self.gpu,
            diagonal=diagonal,
            both_directions=both_directions
            )
        self.anatomical_grad = self.gradient.direct(self.anatomical)
        if eta is None:
            # make 1000 times smaller than the dynamic range of gradient
            if torch is not None and isinstance(
                self.anatomical_grad, 
                torch.Tensor
            ):
                # Use tensor‐native max/min
                max_val = self.anatomical_grad.max().item()
                min_val = self.anatomical_grad.min().item()
            else:
                # Fallback to NumPy
                arr = np.asarray(self.anatomical_grad)
                max_val = arr.max()
                min_val = arr.min()

            self.eta = (max_val - min_val) / 100000
        else:
            self.eta = eta

        if gpu:
            self.directional_op = gpu_directional_op   
            self.eta = torch.tensor(self.eta, device=device)
            self.gamma = torch.tensor(self.gamma, device=device)
            if not isinstance(self.anatomical_grad, torch.Tensor):
                self.anatomical_grad = torch.tensor(self.anatomical_grad, device=device)
            else:
                self.anatomical_grad.to(device)  
        else:
            self.directional_op = directional_op
            # astype float64 for numba
            self.eta = np.float32(self.eta)
            self.gamma = np.float32(self.gamma)
            self.anatomical_grad = self.anatomical_grad.astype(np.float32)

    def direct(self, x):
        if self.gpu:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, device=device)
            else:
                x = x.to(device)
        gradient = self.gradient.direct(x)
        res =  self.directional_op(gradient, self.anatomical_grad, self.gamma, self.eta)
        if self.gpu:
            return res.cpu().numpy() if self.numpy_out else res
        else:
            return res
        
    def adjoint(self, x):
        if self.gpu:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, device=device)
            else:
                x = x.to(device)
        else:
            x = x.astype(np.float32)
        x = self.directional_op(x, self.anatomical_grad, self.gamma, self.eta)
        res = self.gradient.adjoint(x)
        if self.gpu:
            return res.cpu().numpy() if self.numpy_out else res
        else:
            return res

class CPUFiniteDifferenceOperator(Operator):

    """
    Numpy implementation of finite difference operator
    """
    
    def __init__(self, voxel_size, direction=None, method='forward', bnd_cond='Neumann'):
        self.voxel_size= voxel_size
        self.direction = direction
        self.method = method
        self.bnd_cond = bnd_cond
        
        if self.voxel_size<= 0:
            raise ValueError('Need a positive voxel size')

    def get_slice(self, x, start, stop, end=None):
        tmp = [slice(None)] * x.ndim
        tmp[self.direction] = slice(start, stop, end)
        return tmp
    

    def direct(self, x, out = None):
        
        outa = np.zeros_like(x) if out is None else out

        #######################################################################
        ##################### Forward differences #############################
        #######################################################################
                
        if self.method == 'forward':  
            
            # interior nodes
            np.subtract( x[tuple(self.get_slice(x, 2, None))], \
                             x[tuple(self.get_slice(x, 1,-1))], \
                             out = outa[tuple(self.get_slice(x, 1, -1))])               

            if self.bnd_cond == 'Neumann':
                
                # left boundary
                np.subtract(x[tuple(self.get_slice(x, 1,2))],\
                            x[tuple(self.get_slice(x, 0,1))],
                            out = outa[tuple(self.get_slice(x, 0,1))]) 
                
                
            elif self.bnd_cond == 'Periodic':
                
                # left boundary
                np.subtract(x[tuple(self.get_slice(x, 1,2))],\
                            x[tuple(self.get_slice(x, 0,1))],
                            out = outa[tuple(self.get_slice(x, 0,1))])  
                
                # right boundary
                np.subtract(x[tuple(self.get_slice(x, 0,1))],\
                            x[tuple(self.get_slice(x, -1,None))],
                            out = outa[tuple(self.get_slice(x, -1,None))])  
                
            else:
                raise ValueError('Not implemented')                
                
        #######################################################################
        ##################### Backward differences ############################
        #######################################################################                

        elif self.method == 'backward':   
                                   
            # interior nodes
            np.subtract( x[tuple(self.get_slice(x, 1, -1))], \
                             x[tuple(self.get_slice(x, 0,-2))], \
                             out = outa[tuple(self.get_slice(x, 1, -1))])              
            
            if self.bnd_cond == 'Neumann':
                    
                    # right boundary
                    np.subtract( x[tuple(self.get_slice(x, -1, None))], \
                                 x[tuple(self.get_slice(x, -2,-1))], \
                                 out = outa[tuple(self.get_slice(x, -1, None))]) 
                    
            elif self.bnd_cond == 'Periodic':
                  
                # left boundary
                np.subtract(x[tuple(self.get_slice(x, 0,1))],\
                            x[tuple(self.get_slice(x, -1,None))],
                            out = outa[tuple(self.get_slice(x, 0,1))])  
                
                # right boundary
                np.subtract(x[tuple(self.get_slice(x, -1,None))],\
                            x[tuple(self.get_slice(x, -2,-1))],
                            out = outa[tuple(self.get_slice(x, -1,None))]) 
                
            else:
                raise ValueError('Not implemented')                 
        
        #######################################################################
        ##################### central differences ############################
        #######################################################################
        
        
        elif self.method == 'central':
            
            # interior nodes
            np.subtract( x[tuple(self.get_slice(x, 2, None))], \
                             x[tuple(self.get_slice(x, 0,-2))], \
                             out = outa[tuple(self.get_slice(x, 1, -1))]) 
            
            outa[tuple(self.get_slice(x, 1, -1))] /= 2.
            
            if self.bnd_cond == 'Neumann':
                            
                # left boundary
                np.subtract( x[tuple(self.get_slice(x, 1, 2))], \
                                 x[tuple(self.get_slice(x, 0,1))], \
                                 out = outa[tuple(self.get_slice(x, 0, 1))])  
                outa[tuple(self.get_slice(x, 0, 1))] /=2.
                
                # left boundary
                np.subtract( x[tuple(self.get_slice(x, -1, None))], \
                                 x[tuple(self.get_slice(x, -2,-1))], \
                                 out = outa[tuple(self.get_slice(x, -1, None))])
                outa[tuple(self.get_slice(x, -1, None))] /=2.                
                
            elif self.bnd_cond == 'Periodic':
                pass
                
               # left boundary
                np.subtract( x[tuple(self.get_slice(x, 1, 2))], \
                                 x[tuple(self.get_slice(x, -1,None))], \
                                 out = outa[tuple(self.get_slice(x, 0, 1))])                  
                outa[tuple(self.get_slice(x, 0, 1))] /= 2.
                
                
                # left boundary
                np.subtract( x[tuple(self.get_slice(x, 0, 1))], \
                                 x[tuple(self.get_slice(x, -2,-1))], \
                                 out = outa[tuple(self.get_slice(x, -1, None))]) 
                outa[tuple(self.get_slice(x, -1, None))] /= 2.

            else:
                raise ValueError('Not implemented')                 
                
        else:
                raise ValueError('Not implemented')                
        
        if self.voxel_size!= 1.0:
            outa /= self.voxel_size

        return outa            
                 
        
    def adjoint(self, x, out=None):
        
        # Adjoint operation defined as  
                      
        outa = np.zeros_like(x) if out is None else out
            
        #######################################################################
        ##################### Forward differences #############################
        #######################################################################            
            

        if self.method == 'forward':    
            
            # interior nodes
            np.subtract( x[tuple(self.get_slice(x, 1, -1))], \
                             x[tuple(self.get_slice(x, 0,-2))], \
                             out = outa[tuple(self.get_slice(x, 1, -1))])              
            
            if self.bnd_cond == 'Neumann':            

                # left boundary
                outa[tuple(self.get_slice(x, 0,1))] = x[tuple(self.get_slice(x, 0,1))]                
                
                # right boundary
                outa[tuple(self.get_slice(x, -1,None))] = - x[tuple(self.get_slice(x, -2,-1))]  
                
            elif self.bnd_cond == 'Periodic':            

                # left boundary
                np.subtract(x[tuple(self.get_slice(x, 0,1))],\
                            x[tuple(self.get_slice(x, -1,None))],
                            out = outa[tuple(self.get_slice(x, 0,1))])  
                # right boundary
                np.subtract(x[tuple(self.get_slice(x, -1,None))],\
                            x[tuple(self.get_slice(x, -2,-1))],
                            out = outa[tuple(self.get_slice(x, -1,None))]) 
                
            else:
                raise ValueError('Not implemented')                 

        #######################################################################
        ##################### Backward differences ############################
        #######################################################################                
                
        elif self.method == 'backward': 
            
            # interior nodes
            np.subtract( x[tuple(self.get_slice(x, 2, None))], \
                             x[tuple(self.get_slice(x, 1,-1))], \
                             out = outa[tuple(self.get_slice(x, 1, -1))])             
            
            if self.bnd_cond == 'Neumann':             
                
                # left boundary
                outa[tuple(self.get_slice(x, 0,1))] = x[tuple(self.get_slice(x, 1,2))]                
                
                # right boundary
                outa[tuple(self.get_slice(x, -1,None))] = - x[tuple(self.get_slice(x, -1,None))] 
                
                
            elif self.bnd_cond == 'Periodic':
            
                # left boundary
                np.subtract(x[tuple(self.get_slice(x, 1,2))],\
                            x[tuple(self.get_slice(x, 0,1))],
                            out = outa[tuple(self.get_slice(x, 0,1))])  
                
                # right boundary
                np.subtract(x[tuple(self.get_slice(x, 0,1))],\
                            x[tuple(self.get_slice(x, -1,None))],
                            out = outa[tuple(self.get_slice(x, -1,None))])              
                            
            else:
                raise ValueError('Not implemented')
                
                
        #######################################################################
        ##################### central differences ############################
        #######################################################################

        elif self.method == 'central':
            
            # interior nodes
            np.subtract( x[tuple(self.get_slice(x, 2, None))], \
                             x[tuple(self.get_slice(x, 0,-2))], \
                             out = outa[tuple(self.get_slice(x, 1, -1))]) 
            outa[tuple(self.get_slice(x, 1, -1))] /= 2.0
            

            if self.bnd_cond == 'Neumann':
                
                # left boundary
                np.add(x[tuple(self.get_slice(x, 0,1))],\
                            x[tuple(self.get_slice(x, 1,2))],
                            out = outa[tuple(self.get_slice(x, 0,1))])
                outa[tuple(self.get_slice(x, 0,1))] /= 2.0

                # right boundary
                np.add(x[tuple(self.get_slice(x, -1,None))],\
                            x[tuple(self.get_slice(x, -2,-1))],
                            out = outa[tuple(self.get_slice(x, -1,None))])  

                outa[tuple(self.get_slice(x, -1,None))] /= -2.0               
                                                            
                
            elif self.bnd_cond == 'Periodic':
                
                # left boundary
                np.subtract(x[tuple(self.get_slice(x, 1,2))],\
                            x[tuple(self.get_slice(x, -1,None))],
                            out = outa[tuple(self.get_slice(x, 0,1))])
                outa[tuple(self.get_slice(x, 0,1))] /= 2.0
                
                # right boundary
                np.subtract(x[tuple(self.get_slice(x, 0,1))],\
                            x[tuple(self.get_slice(x, -2,-1))],
                            out = outa[tuple(self.get_slice(x, -1,None))])
                outa[tuple(self.get_slice(x, -1,None))] /= 2.0
                
                                
            else:
                raise ValueError('Not implemented') 
                                             
        else:
                raise ValueError('Not implemented')                  
                               
        #outa *= -1.
        if self.voxel_size!= 1.0:
            outa /= self.voxel_size                     
            
        return outa
    
@njit(parallel=True)
def directional_op(image_gradient, anatomical_gradient, gamma=1, eta=1e-6):
    """
    Calculate the directional operator of a 3D image optimized with Numba JIT
    image_gradient: 3D array of image gradients
    anatomical_gradient: 3D array of anatomical gradients
    """
    out = np.empty_like(image_gradient)
    
    D, H, W, i = anatomical_gradient.shape

    for d in prange(D):
        for h in prange(H):
            for w in prange(W):
                xi = anatomical_gradient[d, h, w] / (np.sqrt(np.sum(anatomical_gradient[d, h, w]**2)) + eta**2)
                out[d,h,w] = (image_gradient[d,h,w] - gamma * np.dot(image_gradient[d,h,w], xi) * xi)
    return out

def gpu_directional_op(image_gradient, anatomical_gradient, gamma=1, eta=1e-6):
    """
    Calculate the directional operator of a 3D image optimized with torch
    image_gradient: 3D array of image gradients
    anatomical_gradient: 3D array of anatomical gradients
    """

    xi = anatomical_gradient / (torch.norm(anatomical_gradient, p=2, dim=-1, keepdim=True) + eta**2)
    
    out = image_gradient - gamma * torch.sum(image_gradient * xi, dim=-1, keepdim=True) * xi
    return out

@jit(forceobj=True)
def power_iteration(operator, input_shape, num_iterations=100):
    """
    Approximate the largest singular value of an operator using power iteration.

    Args:
        operator: The operator with defined forward and adjoint methods.
        num_iterations (int): The number of iterations to refine the approximation.

    Returns:
        float: The approximated largest singular value of the operator.
    """
    
    # Start with a random input of appropriate shape
    input_data = np.random.randn(*input_shape)
    length = len(input_shape)
    
    for i in range(num_iterations):
        # Apply forward operation
        output_data = operator.forward(input_data)
        
        # Apply adjoint operation
        input_data = operator.adjoint(output_data)
        
        # Normalize the result
        if length == 3:
            norm = fast_norm_parallel_3d(input_data)
        elif length == 4:
            norm = fast_norm_parallel_4d(input_data)

        input_data /= norm

        # print iteration on and remove previous line
        print(f'Iteration {i+1}/{num_iterations}', end='\r')
    
    return norm

@njit(parallel=True)
def fast_norm_parallel_3d(arr):
    s = arr.shape
    total = 0.0
    
    for i in prange(s[0]):
        for j in prange(s[1]):
            for k in prange(s[2]):
                total += arr[i, j, k]**2

    return np.sqrt(total)

@njit(parallel=True)
def fast_norm_parallel_4d(arr):
    s = arr.shape
    total = 0.0
    
    for i in prange(s[0]):
        for j in prange(s[1]):
            for k in prange(s[2]):
                for l in prange(s[3]):
                    total += arr[i, j, k, l]**2

    return np.sqrt(total)