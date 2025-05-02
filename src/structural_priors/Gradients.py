import numpy as np
from numba import njit, prange, jit
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class Jacobian(Operator):
    """ Jacobian operation with optional weighting """
    def __init__(self, voxel_sizes=(1, 1, 1),
                 bnd_cond='Neumann', method='forward',
                 anatomical=None, numpy_out=True, gpu=False) -> None:
        
        self.gpu = gpu
        
        self.voxel_sizes = voxel_sizes
        self.method = method
        
        self.numpy_out = numpy_out
        self.anatomical = anatomical
        # check if anatomical is a list
        if isinstance(anatomical, list):
            self.grad = [self._initialise_gradient(anatomical, voxel_sizes, method, bnd_cond, gpu) for anatomical in self.anatomical]
            print('Multiple anatomical images detected. This will fail if different number of images are passed to direct and adjoint methods')
        else:
            self.grad = self._initialise_gradient(anatomical, voxel_sizes, method, bnd_cond, gpu)

    def _initialise_gradient(self, anatomical, voxel_sizes, method, bnd_cond, gpu):

        if anatomical is None:
            return Gradient(voxel_sizes=voxel_sizes, method=method, bnd_cond=bnd_cond, gpu=gpu, numpy_out=False)
        
        if gpu:
            anatomical = torch.tensor(anatomical, device=device) if isinstance(anatomical, np.ndarray) else anatomical.to(device)
       
        return DirectionalGradient(anatomical, voxel_sizes=voxel_sizes, method=method, bnd_cond=bnd_cond, gpu=gpu, numpy_out=False)
    
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
        Return [√f_x/v_x, √f_y/v_y, √f_z/v_z]   with f_k = 2 for fwd/bwd,
        f_k = 1 for central.  Keeps geometry *and* stencil together.
        """
        fx, fy, fz = (2.0, 2.0, 2.0) if self.method != 'central' else (1.0, 1.0, 1.0)
        vs = self.voxel_sizes
        if self.gpu:
            S = torch.ones((*images.shape, 3), device=images.device)
            S[..., 0].mul_((fx**0.5)/vs[0])
            S[..., 1].mul_((fy**0.5)/vs[1])
            S[..., 2].mul_((fz**0.5)/vs[2])
            return S
        else:
            S = np.ones((*images.shape, 3))
            S[..., 0] *= (fx**0.5)/vs[0]
            S[..., 1] *= (fy**0.5)/vs[1]
            S[..., 2] *= (fz**0.5)/vs[2]
            return S
        
    # --- Jacobian.calculate_norm ---------------------------------------
    def calculate_norm(self):
        """‖J‖₂ for the chosen stencil."""
        if not hasattr(self, '_norm'):
            fx, fy, fz = (2.0, 2.0, 2.0) if self.method != 'central' else (1.0, 1.0, 1.0)
            vx, vy, vz = self.voxel_sizes
            self._norm = 2.0 * np.sqrt(fx/vx**2 + fy/vy**2 + fz/vz**2)
        return self._norm

       
       
class Gradient(Operator):
    def __init__(self, voxel_sizes, method='forward', bnd_cond='Neumann', 
                 numpy_out=False, gpu=False):
        self.voxel_sizes = voxel_sizes
        self.method = method
        self.bnd_cond = bnd_cond
        self.numpy_out = numpy_out
        self.gpu = gpu

        if not gpu:
            self.FD = CPUFiniteDifferenceOperator(self.voxel_sizes[0], direction=0, method=self.method, bnd_cond=bnd_cond)

    def direct(self, x):
        res = []
        if self.gpu:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, device=device)
            else:
                x = x.to(device)
            for i in range(x.ndim):
                if self.method == 'forward':
                    res.append(self.forward_diff(x, i))
                elif self.method == 'backward':
                    res.append(self.backward_diff(x, i))
                elif self.method == 'central':
                    res.append((self.forward_diff(x, i) + self.backward_diff(x, i)) / 2)
                else:
                    raise ValueError('Not implemented')
                if self.voxel_sizes[i] != 1.0:
                    res[-1] /= self.voxel_sizes[i]
            result = torch.stack(res, dim=-1)
            return result.cpu().numpy() if self.numpy_out else result
        else:
            for i in range(x.ndim):
                self.FD.direction = i
                self.FD.voxel_size = self.voxel_sizes[i]
                res.append(self.FD.direct(x))
            return np.stack(res, axis=-1)

    def adjoint(self, x):
        res = []
        if self.gpu:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, device=device)
            else:
                x = x.to(device)
            for i in range(x.size(-1)):
                if self.method == 'forward':
                    res.append(-self.backward_diff(x[..., i], i))
                elif self.method == 'backward':
                    res.append(-self.forward_diff(x[..., i], i))
                elif self.method == 'central':
                    res.append((-self.forward_diff(x[..., i], i) - self.backward_diff(x[..., i], i)) / 2)
                else:
                    raise ValueError('Not implemented')
                if self.voxel_sizes[i] != 1.0:
                    res[-1] /= self.voxel_sizes[i]
            result = torch.stack(res, dim=-1).sum(dim=-1)
            return result.cpu().numpy() if self.numpy_out else result
        for i in range(x.shape[-1]):
            self.FD.direction = i
            self.FD.voxel_size = self.voxel_sizes[i]
            res.append(self.FD.adjoint(x[..., i]))
        return -sum(res)

    def forward_diff(self, x, direction):
        append_tensor = x.select(direction, 0 if self.bnd_cond == 'Periodic' else -1).unsqueeze(direction)
        out = torch.diff(x, n=1, dim=direction, append=append_tensor)
        if self.bnd_cond == 'Neumann':
            out.select(direction, -1).zero_()
        return out

    def backward_diff(self, x, direction):
        flipped_x = x.flip(direction)
        append_tensor = flipped_x.select(direction, 0 if self.bnd_cond == 'Periodic' else -1).unsqueeze(direction)
        out = -torch.diff(flipped_x, n=1, dim=direction, append=append_tensor).flip(direction)
        if self.bnd_cond == 'Neumann':
            # Left boundary: Set first slice of out to the first slice of x
            out.select(direction, 0).copy_(x.select(direction, 0))
            # Right boundary: Set last slice of out to the negative of the penultimate slice of x
            out.select(direction, -1).copy_(-x.select(direction, -2))
        return out
    
    def calculate_norm(self):
        """Spectral norm of the 3-D finite-difference gradient."""
        # cache if geometry is immutable
        if not hasattr(self, '_norm') or self._norm is None:
            vs = self.voxel_sizes
            # 2 * sqrt(1/vx^2 + 1/vy^2 + 1/vz^2)
            self._norm = 2.0 * np.sqrt(sum(1.0/(v**2) for v in vs))
        return self._norm
    
class DirectionalGradient(Operator):

    def __init__(self, anatomical, voxel_sizes, gamma=1, eta=1e-6,
                  method='forward', bnd_cond='Neumann', numpy_out=False,
                  gpu=False) -> None:

        self.anatomical = anatomical
        self.voxel_size = voxel_sizes
        self.gamma = gamma
        self.eta = eta
        self.method = method
        self.bnd_cond = bnd_cond
        self.numpy_out = numpy_out
        self.gpu = gpu
        self.gradient = Gradient(voxel_sizes=self.voxel_size, method=self.method, bnd_cond=self.bnd_cond, numpy_out=False, gpu=self.gpu)

        self.anatomical_grad = self.gradient.direct(self.anatomical)

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