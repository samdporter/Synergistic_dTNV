import numpy as np
from cil.optimisation.operators import LinearOperator
try:
    import numba
    NUMBA_AVAIL = True
except ImportError:
    NUMBA_AVAIL = False

def get_kernel_operator(domain_geometry, **kwargs):
    if NUMBA_AVAIL and kwargs.get('use_numba', True):
        return NumbaKernelOperator(domain_geometry, **kwargs)
    else:
        return KernelOperator(domain_geometry, **kwargs)

class BaseKernelOperator(LinearOperator):
    def __init__(self, domain_geometry, **kwargs):
        super().__init__(domain_geometry, **kwargs)
        default_parameters = {
            'num_neighbours': 5,
            'sigma_anat': 2.0,
            'sigma_dist': 1.0,      # Standard deviation for Euclidean distance weighting
            'kernel_type': 'neighbourhood',
            'normalize_features': False,
            'normalize_kernel': False,
        }
        self.parameters = {**default_parameters, **kwargs}
        self.anatomical_image = None
        self.current_image = None
        self.current_alpha = None

    def set_anatomical_image(self, image):
        # Optionally normalize anatomical image features.
        if self.parameters['normalize_features']:
            arr = image.as_array()
            std_val = arr.std()
            norm_arr = arr / std_val if std_val > 1e-12 else arr
            temp = image.clone()
            temp.fill(norm_arr)
            self.anatomical_image = temp
        else:
            self.anatomical_image = image

    def get_anatomical_neighbourhood(self, x):
        return self.neighbourhood_kernel(
            x,
            self.anatomical_image,
            self.parameters['num_neighbours'],
            self.parameters['sigma_anat']
        )

    def apply(self, x):
        return self.get_anatomical_neighbourhood(x)

    def direct(self, x, out=None):
        res = self.apply(x)
        if out is None:
            self.current_image = res
            return res
        else:
            out.fill(res.as_array())
            self.current_image = out.clone()
            return out

    def adjoint(self, x, out=None):
        res = self.apply(x)
        if out is None:
            self.current_alpha = res
            return res
        else:
            out.fill(res.as_array())
            self.current_alpha = out.clone()
            return out

    def neighbourhood_kernel(self, x, image, n, sigma):
        # This method is meant to be overridden in subclasses.
        raise NotImplementedError("Subclasses must implement the neighbourhood_kernel method.")

if NUMBA_AVAIL:
    @numba.njit(parallel=True)
    def nb_voxel_neighbourhood_kernel(x_arr, anat_arr, n, sigma, sigma_distance, normalize_kernel):
        s0, s1, s2 = anat_arr.shape
        out = np.empty_like(anat_arr, dtype=np.float64)
        half = n // 2
        sig2 = 2 * sigma * sigma
        dist_denom = 2 * sigma_distance * sigma_distance
        for i in numba.prange(s0):
            for j in range(s1):
                for k in range(s2):
                    i0 = max(i - half, 0)
                    i1 = min(i + half + 1, s0)
                    j0 = max(j - half, 0)
                    j1 = min(j + half + 1, s1)
                    k0 = max(k - half, 0)
                    k1 = min(k + half + 1, s2)
                    center_val = anat_arr[i, j, k]
                    sum_val = 0.0
                    weight_sum = 0.0
                    for ii in range(i0, i1):
                        for jj in range(j0, j1):
                            for kk in range(k0, k1):
                                diff = anat_arr[ii, jj, kk] - center_val
                                weight_intensity = np.exp(- (diff * diff) / sig2)
                                d2 = (ii - i)**2 + (jj - j)**2 + (kk - k)**2
                                weight_distance = np.exp(- d2 / dist_denom)
                                combined_weight = weight_intensity * weight_distance
                                sum_val += x_arr[ii, jj, kk] * combined_weight
                                weight_sum += combined_weight
                    if normalize_kernel and weight_sum > 1e-12:
                        sum_val /= weight_sum
                    out[i, j, k] = sum_val
        return out

    class NumbaKernelOperator(BaseKernelOperator):
        def neighbourhood_kernel(self, x, image, n, sigma):
            out = image.clone()
            anat_arr = image.as_array()
            x_arr = x.as_array()
            sigma_distance = self.parameters.get('sigma_dist', 1.0)
            res = nb_voxel_neighbourhood_kernel(x_arr, anat_arr, n, sigma,
                                                sigma_distance, self.parameters['normalize_kernel'])
            out.fill(res)
            return out

class KernelOperator(BaseKernelOperator):
    def voxel_neighbourhood_kernel(self, x_arr, anat_arr, cx, cy, cz, n, sigma, sigma_distance):
        half = n // 2
        s0, s1, s2 = anat_arr.shape
        i0, i1 = max(cx - half, 0), min(cx + half + 1, s0)
        j0, j1 = max(cy - half, 0), min(cy + half + 1, s1)
        k0, k1 = max(cz - half, 0), min(cz + half + 1, s2)
        neighborhood = anat_arr[i0:i1, j0:j1, k0:k1]
        x_neighborhood = x_arr[i0:i1, j0:j1, k0:k1]
        center_val = anat_arr[cx, cy, cz]
        I = np.arange(i0, i1) - cx
        J = np.arange(j0, j1) - cy
        K = np.arange(k0, k1) - cz
        D2 = I[:, None, None]**2 + J[None, :, None]**2 + K[None, None, :]**2
        weights = np.exp(-((neighborhood - center_val) ** 2) / (2 * sigma * sigma))
        weights *= np.exp(-D2 / (2 * sigma_distance * sigma_distance))
        if self.parameters['normalize_kernel']:
            denom = weights.sum()
            if denom > 1e-12:
                weights /= denom
        return np.sum(x_neighborhood * weights)
    
    def neighbourhood_kernel(self, x, image, n, sigma):
        out = image.clone()
        arr = image.as_array()
        x_arr = x.as_array()
        res = np.empty_like(arr, dtype=np.float64)
        sigma_distance = self.parameters.get('sigma_dist', 1.0)
        for idx in np.ndindex(arr.shape):
            res[idx] = self.voxel_neighbourhood_kernel(x_arr, arr, idx[0], idx[1], idx[2],
                                                       n, sigma, sigma_distance)
        out.fill(res)
        return out