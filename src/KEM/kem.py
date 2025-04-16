from cil.optimisation.operators import LinearOperator
from cil.optimisation.utilities import Preconditioner
from sirf.STIR import (KOSMAPOSLReconstructor, 
                       make_Poisson_loglikelihood, 
                       AcquisitionModelUsingRayTracingMatrix)

class KernelOperator(LinearOperator):

    def __init__(self, 
                    template_image, template_data, 
                    anatomical_image,
                    num_neighbours = 5, 
                    num_non_zero_features = 1,
                    sigma_m = 2.0, # anatomical
                    sigma_p = 3.0, # functional
                    sigma_dm = 5.0,
                    sigma_dp = 5.0,
                    only_2D = False,
                    hybrid = True):
        
        tmp_acq_model = AcquisitionModelUsingRayTracingMatrix()
        tmp_obj_fun = make_Poisson_loglikelihood(template_data)
        tmp_obj_fun.set_acquisition_model(tmp_acq_model)

        self.recon = KOSMAPOSLReconstructor()
        self.recon.set_objective_function(tmp_obj_fun)
        self.recon.set_num_neighbours(num_neighbours)
        self.recon.set_num_non_zero_features(num_non_zero_features)
        self.recon.set_sigma_m(sigma_m)
        self.recon.set_sigma_p(sigma_p)
        self.recon.set_sigma_dm(sigma_dm)
        self.recon.set_sigma_dp(sigma_dp)
        self.recon.set_only_2D(only_2D)
        self.recon.set_hybrid(hybrid)
        self.recon.set_anatomical_prior(anatomical_image)
        self.recon.set_input(template_data)
        self.recon.set_up(template_image)

        super().__init__(domain_geometry=template_image, 
                      range_geometry=template_image)
        
        self.current_alpha = None
        self.freeze_alpha = False
        
        del tmp_acq_model, tmp_obj_fun

    def get_alpha(self, x):
        if self.freeze_alpha:
            return self.current_alpha
        else:
            return x
        
    def direct(self, x, out=None):

        if out is None:
            out = x.copy()

        self.current_alpha = self.get_alpha(x)
        out.fill(
            self.recon.compute_kernelised_image(
                x, self.current_alpha
            )
        )
        return out
    
    def adjoint(self, x, out=None):

        if self.current_alpha is None:
            raise ValueError("No current alpha value set.")
        
        if out is None:
            out = x.copy()

        out.fill(
            self.recon.compute_kernelised_image(
                x, self.current_alpha
            )
        )
        return out