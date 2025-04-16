
from sirf.Reg import NiftyResample
from cil.optimisation.operators import LinearOperator

class NiftyResampleOperator(LinearOperator):

    def __init__(self, reference, floating, transform):

        self.reference = reference.get_uniform_copy(0)
        self.floating = floating.get_uniform_copy(0)
        self.transform = transform

        self.resampler = NiftyResample()
        self.resampler.set_reference_image(reference)
        self.resampler.set_floating_image(floating)
        self.resampler.set_interpolation_type_to_cubic_spline()
        self.resampler.set_padding_value(0)
        self.resampler.add_transformation(self.transform)

    def direct(self, x, out=None):
        res = self.resampler.forward(x)
        res = res.maximum(0)
        if out is not None:
            out.fill(res)
        return res
    
    def adjoint(self, x, out=None):
        res = self.resampler.backward(x)
        res = res.maximum(0)
        if out is not None:
            out.fill(res)
        return res
    
    def domain_geometry(self):
        return self.floating
    
    def range_geometry(self):
        return self.reference