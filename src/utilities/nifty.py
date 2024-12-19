
from sirf.Reg import NiftyResample
from cil.optimisation.operators import LinearOperator

class NiftyResampleOperator(LinearOperator):

    def __init__(self, ref, float, transform):

        self.ref = ref.get_uniform_copy(0)
        self.float = float.get_uniform_copy(0)
        self.transform = transform

        self.resampler = NiftyResample()
        self.resampler.set_reference_image(ref)
        self.resampler.set_floating_image(float)
        self.resampler.set_interpolation_type_to_linear()
        self.resampler.set_padding_value(0)
        self.resampler.add_transformation(self.transform)

    def direct(self, x, out=None):
        res = self.resampler.forward(x)
        if out is not None:
            out.fill(res)
        return res
    
    def adjoint(self, x, out=None):
        res = self.resampler.backward(x)
        if out is not None:
            out.fill(res)
        return res
    
    def domain_geometry(self):
        return self.float
    
    def range_geometry(self):
        return self.ref