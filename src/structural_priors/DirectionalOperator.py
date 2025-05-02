from cil.optimisation.operators import LinearOperator
from cil.framework import BlockGeometry

class DirectionalOperator(LinearOperator):

    def __init__(self, anatomical_gradient, gamma = 1, eta=1e-6):

        self.anatomical_gradient = anatomical_gradient
        geometry = BlockGeometry(*anatomical_gradient.containers)
        self.tmp = self.anatomical_gradient.containers[0].clone()

        self.gamma = gamma

        self.xi = self.anatomical_gradient/(self.anatomical_gradient.pnorm().power(2)+eta**2).sqrt()

        self.calculate_norm = lambda _: 1

        super(DirectionalOperator, self).__init__(domain_geometry=geometry,
                                       range_geometry=geometry,)

        
    def direct(self, x, out=None):

        if out is None:
            return x - self.gamma * self.xi * self.dot(self.xi, x)
        else:
            out.fill(x - self.gamma * self.xi * self.dot(self.xi, x))
    
    def adjoint(self, x, out=None):
        # This is the same as the direct operator
        return self.direct(x, out)
    
    def dot(self, x, y):
        self.tmp.fill(0)
        for el_x, el_y in zip(x.containers, y.containers):
            self.tmp += el_x * el_y
        return self.tmp