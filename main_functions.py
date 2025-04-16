# import method type
from types import MethodType

def compute_kappa_squared_image_from_partitioned_objective(obj_funs, initial_image, normalise=True):
    """
    Computes a "kappa" image for a prior as sqrt(H.1).
    This will attempt to give uniform "perturbation response".
    See Yu-jung Tsai et al. TMI 2020 https://doi.org/10.1109/TMI.2019.2913889

    WARNING: Assumes the objective function has been set-up already.
    """
    out = initial_image.get_uniform_copy(0)
    for obj_fun in obj_funs:
        # need to get the function from the ScaledFunction OperatorCompositionFunction
        out += obj_fun.function.multiply_with_Hessian(initial_image, initial_image.allocate(1))
    out = out.abs()
    # shouldn't really need to do this, but just in case
    out = out.maximum(0)
    # debug printing
    print(f"max: {out.max()}")
    mean = out.sum()/out.size
    print(f"mean: {mean}")
    if normalise:
        out /= mean
    return out

def attach_prior_hessian(prior, epsilon = 0) -> None:
    """Attach an inv_hessian_diag method to the prior function."""

    def inv_hessian_diag(self, x, out=None, epsilon=epsilon):
        ret = self.function.operator.adjoint(
            self.function.function.inv_hessian_diag(
                self.function.operator.direct(x),
                epsilon=epsilon
            )
        )
        if out is not None:
            out.fill(ret)
        return ret

    prior.inv_hessian_diag = MethodType(inv_hessian_diag, prior)