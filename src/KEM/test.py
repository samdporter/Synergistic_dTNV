#%% %matplotlib inline

__version__ = '0.1.0'

import numpy as np
import pylab
from sirf.Utilities import error, examples_data_path, existing_filepath
import importlib

# Instead of using docopt, define the parameters manually
args = {
    '--subs': '12',
    '--subiter': '5',
    '--file': 'my_forward_projection.hs',
    '--anim': 'test_image_PM_QP_6.hv',
    '--path': None,  # if None, we'll use the default examples data path for PET
    '--engine': 'STIR',
    '--non-interactive': False
}

# Import the engine module (e.g. STIR) via SIRF.
engine = args['--engine']
pet = importlib.import_module('sirf.' + engine)

num_subsets = int(args['--subs'])
num_subiterations = int(args['--subiter'])
data_file = args['--file']
data_path = args['--path']
if data_path is None:
    data_path = examples_data_path('PET')
raw_data_file = existing_filepath(data_path, data_file)
if args['--anim'] is not None:
    ai_file = existing_filepath(data_path, args['--anim'])
else:
    ai_file = None
show_plot = not args['--non-interactive']

#%% # Define the functions used in the main reconstruction script
def divide(numerator, denominator, small_num):
    """Division similar to STIR's, element-wise."""
    small_value = np.max(numerator)*small_num
    if small_value <= 0:
        small_value = 0
    X, Y, Z = numerator.shape
    for i in range(Z):
        for j in range(Y):
            for k in range(X):
                if numerator[k, j, i] <= small_value and denominator[k, j, i] <= small_value:
                    numerator[k, j, i] = 0
                else:
                    numerator[k, j, i] /= denominator[k, j, i]
    return numerator

def divide_sino(numerator, denominator, small_num):
    """Division for sinogram data."""
    small_value = np.max(numerator)*small_num
    if small_value <= 0:
        small_value = 0
    X, Y, Z = numerator.shape
    for i in range(Z):
        for j in range(Y):
            for k in range(X):
                if numerator[k, j, i] <= small_value and denominator[k, j, i] <= small_value:
                    numerator[k, j, i] = 0
                else:
                    numerator[k, j, i] /= denominator[k, j, i]
    return numerator

def image_data_processor(image_array, im_num):
    """Process or display an image in Jupyter."""
    if not show_plot:
        return image_array
    pylab.figure(im_num)
    pylab.title('Image estimate %d' % im_num)
    pylab.imshow(image_array[20,:,:])
    print('You may need to close Figure %d window to continue' % im_num)
    return image_array

cyl = pet.TruncateToCylinderProcessor()

#%% Redirect engine messages to files
_ = pet.MessageRedirector('info.txt', 'warn.txt')

# Set up the acquisition model
acq_model = pet.AcquisitionModelUsingRayTracingMatrix()

print('raw data: %s' % raw_data_file)
acq_data = pet.AcquisitionData(raw_data_file)

# Read the anatomical image
anatomical_image = pet.ImageData(ai_file)
if show_plot:
    anatomical_image.show(title='Anatomical Prior')
image_array = anatomical_image.as_array()
image_array[image_array < 0] = 0
anatomical_image.fill(image_array)

#%% let's have some fun

from cil.optimisation.operators import LinearOperator
from sirf.STIR import KOSMAPOSLReconstructor, make_Poisson_loglikelihood, AcquisitionModelUsingRayTracingMatrix

class KernelOperator(LinearOperator):

    def __init__(self, 
                    template_image, template_data,
                    anatomical_image,
                    num_neighbours = 5, 
                    num_non_zero_features = 1,
                    sigma_m = 2.0,
                    sigma_p = 3.0,
                    sigma_dm = 5.0,
                    sigma_dp = 5.0,
                    only_2D = True,
                    hybrid = False):
        
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
        
        del tmp_acq_model, tmp_obj_fun, template_data, template_image

        
    def direct(self, x, out=None):
        if out is None:
            out = x.copy()
        image_update = self.recon.compute_kernelised_image(x, x)
        out.fill(image_update)
        return out
    
    def adjoint(self, x, out=None):
        # self adjoint, innit
        return self.direct(x, out)
    
#%%

# Create initial image estimate and related images
image = anatomical_image.get_uniform_copy()
current_alpha1 = anatomical_image.get_uniform_copy(1)
current_alpha2 = anatomical_image.get_uniform_copy(1)
iterative_kernel_info1 = current_alpha1.get_uniform_copy(1)
iterative_kernel_info2 = current_alpha2.get_uniform_copy(1)
image_update1 = current_alpha1.get_uniform_copy(1)
image_update2 = current_alpha1.get_uniform_copy(1)
if show_plot:
    image.show(title='Initial Guess')




#%% create a KernelOperator
K = KernelOperator(image, acq_data, anatomical_image)

#%%
test = K.direct(anatomical_image)

# let's have a look
if show_plot:
    test.show(title='Test')

#%%
# Set up the acquisition model with the image estimate
#acq_model.set_up(acq_data, image)

# Define the objective function: Poisson log-likelihood
obj_fun = pet.make_Poisson_loglikelihood(acq_data)
obj_fun.set_acquisition_model(acq_model)

#%%
# Set up the first reconstructor (Kernelized OS MAP OSL)
recon = pet.KOSMAPOSLReconstructor()
recon.set_objective_function(obj_fun)
recon.set_num_subsets(num_subsets)
recon.set_num_subiterations(num_subiterations)
recon.set_input(acq_data)
recon.set_anatomical_prior(anatomical_image)
recon.set_num_neighbours(5)
recon.set_num_non_zero_features(1)
recon.set_sigma_m(2.0)
recon.set_sigma_p(3.0)
recon.set_sigma_dm(5.0)
recon.set_sigma_dp(5.0)
recon.set_only_2D(True)
recon.set_hybrid(False)

#%%

print('Setting up reconstructor, please wait...')
sensitivity = acq_model.backward(acq_data.get_uniform_copy(1))
mult_update = image.get_uniform_copy(1)
diff_im = image.get_uniform_copy()

#%%
recon.set_up(current_alpha1)
#sens1 = obj_fun.get_subset_sensitivity(0)
K = recon.compute_kernelised_image(current_alpha1, iterative_kernel_info1)
#Ksensitivity1 = recon.compute_kernelised_image(sens1, iterative_kernel_info1)
recon.set_current_estimate(current_alpha1)

#%% Run first reconstruction branch
for subiteration in range(num_subiterations * num_subsets):
    print('\n--- Sub-iteration %d ---' % subiteration)
    recon.update_current_estimate()
    current_alpha1 = recon.get_current_estimate()
    image_update1 = recon.compute_kernelised_image(current_alpha1, iterative_kernel_info1)
    iterative_kernel_info1 = current_alpha1
    recon.set_current_estimate(current_alpha1)
    cyl.apply(image_update1)

#%%
# Set up a second reconstructor branch
recon2 = pet.KOSMAPOSLReconstructor()

recon2.set_objective_function(obj_fun)
recon2.set_num_subsets(num_subsets)
recon2.set_num_subiterations(num_subiterations)
recon2.set_input(acq_data)
recon2.set_anatomical_prior(anatomical_image)
recon2.set_num_neighbours(5)
recon2.set_num_non_zero_features(1)
recon2.set_sigma_m(2.0)
recon2.set_sigma_p(3.0)
recon2.set_sigma_dm(5.0)
recon2.set_sigma_dp(5.0)
recon2.set_only_2D(True)
recon2.set_hybrid(False)
recon2.set_up(current_alpha2)
recon2.set_current_estimate(current_alpha2)

for subiteration in range(num_subiterations * num_subsets):
    print('\n--- Sub-iteration %d (branch 2) ---' % subiteration)
    Kalpha = recon2.compute_kernelised_image(current_alpha2, iterative_kernel_info2)
    Ksensitivity = recon2.compute_kernelised_image(sensitivity, iterative_kernel_info2)
    gradient_plus_sensitivity = acq_model.backward(acq_data/acq_model.forward(Kalpha))
    Kgradient_plus_sensitivity = recon2.compute_kernelised_image(gradient_plus_sensitivity, iterative_kernel_info2)
    mult_update_array = divide(Kgradient_plus_sensitivity.as_array(), Ksensitivity.as_array(), 0.000001)
    mult_update.fill(mult_update_array)
    current_alpha2 *= mult_update
    image_update2 = recon.compute_kernelised_image(current_alpha2, iterative_kernel_info2)
    iterative_kernel_info2 = current_alpha2 
    cyl.apply(image_update2)

#%%
mean2 = np.mean(image_update2.as_array())
mean1 = np.mean(image_update1.as_array())
diff_im.fill(np.abs(image_update2.as_array()/mean2 - image_update1.as_array()/mean1))
max_diff = np.max(diff_im.as_array())
mean_diff = np.mean(diff_im.as_array())
max_im = np.max(image_update1.as_array())
rel_diff = (max_diff)/(max_im)*100
print('Max perc difference : %e' % (rel_diff))
if rel_diff < 1:
    print('Max difference is less than 1% so is probably OK')
else:
    print('Max difference is higher than 1% - something could be wrong')

print('Projecting...')
simulated_data = acq_model.forward(image)
diff = simulated_data * (acq_data.norm()/simulated_data.norm()) - acq_data
print('Relative residual norm: %e' % (diff.norm()/acq_data.norm()))

#%% show the output images
if show_plot:
    image.show(title='Reconstructed Image')
    image_update1.show(title='Image Update 1')
    image_update2.show(title='Image Update 2')
    diff_im.show(title='Difference Image')
# %% show final alpha images
if show_plot:
    current_alpha1.show(title='Final Alpha 1')
    current_alpha2.show(title='Final Alpha 2')
# %%
