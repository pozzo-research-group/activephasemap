import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from botorch.sampling.stochastic_samplers import StochasticSampler 
from botorch.sampling.normal import SobolQMCNormalSampler 
from botorch.sampling.qmc import NormalQMCEngine
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.acquisition.acquisition import AcquisitionFunction
from activephasemap.models.gp import SingleTaskGP, MultiTaskGP 
from activephasemap.models.dkl import SingleTaskDKL, MultiTaskDKL

def initialize_model(model_name, model_args, input_dim, output_dim, device):
    if model_name == 'gp':
        if output_dim == 1:
            return SingleTaskGP(model_args, input_dim, output_dim)
        else:
            return MultiTaskGP(model_args, input_dim, output_dim)
    elif model_name == 'dkl':
        if output_dim == 1:
            return SingleTaskDKL(model_args, input_dim, output_dim, device)
        else:
            return MultiTaskDKL(model_args, input_dim, output_dim, device)
    else:
        raise NotImplementedError("Model type %s does not exist" % model_name)


def initialize_points(bounds, n_init_points, output_dim, device):
    if n_init_points < 1:
        init_x = torch.zeros(1, 1).to(device)
    else:
        bounds = bounds.to(device, dtype=torch.double)
        init_x = draw_sobol_samples(bounds=bounds, n=n_init_points, q=1).squeeze(-2)

    return init_x

def construct_acqf_by_model(model, train_x, train_y, num_objectives=1):
    dim = train_y.shape[1]
    sampler = StochasticSampler(sample_shape=torch.Size([256]))
    if num_objectives==1:
        acqf = qUpperConfidenceBound(model=model, beta=100, sampler=sampler)
    else:
        weights = torch.ones(dim)/dim
        posterior_transform = ScalarizedPosteriorTransform(weights.to(train_x))
        acqf = qUpperConfidenceBound(model=model, 
        beta=100, 
        sampler=sampler,
        posterior_transform = posterior_transform
        )


    return acqf