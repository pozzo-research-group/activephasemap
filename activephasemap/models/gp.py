from typing import Any, Callable, List, Optional, Union
import abc, pdb

import torch
from torch import Tensor

import botorch
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.multitask import MultiTaskGP as MultiTaskGpBoTorch
from botorch.models.gpytorch import MultiTaskGPyTorchModel
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils import gpt_posterior_settings
from botorch.models.utils.gpytorch_modules import get_matern_kernel_with_gamma_prior

import gpytorch
from gpytorch.models.gp import GP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GPModel(GP):
    def __init__(self, model_args, input_dim, output_dim):
        super().__init__()
        self.gp = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nu = model_args["nu"] if "nu" in model_args else 2.5 
        self.num_epochs = model_args["num_epochs"] if "num_epochs" in model_args else 100 
        self.learning_rate = model_args["learning_rate"] if "learning_rate" in model_args else 3e-4
        self.verbose =  model_args["verbose"] if "verbose" in model_args else 1
        self.debug = model_args["debug"] if "debug" in model_args else False

    def fit(self):
        optimizer = torch.optim.Adam(self.gp.parameters(), lr=self.learning_rate)
        self.gp.train()
        train_loss = []
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            with gpytorch.settings.cholesky_jitter(1e-1):
                output = self.gp(*self.gp.train_inputs)
                loss = -self.mll(output, self.gp.train_targets)
            loss.backward()
            train_loss.append(loss.item())
            if ((epoch) % self.verbose == 0) or (epoch==self.num_epochs-1):
                print(
                    f"Epoch {epoch+1:>3}/{self.num_epochs} - Loss: {loss.item():>4.3f} "
                )
                if self.debug:
                    self.print_hyperparams()
            optimizer.step()    

        return train_loss

    def print_hyperparams(self):
        for name, param in self.gp.named_parameters():
            print(f"{name:>3} : value: {param.data}")

        return       

    def posterior(self, X, **kwargs):
        return self.gp.posterior(X, **kwargs) 

    @property
    def batch_shape(self) -> torch.Size:
        return self.gp.batch_shape

    @property
    def num_outputs(self) -> int:
        return self.gp.num_outputs
    
    @abc.abstractmethod
    def get_covaraince(self, x, xp):
        pass

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, num_tasks, likelihood):
        outcome_transform = Standardize(num_tasks)
        train_y,_ = outcome_transform(train_y)
        super().__init__(train_x, train_y, likelihood)
        self.outcome_transform = outcome_transform
        self.num_outputs = num_tasks
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        base_kernel = get_matern_kernel_with_gamma_prior(ard_num_dims=train_x.shape[-1])
        base_covar_module = gpytorch.kernels.MultitaskKernel(base_kernel, num_tasks=num_tasks, rank=2)

        n_devices = torch.cuda.device_count()
        if n_devices>1:
            print('Planning to run on GP-fiiting on {} GPUs.'.format(n_devices))
            self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                base_covar_module, device_ids=range(n_devices),
                output_device=device
            )
        else:
            self.covar_module = base_covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    def posterior(self, X, **kwargs):
        self.eval()  # make sure model is in eval mode
        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        with gpt_posterior_settings():
            mvn = self(X)
            mvn = self.likelihood(mvn)

        posterior = GPyTorchPosterior(distribution=mvn)
        posterior = self.outcome_transform.untransform_posterior(posterior)
        posterior_tranform = kwargs.get("posterior_tranform", None)
        if  posterior_tranform is not None:
            posterior = posterior_tranform(posterior)

        return posterior

class MultiTaskGP(GPModel):
    def __init__(self, train_x, train_y, model_args, input_dim, output_dim, train_y_std=None):
        super().__init__(model_args, input_dim, output_dim)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim, has_global_noise=False).to(device)
        self.gp = MultitaskGPModel(train_x, train_y, output_dim, likelihood).to(device)
        self.mll = ExactMarginalLogLikelihood(likelihood, self.gp).to(device)

    def get_covaraince(self, x, xp):
        cov = self.gp.covar_module.data_covar_module(x, xp).to_dense()
        K = cov.cpu().numpy().squeeze()

        return K
