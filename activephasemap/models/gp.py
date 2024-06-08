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
from botorch.posteriors import Posterior
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.objective import PosteriorTransform

import gpytorch
from gpytorch.models.gp import GP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal


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
            if (epoch) % self.verbose == 0:
                print(
                    f"Epoch {epoch+1:>3}/{self.num_epochs} - Loss: {loss.item():>4.3f} "
                )
                if self.verbose>10000:
                    for name, param in self.gp.named_parameters():
                        print(f"{name:>3} : value: {param.data}")
            optimizer.step()    

        return train_loss

    def fit_botorch_style(self):
        fit_gpytorch_mll(self.mll)

        return 

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        
        return self.gp.posterior(X, output_indices, observation_noise, posterior_transform, **kwargs)

    @property
    def batch_shape(self) -> torch.Size:
        return self.gp.batch_shape

    @property
    def num_outputs(self) -> int:
        return self.gp.num_outputs
    
    @abc.abstractmethod
    def get_covaraince(self, x, xp):
        pass

class SingleTaskGP(GPModel):

    def __init__(self, train_x, train_y, model_args, input_dim, output_dim):
        super().__init__(model_args, input_dim, output_dim)
        if self.output_dim > 1:
            raise RuntimeError("SingleTaskGP does not fit tasks with multiple objectives")

        self.gp = botorch.models.SingleTaskGP(
            train_x, train_y, outcome_transform=Standardize(m=1)).to(train_x)
        self.mll = ExactMarginalLogLikelihood(
            self.gp.likelihood, self.gp).to(train_x)

    def get_covaraince(self, x, xp):
        cov = self.gp.covar_module(x, xp).to_dense()
        K = cov.mean(axis=0).cpu().numpy().squeeze()

        return K

class MultiTaskListGP(GPModel):

    def __init__(self, train_x, train_y, model_args, input_dim, output_dim):
        super().__init__(model_args, input_dim, output_dim)
        models = []
        for d in range(self.output_dim):
            models.append(
                botorch.models.SingleTaskGP(
                    train_x,
                    train_y[:, d].unsqueeze(-1),
                    outcome_transform=Standardize(m=1)).to(train_x))

        self.gp = ModelListGP(*models)
        self.mll = SumMarginalLogLikelihood(self.gp.likelihood, self.gp).to(train_x)
         
    def get_covaraince(self, x, xp):  
        cov = torch.zeros((1,len(xp))).to(xp)
        for m in self.gp.models:        
            cov += m.covar_module(x, xp).to_dense()
        K = cov.mean(axis=0).cpu().numpy().squeeze()

        return K

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, num_tasks, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.num_outputs = num_tasks
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    def posterior(self, X, output_indices = None, observation_noise = False,posterior_transform = None):
        self.eval()  # make sure model is in eval mode
        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        X = self.transform_inputs(X)
        with gpt_posterior_settings():
            mvn = self(X)

        posterior = GPyTorchPosterior(distribution=mvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        if posterior_transform is not None:
            return posterior_transform(posterior)

        return posterior

    def transform_inputs(
        self,
        X: Tensor,
        input_transform: Optional[torch.nn.Module] = None,
    ) -> Tensor:
        r"""Transform inputs.

        Args:
            X: A tensor of inputs
            input_transform: A Module that performs the input transformation.

        Returns:
            A tensor of transformed inputs
        """
        if input_transform is not None:
            input_transform.to(X)
            return input_transform(X)
        try:
            return self.input_transform(X)
        except AttributeError:
            return X

class MultiTaskGP(GPModel):
    def __init__(self, train_x, train_y, model_args, input_dim, output_dim):
        super().__init__(model_args, input_dim, output_dim)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim)
        self.gp = MultitaskGPModel(train_x, train_y, output_dim, likelihood)
        self.mll = ExactMarginalLogLikelihood(likelihood, self.gp).to(train_x)
