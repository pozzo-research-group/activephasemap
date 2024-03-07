from typing import Any, Callable, List, Optional

import botorch
import torch
from botorch.fit import fit_gpytorch_mll 
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors import Posterior
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch import Tensor
import pdb

def fit_gp_model(model, mll, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs.pop("lr", 1e-3))
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(*model.train_inputs)
        loss = -mll(output, model.train_targets)
        loss.backward()
        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch+1:>3}/{num_epochs} - Loss: {loss.item():>4.3f} "
            )
            if kwargs.pop("verbose", 1)>1:
                for name, param in model.named_parameters():
                    print(f"{name:>3} : value: {param.data}")
        optimizer.step()

class SingleTaskGP(Model):

    def __init__(self, model_args, input_dim, output_dim):
        super().__init__()
        self.gp = None
        self.output_dim = output_dim
        self.nu = model_args["nu"] if "nu" in model_args else 2.5
        self.num_epochs = model_args["num_epochs"] if "nu" in model_args else 100 

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

    def fit_and_save(self, train_x, train_y):
        if self.output_dim > 1:
            raise RuntimeError(
                "SingleTaskGP does not fit tasks with multiple objectives")

        self.gp = botorch.models.SingleTaskGP(
            train_x, train_y, outcome_transform=Standardize(m=1)).to(train_x)
        mll = ExactMarginalLogLikelihood(
            self.gp.likelihood, self.gp).to(train_x)
        fit_gpytorch_mll_torch(mll)

    def get_covaraince(self, x, xp):
        cov = self.gp.covar_module(x, xp).to_dense()
        K = cov.mean(axis=0).cpu().numpy().squeeze()

        return K


class MultiTaskGP(Model):

    def __init__(self, model_args, input_dim, output_dim):
        super().__init__()
        self.gp = None
        self.output_dim = output_dim
        self.nu = model_args["nu"] if "nu" in model_args else 2.5 
        self.num_epochs = model_args["num_epochs"] if "nu" in model_args else 100 

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
        
    def fit_and_save(self, train_x, train_y):
        models = []
        for d in range(self.output_dim):
            models.append(
                botorch.models.SingleTaskGP(
                    train_x,
                    train_y[:, d].unsqueeze(-1),
                    outcome_transform=Standardize(m=1)).to(train_x))

        self.gp = ModelListGP(*models)
        self.mll = SumMarginalLogLikelihood(self.gp.likelihood, self.gp).to(train_x)
        fit_gp_model(self.gp, self.mll, self.num_epochs) 
         
    def get_covaraince(self, x, xp):  
        cov = torch.zeros((1,len(xp))).to(xp)
        for m in self.gp.models:        
            cov += m.covar_module(x, xp).to_dense()
        K = cov.mean(axis=0).cpu().numpy().squeeze()

        return K