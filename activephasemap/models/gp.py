from typing import Any, Callable, List, Optional
import abc 

import botorch
import torch
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

class GPModel(Model):
    def __init__(self, model_args, input_dim, output_dim):
        self.gp = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nu = model_args["nu"] if "nu" in model_args else 2.5 
        self.num_epochs = model_args["num_epochs"] if "num_epochs" in model_args else 100 
        self.learning_rate = model_args["learning_rate"] if "learning_rate" in model_args else 3e-4
        self.verbose =  model_args["verbose"] if "verbose" in model_args else 1
        super().__init__()

    def fit(self):
        optimizer = torch.optim.Adam(self.gp.parameters(), lr=self.learning_rate)
        self.gp.train()
        train_loss = []
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            output = self.gp(*self.gp.train_inputs)
            loss = -self.mll(output, self.gp.train_targets)
            loss.backward()
            train_loss.append(loss.item())
            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch+1:>3}/{self.num_epochs} - Loss: {loss.item():>4.3f} "
                )
                if self.verbose>1:
                    for name, param in self.gp.named_parameters():
                        print(f"{name:>3} : value: {param.data}")
            optimizer.step()    

        return train_loss

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

    def __init__(self, model_args, input_dim, output_dim):
        super().__init__(model_args, input_dim, output_dim)
        if self.output_dim > 1:
            raise RuntimeError("SingleTaskGP does not fit tasks with multiple objectives")

        self.gp = botorch.models.SingleTaskGP(
            train_x, train_y, outcome_transform=Standardize(m=1)).to(train_x)
        mll = ExactMarginalLogLikelihood(
            self.gp.likelihood, self.gp).to(train_x)

    def get_covaraince(self, x, xp):
        cov = self.gp.covar_module(x, xp).to_dense()
        K = cov.mean(axis=0).cpu().numpy().squeeze()

        return K


class MultiTaskGP(GPModel):

    def __init__(self, model_args, input_dim, output_dim):
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