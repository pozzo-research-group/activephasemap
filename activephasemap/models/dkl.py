from typing import Any, Callable, List, Optional

import gpytorch
import torch
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import (Log, OutcomeTransform,
                                               Standardize)
from botorch.models.utils import validate_input_scaling
from botorch.posteriors import Posterior
from gpytorch.constraints import GreaterThan
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import Kernel, MaternKernel, RBFKernel, ScaleKernel

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.means.mean import Mean
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.models.exact_gp import ExactGP
from gpytorch.module import Module
from gpytorch.priors import GammaPrior
from torch import Tensor
from torch.nn import Module 

import numpy as np
import pdb
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RegNet(torch.nn.Sequential):
    def __init__(self, dimensions, activation, input_dim=1, output_dim=1,
                        dtype=torch.float64, device="cpu"):
        super(RegNet, self).__init__()
        self.dimensions = [input_dim, *dimensions, output_dim]
        for i in range(len(self.dimensions) - 2):
            self.add_module('linear%d' % i, torch.nn.Linear(
                self.dimensions[i], self.dimensions[i + 1], dtype=dtype, device=device)
            )
            if i < len(self.dimensions) - 2:
                if activation == "tanh":
                    self.add_module('tanh%d' % i, torch.nn.Tanh())
                elif activation == "relu":
                    self.add_module('relu%d' % i, torch.nn.ReLU(inplace=True))
                else:
                    raise NotImplementedError("Activation type %s is not supported" % activation)
        self.add_module('last', torch.nn.Linear(
            self.dimensions[-2], self.dimensions[-1], dtype=dtype, device=device)
        )


class DKLGP(BatchedMultiOutputGPyTorchModel, ExactGP):

    def __init__(
        self,
        model_args,
        train_X: Tensor,
        train_Y: Tensor,
        likelihood: Optional[Likelihood] = None,
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        ignore_X_dims = getattr(self, "_ignore_X_dims_scaling_check", None)
        validate_input_scaling(
            train_X=transformed_X, train_Y=train_Y, ignore_X_dims=ignore_X_dims
        )
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, _ = self._transform_tensor_args(X=train_X, Y=train_Y)

        noise_prior = GammaPrior(1.1, 0.05)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            batch_shape=self._aug_batch_shape,
            noise_constraint=GreaterThan(
                1e-4,
                transform=None,
                initial_value=noise_prior_mode,
            ),
        )
        ExactGP.__init__(
            self, train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )
        self.mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        self.covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=model_args["regnet_dims"][-1],
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                batch_shape=self._aug_batch_shape,
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
        self._subset_batch_dict = {
            "likelihood.noise_covar.raw_noise": -2,
            "mean_module.raw_constant": -1,
            "covar_module.raw_outputscale": -1,
            "covar_module.base_kernel.raw_lengthscale": -3,
        }
        # TODO: Allow subsetting of other covar modules
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.to(train_X)

        # initialize + pretrain feature extractor
        self.feature_extractor = RegNet(dimensions=model_args["regnet_dims"],
                                        activation=model_args["regnet_activation"],
                                        input_dim=transformed_X.size(-1),
                                        dtype=torch.float64,
                                        device=train_X.device)
        self.feature_extractor.last = torch.nn.Identity()

    def forward(self, x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return MultivariateNormal(mean_x, covar_x)

class MultiTaskDKLGP(gpytorch.Module):
    def __init__(self, x, y, model_args, input_dim, output_dim, y_std=None):
        super().__init__()
        self.learning_rate = model_args["learning_rate"] if "learning_rate" in model_args else 1e-3
        self.model_args = model_args
        if y_std is not None:
            y = torch.cat((y, y_std), dim=1)
            self.output_dim = 2*output_dim
        else:
            self.output_dim = output_dim

        train_ind = np.random.randint(0, len(x), int(0.8*len(x)))
        test_ind = np.setdiff1d(np.arange(len(x)), train_ind)
        self.train_x, self.train_y = x[train_ind,:], y[train_ind,:]
        self.test_x, self.test_y = x[test_ind,:], y[test_ind,:]
        models = []
        self.params = []
        for d in range(self.output_dim):
            model = DKLGP(
                    self.model_args,
                    self.train_x,
                    self.train_y[:, d].unsqueeze(-1),
                    outcome_transform=Standardize(m=1)
                ).to(device)
            models.append(model)

            self.params.append({"params": model.feature_extractor.parameters()})
            self.params.append({"params": model.covar_module.parameters()})
            self.params.append({"params": model.mean_module.parameters()})
            self.params.append({"params": model.likelihood.parameters()})

        self.gp = ModelListGP(*models)
        self.mll = SumMarginalLogLikelihood(self.gp.likelihood, self.gp).to(device)

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
    
    def fit(self):
        n_epochs = self.model_args["train_steps"]
        optimizer = torch.optim.Adam(self.params, lr=self.learning_rate)

        self.gp.train()
        train_loss_trace = []
        for epoch in range(n_epochs):
            self.gp.train()
            optimizer.zero_grad()
            output = self.gp(*[self.train_x for _ in range(self.output_dim)])
            loss = -self.mll(output, self.train_y.transpose(-1, -2))
            loss.backward()
            train_loss_trace.append(loss.item())
            # print every 100 iterations
            if np.remainder(100*(epoch+1)/n_epochs,10)==0:
                print(
                    f"Epoch {epoch+1:>3}/{n_epochs} - Train Loss: {loss.item():>4.3f} ", end=""
                )
            optimizer.step()
            
            if np.remainder(100*(epoch+1)/n_epochs,10)==0:
                self.gp.eval()
                with torch.no_grad():
                    output = self.gp(*[self.test_x for _ in range(self.output_dim)])
                    test_loss = -self.mll(output, self.test_y.transpose(-1, -2)) 
                    print(f" Test Loss: {test_loss.item():>4.3f} ")

        return train_loss_trace

    def get_covaraince(self, x, xp):
        cov = torch.zeros((1,len(xp))).to(xp)
        for m in self.gp.models:    
            proj_x = m.feature_extractor(x)
            proj_xp = m.feature_extractor(xp)  
            cov += m.covar_module(proj_x, proj_xp).to_dense()
        K = cov.mean(axis=0).cpu().numpy().squeeze()

        return K