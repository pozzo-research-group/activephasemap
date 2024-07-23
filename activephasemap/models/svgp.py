import torch
import gpytorch
import tqdm 
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils import gpt_posterior_settings
from botorch.models.transforms.outcome import Standardize
from .gp import GPModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import TensorDataset, DataLoader
import numpy as np 

import pdb

class MultitaskSVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, inducing_points, num_tasks):
        num_latents = inducing_points.size(0)
        self.batch_shape = torch.Size([num_latents])

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=self.batch_shape
        )
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=self.batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                    nu=2.5,
                    ard_num_dims=input_dim,
                    batch_shape=self.batch_shape,
                    lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
                ),
                batch_shape=self.batch_shape,
                outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class MultiTaskSVGP(gpytorch.models.gp.GP):
    def __init__(self, x, y, **kwargs):
        super().__init__()
        self.x = x 
        self.y = y

        self.num_latents = kwargs.get("num_latents", int(self.y.size(-1)-1))
        self.num_inducing_points = kwargs.get("num_inducing_points", int(0.1*self.x.size(0)))
        self.learning_rate = kwargs.get("learning_rate",0.01)
        self.num_epochs = kwargs.get("num_epochs", 16)
        self.debug = kwargs.get("debug", False)
        self.verbose = kwargs.get("verbose", 1)
        self.output_dim = y.shape[1]
        self.input_dim = x.shape[1]
        self.outcome_transform = Standardize(self.output_dim)
        self.y,_ = self.outcome_transform(self.y)
        u = self.get_inducing_points()
        self.gp = MultitaskSVGPModel(self.input_dim, u, self.output_dim).to(device)
        noise_prior = gpytorch.priors.GammaPrior(1.1, 0.05)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        noise_constraint = gpytorch.constraints.GreaterThan(1e-4,transform=None,initial_value=noise_prior_mode)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.output_dim,
                                                                           noise_prior=noise_prior,
                                                                           noise_constraint=noise_constraint).to(device)
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, 
                                                 self.gp, 
                                                 num_data=self.y.size(0)
                                                 )

    def get_covaraince(self, x, xp):
        cov = self.gp.covar_module(x, xp).to_dense()
        cov_mean_tasks = cov.squeeze().mT.mean(dim=1)
        K = cov_mean_tasks.cpu().numpy().squeeze()

        return K
    
    def get_inducing_points(self):
        out = []
        for _ in range(self.num_latents-1):
            rid = torch.randperm(self.x.size(0))[:self.num_inducing_points]
            u = self.x[rid,...]
            out.append(u)
            
        out = torch.stack(out)

        return out.to(device)

    def _setup_train_eval_data(self):
        train_ind = np.random.randint(0, len(self.x), int(0.95*len(self.x)))
        test_ind = np.setdiff1d(np.arange(len(self.x)), train_ind)
        self.train_x, self.train_y = self.x[train_ind,:], self.y[train_ind,:]
        self.test_x, self.test_y = self.x[test_ind,:], self.y[test_ind,:]
        train_dataset = TensorDataset(self.train_x, self.train_y)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        test_dataset = TensorDataset(self.train_x, self.train_y)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        return train_loader, test_loader

    def fit(self):
        optimizer = torch.optim.Adam([{'params': self.gp.parameters()},
                                      {'params': self.likelihood.parameters()},], 
                                      lr=self.learning_rate
                                      )
        self.gp.train()
        train_loss = []
        train_loader, eval_loader = self._setup_train_eval_data()
        for epoch in range(self.num_epochs):
            epoch_loss = []
            for xb, yb in train_loader:
                optimizer.zero_grad()
                with gpytorch.settings.cholesky_jitter(1e-1):
                    output = self.gp(xb)
                    loss = -self.mll(output, yb)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())   

            train_loss.append(sum(epoch_loss)/len(epoch_loss))
            if ((epoch) % self.verbose == 0) or (epoch==self.num_epochs-1):
                print(
                    f"Epoch {epoch+1:>3}/{self.num_epochs} - Loss: {train_loss[-1]:>4.3f} ", end=""
                )
                with torch.no_grad():
                    eval_loss = []
                    for xb, yb in eval_loader:
                        output = self.gp(xb)
                        eval_loss.append(-self.mll(output, yb).item())
                    print(f" Evaluation Loss: {sum(eval_loss)/len(eval_loss):>4.3f} ")
                if self.debug:
                    self.print_hyperparams() 

        return train_loss

    def print_hyperparams(self):
        for name, param in self.gp.named_parameters():
            print(f"{name:>3} : value: {param.data}")

        return  
    def posterior(self, x, **kwargs):
        self.gp.eval()
        self.likelihood.eval()
        with gpytorch.settings.fast_pred_var():
            mvn = self.gp(x)
            mvn = self.likelihood(mvn)

        posterior = GPyTorchPosterior(distribution=mvn)
        posterior = self.outcome_transform.untransform_posterior(posterior)
        posterior_tranform = kwargs.get("posterior_tranform", None)
        if  posterior_tranform is not None:
            posterior = posterior_tranform(posterior)

        return posterior

