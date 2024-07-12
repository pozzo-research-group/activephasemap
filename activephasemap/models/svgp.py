import torch
import gpytorch
import tqdm 
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils import gpt_posterior_settings
from .gp import GPModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.utils.data import TensorDataset, DataLoader
import numpy as np 

import pdb

class MultitaskSVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, num_latents, num_tasks):
        self.num_outputs = num_tasks
        inducing_points = torch.rand(num_latents, 16, input_dim).to(device)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
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

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=2.5,
                    ard_num_dims=input_dim,
                    batch_shape=torch.Size([num_latents]),
                    lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0),
                ),
                batch_shape=torch.Size([num_latents]),
                outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15),
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(self, X, output_indices = None, observation_noise = False,posterior_transform = None):
        self.eval()
        X = self.transform_inputs(X)
        with gpt_posterior_settings():
            if len(X.size())==3:
                mvn = self(X.squeeze())
            else:
                mvn = self(X)

        posterior = GPyTorchPosterior(distribution=mvn)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        if posterior_transform is not None:
            return posterior_transform(posterior)

        return posterior

    def transform_inputs(self, X, input_transform=None):
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

class MultiTaskSVGP(GPModel):
    def __init__(self, x, y, model_args, input_dim, output_dim):
        super().__init__(model_args, input_dim, output_dim)
        num_latents = model_args["num_latents"] if "num_latents" in model_args else 5
        self.x = x 
        self.y = y
        self.gp = MultitaskSVGPModel(input_dim, num_latents, output_dim).to(device)
        noise_prior = gpytorch.priors.GammaPrior(1.1, 0.05)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        self.likelihood= gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim,
            rank = 1,
            noise_prior=noise_prior,
            batch_shape=torch.Size([num_latents]),
            noise_constraint=gpytorch.constraints.GreaterThan(
                1e-4,
                transform=None,
                initial_value=noise_prior_mode,
            ),
        ).to(device)
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.gp, num_data=self.y.size(0))

    def get_covaraince(self, x, xp):
        cov = self.gp.covar_module.data_covar_module(x, xp).to_dense()
        K = cov.cpu().numpy().squeeze()

        return K

    def _setup_train_eval_data(self):
        train_ind = np.random.randint(0, len(self.x), int(0.8*len(self.x)))
        test_ind = np.setdiff1d(np.arange(len(self.x)), train_ind)
        self.train_x, self.train_y = self.x[train_ind,:], self.y[train_ind,:]
        self.test_x, self.test_y = self.x[test_ind,:], self.y[test_ind,:]
        train_dataset = TensorDataset(self.train_x, self.train_y)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

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
        for epoch in range(self.num_epochs):
            train_loader, test_loader = self._setup_train_eval_data()
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
                    test_loss = []
                    for xb, yb in test_loader:
                        output = self.gp(xb)
                        test_loss.append(-self.mll(output, yb).item())
                    print(f" Test Loss: {sum(test_loss)/len(test_loss):>4.3f} ")
            if self.debug:
                self.print_hyperparams() 

        return train_loss
