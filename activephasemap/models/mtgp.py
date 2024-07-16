import torch
import gpytorch
import botorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pdb

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        num_tasks = train_y.shape[1]
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), 
                                                        num_tasks=num_tasks
                                                        )
        base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(
                                                   nu=2.5, 
                                                   ard_num_dims=train_x.shape[-1],
                                                   ),
                                                   )
        self.covar_module = gpytorch.kernels.MultitaskKernel(base_kernel, num_tasks=num_tasks, rank=num_tasks)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class MultiTaskGPVersion2(gpytorch.models.gp.GP):
    def __init__(self, x, y, **kwargs):
        super().__init__()
        self.x = x 
        self.y = y
        self.learning_rate = kwargs.get("learning_rate",0.01)
        self.num_epochs = kwargs.get("num_epochs", 16)
        self.debug = kwargs.get("debug", False)
        self.verbose = kwargs.get("verbose", 1)
        self.output_dim = y.shape[1]
        self.input_dim = x.shape[1]
        self.outcome_transform = botorch.models.transforms.outcome.Standardize(self.output_dim)
        train_y,_ = self.outcome_transform(self.y)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.output_dim).to(device)
        self.gp = MultitaskGPModel(self.x, 
                                   train_y, 
                                   self.likelihood
                                   ).to(device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp).to(device)

    def get_covaraince(self, x, xp):
        cov = self.gp.covar_module.data_covar_module(x, xp).to_dense()
        K = cov.cpu().numpy().squeeze()

        return K

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

    def posterior(self, x):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            self.gp.eval()
            self.likelihood.eval()
            mvn = self.gp(x)
            mvn = self.likelihood(mvn)
            posterior = botorch.posteriors.gpytorch.GPyTorchPosterior(mvn)
            posterior = self.outcome_transform.untransform_posterior(posterior)

            return posterior
