from abc import ABC, abstractmethod
import torch 
import gpytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
from botorch.utils.transforms import normalize
import pdb, time, datetime

class UncertainitySelector(torch.nn.Module):
    def __init__(self, input_dim, model, bounds):
        super().__init__()
        self.model = model
        self.model.gp.eval()
        self.model.likelihood.eval()
        self.bounds = bounds
        self.input_dim = input_dim

    def forward(self, x):
        if len(x.size())==3:
            x = x.squeeze()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mvn = self.model.posterior(x)
            _, upper = mvn.confidence_region()
            acqf = torch.abs(upper).mean(dim=1)

            return acqf

    def optimize(self, batch_size, num_points_per_dim=50):
        grid = torch.rand(num_points_per_dim**self.input_dim, len(self.bounds)).to(device)
        grid = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * grid
        grid_normalized = normalize(grid, self.bounds)
        acqf_grid = self(grid_normalized)
        ind_top = torch.argwhere(torch.ge(acqf_grid, (0.8*max(acqf_grid)))).squeeze()
        argmax_ind = ind_top[torch.randperm(len(ind_top))[:batch_size]]

        return grid[argmax_ind,:]

class CompositeModelUncertainity(torch.nn.Module):
    def __init__(self, t, bounds, NP, MLP, **kwargs):
        super().__init__()
        NP.eval()
        MLP.eval()
        self.t = t
        self.z_to_y = NP 
        self.c_to_z = MLP 
        self.bounds = bounds
        self.input_dim = len(bounds)
        self.nz = kwargs.get("num_z_sample", 20) 

    def forward(self, x):
        z_mu, z_std = self.c_to_z.mlp(x)
        nr, nb, d = z_mu.shape
        z_dist = torch.distributions.Normal(z_mu, z_std)
        z = z_dist.rsample(torch.Size([self.nz])).reshape(self.nz*nr*nb, d)
        t = torch.from_numpy(self.t).repeat(self.nz*nr*nb, 1, 1).to(device)
        t = torch.swapaxes(t, 1, 2)
        y_samples, _ = self.z_to_y.xz_to_y(t, z)
        sigma_pred = y_samples.reshape(self.nz, nr, nb, len(self.t), 1).std(dim=0)
        acqv = sigma_pred.mean(dim=-2).squeeze(-1)

        return acqv

    def optimize(self, batch_size, num_restarts=8):
        X = torch.rand(num_restarts, batch_size, len(self.bounds)).to(device)
        X = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * X
        X.requires_grad_(True)
        optimizer = torch.optim.Adam([X], lr=0.05)

        start = time.time()
        for i in range(100):
            optimizer.zero_grad()
            acqv = -self(X)
            loss = acqv.sum()
            loss.backward() 
            optimizer.step()

            # clamp values to the feasible set
            for j, (lb, ub) in enumerate(zip(*self.bounds)):
                X.data[..., j].clamp_(lb, ub)  # need to do this on the data not X itself

            if (i + 1) % 15 == 0:
                end = time.time()
                time_str =  str(datetime.timedelta(seconds=end-start))
                print(f"({time_str:>s}) Iteration {i+1:>3}/100 - Loss: {loss.item():>4.3f}")
        
        return X[acqv.sum(dim=1).argmin(),...].detach()