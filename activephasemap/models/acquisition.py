from abc import ABC, abstractmethod
import torch 
import gpytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
from botorch.utils.transforms import normalize
import pdb, time, datetime

class BaseAcquisiton(torch.nn.Module):
    def __init__(self, t, bounds, z2y, c2z, **kwargs):
        super().__init__()
        self.t = t
        self.z_to_y = z2y 
        self.c_to_z = c2z 
        self.bounds = bounds
        self.input_dim = len(bounds)
        self.nz = kwargs.get("num_z_sample", 20) 

    @abstractmethod
    def forward(self, x):
        """
        This method must be implemented by any subclass of BaseClass.
        """
        pass

    def optimize(self, batch_size, num_restarts=8, n_iterations=200):
        X = torch.rand(num_restarts, batch_size, len(self.bounds)).to(device)
        X = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * X
        X.requires_grad_(True)
        optimizer = torch.optim.Adam([X], lr=0.05)

        start = time.time()
        for i in range(n_iterations):
            optimizer.zero_grad()
            acqv = -self(X)
            loss = acqv.sum()
            loss.backward() 
            optimizer.step()

            # clamp values to the feasible set
            for j, (lb, ub) in enumerate(zip(*self.bounds)):
                X.data[..., j].clamp_(lb, ub)  # need to do this on the data not X itself

            if (i + 1) % 50 == 0:
                end = time.time()
                time_str =  str(datetime.timedelta(seconds=end-start))
                print(f"({time_str:>s}) Iteration {i+1:>3}/{n_iterations:>3} - Loss: {loss.item():>4.3f}")
        
        return X[acqv.sum(dim=1).argmin(),...].clone().detach()

class CompositeModelUncertainity(BaseAcquisiton):
    def __init__(self, t, bounds, NP, MLP, **kwargs):
        super().__init__(t, bounds, NP, MLP, **kwargs)

    def forward(self, x):
        z_mu, z_std = self.c_to_z.mlp(x)       
        nr, nb, d = z_mu.shape
        z_dist = torch.distributions.Normal(z_mu, z_std)
        z = z_dist.rsample(torch.Size([self.nz])).view(self.nz*nr*nb, d)
        t = torch.from_numpy(self.t).repeat(self.nz*nr*nb, 1, 1).to(device)
        t = torch.swapaxes(t, 1, 2)
        y_samples, _ = self.z_to_y.xz_to_y(t, z)
        sigma_pred = y_samples.view(self.nz, nr, nb, len(self.t), 1).std(dim=0)
        acqv = sigma_pred.mean(dim=-2).squeeze(-1)

        return acqv

class XGBUncertainity(BaseAcquisiton):
    def __init__(self, t, bounds, NP, XGB, **kwargs):
        super().__init__(t, bounds, NP, XGB, **kwargs)

    def forward(self, x):
        z_mu, z_std = self.c_to_z.predict(x)   
        nr, nb, dz = z_mu.shape
        z_dist = torch.distributions.Normal(z_mu, z_std)
        z = z_dist.rsample(torch.Size([self.nz])).view(self.nz*nr*nb, dz)
        t = torch.from_numpy(self.t).repeat(self.nz*nr*nb, 1, 1).to(device)
        t = torch.swapaxes(t, 1, 2)
        y_samples, _ = self.z_to_y.xz_to_y(t, z)
        sigma_pred = y_samples.view(self.nz, nr, nb, len(self.t), 1).std(dim=0)
        acqv = sigma_pred.mean(dim=-2).squeeze(-1)

        return acqv    