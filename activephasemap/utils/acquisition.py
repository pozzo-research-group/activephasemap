from abc import ABC, abstractmethod
import torch 
import gpytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
from botorch.utils.transforms import normalize
import pdb 

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

    def optimize(self, batch_size, num_points_per_dim=20):
        grid = torch.rand(num_points_per_dim**self.input_dim, len(self.bounds)).to(device)
        grid = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * grid
        grid_normalized = normalize(grid, self.bounds)
        acqf_grid = self(grid_normalized)
        ind_top = torch.argwhere(torch.ge(acqf_grid, (0.8*max(acqf_grid)))).squeeze()
        argmax_ind = ind_top[torch.randperm(len(ind_top))[:batch_size]]

        return grid[argmax_ind,:]


