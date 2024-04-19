import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from botorch.sampling.stochastic_samplers import StochasticSampler 
from botorch.sampling.normal import SobolQMCNormalSampler 
from botorch.sampling.qmc import NormalQMCEngine
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.acquisition.acquisition import AcquisitionFunction
from activephasemap.models.gp import SingleTaskGP, MultiTaskGP 
from activephasemap.models.dkl import SingleTaskDKL, MultiTaskDKL
from torch.utils.data import Dataset
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from autophasemap import BaseDataSet

def initialize_model(model_args, input_dim, output_dim, device):
    if model_args["model"] == 'gp':
        if output_dim == 1:
            return SingleTaskGP(model_args, input_dim, output_dim)
        else:
            return MultiTaskGP(model_args, input_dim, output_dim)
    elif model_args["model"] == 'dkl':
        if output_dim == 1:
            return SingleTaskDKL(model_args, input_dim, output_dim, device)
        else:
            return MultiTaskDKL(model_args, input_dim, output_dim, device)
    else:
        raise NotImplementedError("Model type %s does not exist" % model_args["model"])


def initialize_points(bounds, n_init_points, device):
    if n_init_points < 1:
        init_x = torch.zeros(1, 1).to(device)
    else:
        bounds = bounds.to(device, dtype=torch.double)
        init_x = draw_sobol_samples(bounds=bounds, n=n_init_points, q=1).squeeze(-2)

    return init_x

def construct_acqf_by_model(model, train_x, train_y, num_objectives=1):
    dim = train_y.shape[1]
    sampler = StochasticSampler(sample_shape=torch.Size([256]))
    if num_objectives==1:
        acqf = qUpperConfidenceBound(model=model, beta=100, sampler=sampler)
    else:
        weights = torch.ones(dim)/dim
        posterior_transform = ScalarizedPosteriorTransform(weights.to(train_x))
        acqf = qUpperConfidenceBound(model=model, 
        beta=100, 
        sampler=sampler,
        posterior_transform = posterior_transform
        )


    return acqf 

class ActiveLearningDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = self.to_tensor(x,y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        xs = self.x[idx]
        ys = self.y[idx]

        return xs, ys 

    def to_tensor(self, x, y):
        x_ = torch.Tensor(x).to(device)
        y_ = torch.Tensor(y).to(device)

        return x_, y_
    
    def update(self, x, y):
        x, y = self.to_tensor(x, y)
        self.x = torch.vstack((self.x, x))
        self.y = torch.vstack((self.y, y))

        return
    
def from_comp_to_spectrum(test_function, gp_model, np_model, c):
    with torch.no_grad():
        t_ = test_function.sim.t
        c = torch.tensor(c).to(device)
        gp_model.eval()
        normalized_x = normalize(c, test_function.bounds.to(c))
        posterior = gp_model.posterior(normalized_x)  # based on https://github.com/pytorch/botorch/issues/1110
        t = torch.from_numpy(t_).to(device)
        t = t.repeat(c.shape[0]).view(c.shape[0], len(t_), 1)
        mu = []
        for _ in range(250):
            mu_i, _ = np_model.xz_to_y(t, posterior.rsample().squeeze(0))
            mu.append(mu_i)
        return torch.cat(mu).mean(dim=0, keepdim=True), torch.cat(mu).std(dim=0, keepdim=True)
    
def get_twod_grid(n_grid, bounds):
    x = np.linspace(bounds[0,0],bounds[1,0], n_grid)
    y = np.linspace(bounds[0,1],bounds[1,1], n_grid)
    X,Y = np.meshgrid(x,y)
    points = np.vstack([X.ravel(), Y.ravel()]).T 

    return points 

# Define a autophasemap dataset object
class AutoPhaseMapDataSet(BaseDataSet):
    def __init__(self, C, q, Iq):
        super().__init__(n_domain=q.shape[0])
        self.t = np.linspace(0,1, num=self.n_domain)
        self.q = q
        self.N = C.shape[0]
        self.Iq = Iq
        self.C = C 

        assert self.N==self.Iq.shape[0], "C and Iq should have same number of rows"
        assert self.n_domain==self.Iq.shape[1], "Length of q should match with columns size of Iq"
    
    def generate(self, process=None):
        if process=="normalize":
            self.F = [self.Iq[i,:]/self.l2norm(self.q, self.Iq[i,:]) for i in range(self.N)]
        elif process=="smoothen":
            self.F = [self._smoothen(self.Iq[i,:]/self.l2norm(self.q, self.Iq[i,:]), window_length=7, polyorder=3) for i in range(self.N)]
        elif process is None:
            self.F = [self.Iq[i,:] for i in range(self.N)]

        assert len(self.F)==self.N, "Total number of functions should match the self.N"    

        return