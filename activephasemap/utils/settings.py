import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize
from botorch.sampling.stochastic_samplers import StochasticSampler 
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.optim.initializers import initialize_q_batch_nonneg
from activephasemap.models.gp import MultiTaskGP 
from autophasemap import BaseDataSet
from torch.utils.data import Dataset
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
import pdb

def initialize_model(train_x, train_y, model_args, input_dim, output_dim, device):
    if model_args["model"] == 'gp':
       return MultiTaskGP(train_x, train_y, model_args, input_dim, output_dim)
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
    sampler = StochasticSampler(sample_shape=torch.Size([256]))
    if num_objectives==1:
        acqf = qUpperConfidenceBound(model=model, beta=100, sampler=sampler)
    else:
        weights = torch.ones(model.output_dim)/model.output_dim
        posterior_transform = ScalarizedPosteriorTransform(weights.to(train_x))
        acqf = qUpperConfidenceBound(model=model, 
        beta=100, 
        sampler=sampler,
        posterior_transform = posterior_transform
        )

    return acqf 

def _optimize_acqf(model, bounds, num_batches, num_restarts=16):
    weights = torch.ones(model.num_outputs)/model.num_outputs
    posterior_transform = ScalarizedPosteriorTransform(weights.to(device))
    sampler = StochasticSampler(sample_shape=torch.Size([256]))
    acqf = qUpperConfidenceBound(model=model, 
                                 beta=100, 
                                 sampler=sampler,
                                 posterior_transform = posterior_transform
                                 )
    Xraw = torch.rand(100 * num_restarts, num_batches, len(bounds)).to(device)
    Xraw = bounds[0] + (bounds[1] - bounds[0]) * Xraw
    # evaluate the acquisition function on these q-batches
    Yraw = acqf(Xraw) 
    X = initialize_q_batch_nonneg(Xraw, Yraw, num_restarts)    # apply the heuristic for sampling promising initial conditions
    X.requires_grad_(True)
    optimizer = torch.optim.Adam([X], lr=0.01)

    for i in range(100):
        optimizer.zero_grad()
        pdb.set_trace()
        losses = -acqf(X)
        loss = losses.sum()
        loss.backward() 
        optimizer.step()

        # clamp values to the feasible set
        for j, (lb, ub) in enumerate(zip(*bounds)):
            X.data[..., j].clamp_(lb, ub)  # need to do this on the data not X itself

        if (i + 1) % 15 == 0:
            print(f"Iteration {i+1:>3}/100 - Loss: {loss.item():>4.3f}")

    return X[losses.argmin().item(),...], acqf
    

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
    
def from_comp_to_spectrum(expt, gp_model, np_model, c):
    with torch.no_grad():
        c = torch.tensor(c).to(device)
        gp_model.eval()
        normalized_x = normalize(c, expt.bounds.to(c))
        posterior = gp_model.posterior(normalized_x)  # based on https://github.com/pytorch/botorch/issues/1110
        t = torch.from_numpy(expt.t).to(device).reshape(1, len(expt.t), 1)
        mu = []
        for _ in range(100):
            z = posterior.rsample().squeeze(0)
            mu_i, _ = np_model.xz_to_y(t, z)
            mu.append(mu_i)

        mean_pred = torch.cat(mu).mean(dim=0, keepdim=True)
        sigma_pred = torch.cat(mu).std(dim=0, keepdim=True)

        if torch.isnan(mean_pred).any():
            raise RuntimeError("Predicted mean is nan")
        
        neg_pred_flags = (mean_pred<0)
        if (mean_pred[neg_pred_flags]>0.1).any():
            raise RuntimeError("Spectrum values are below 0 and larger than 0.1 threshold", mean_pred[neg_pred_flags])
        else:
            mean_pred = torch.abs(mean_pred)

        return mean_pred, sigma_pred 
    
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