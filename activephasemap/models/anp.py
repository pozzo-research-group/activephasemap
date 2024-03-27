import numpy as np
import matplotlib.pyplot as plt
import collections

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Code modified from https://github.com/KurochkinAlexey/Attentive-neural-processes/blob/master/anp_1d_regression.ipynb
import numpy as np
import collections
import gpytorch
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
import abc 

# The (A)NP takes as input a `NPRegressionDescription` namedtuple with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tensor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that describes the total
#     number of datapoints used (context + target)
#   `num_context_points`: A vector containing a scalar that describes the number
#     of datapoints used as context


NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"))

class NPRegressionDataset:
    def __init__(self, max_num_context, batch_size, testing=False):
        self.max_num_context = max_num_context 
        self.batch_size = batch_size
        self.testing = testing

    @abc.abstractmethod
    def sample(self):
        """Generate samples from the dataset.

        Parameters
        ----------
        

        Returns
        -------
        x_valyes : array-like, shape=[...,]
            x_values of the data set.
        y_valyes : array-like, shape=[...,]
            f(x_values) of the data set.            
        """

    def get_context_targets(self):
        num_context = int(np.random.rand()*(self.max_num_context - 3) + 3)
        num_target = int(np.random.rand()*(self.max_num_context - num_context))

        return num_context, num_target

    def process(self, x_values, y_values):
        num_context, num_target = self.get_context_targets()
        num_total_points = x_values.shape[1]
        idx = torch.randperm(num_total_points)
        if self.testing:
            # Select the targets
            target_x = x_values
            target_y = y_values

            # Select the observations
            context_x = x_values[:, idx[:num_context],:]
            context_y = y_values[:, idx[:num_context]]
        else:
            # Select the targets which will consist of the context points as well as
            # some new target points
            target_x = x_values[:, idx[:num_target + num_context],:]
            target_y = y_values[:, idx[:num_target + num_context],:]

            # Select the observations
            context_x = x_values[:, idx[:num_context]]
            context_y = y_values[:, idx[:num_context]]

        context_x = context_x.to(device)
        context_y = context_y.to(device)
        target_x = target_x.to(device)
        target_y = target_y.to(device)

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
        query=query,
        target_y=target_y,
        num_total_points=target_x.shape[1],
        num_context_points=num_context)

# ### Attention module for the decoder
class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_type, n_heads=8):
        super().__init__()
        if attention_type == "uniform":
            self._attention_func = self._uniform_attention
        elif attention_type == "laplace":
            self._attention_func = self._laplace_attention
        elif attention_type == "dot":
            self._attention_func = self._dot_attention
        elif attention_type == "multihead":
            self._W_k = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_heads)])
            self._W_v = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_heads)])
            self._W_q = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_heads)])
            self._W = nn.Linear(n_heads*hidden_dim, hidden_dim)
            self._attention_func = self._multihead_attention
            self.n_heads = n_heads
        else:
            raise NotImplementedError
            
    def forward(self, k, v, q):
        rep = self._attention_func(k, v, q)
        return rep
    
    def _uniform_attention(self, k, v, q):
        total_points = q.shape[1]
        rep = torch.mean(v, dim=1, keepdim=True)
        rep = rep.repeat(1, total_points, 1)
        return rep
    
    def _laplace_attention(self, k, v, q, scale=0.5):
        k_ = k.unsqueeze(1)
        v_ = v.unsqueeze(2)
        unnorm_weights = torch.abs((k_ - v_)*scale)
        unnorm_weights = unnorm_weights.sum(dim=-1)
        weights = torch.softmax(unnorm_weights, dim=-1)
        rep = torch.einsum('bik,bkj->bij', weights, v)
        return rep
    
    def _dot_attention(self, k, v, q):
        scale = q.shape[-1]**0.5
        unnorm_weights = torch.einsum('bjk,bik->bij', k, q) / scale
        weights = torch.softmax(unnorm_weights, dim=-1)
        
        rep = torch.einsum('bik,bkj->bij', weights, v)
        return rep
    
    def _multihead_attention(self, k, v, q):
        outs = []
        for i in range(self.n_heads):
            k_ = self._W_k[i](k)
            v_ = self._W_v[i](v)
            q_ = self._W_q[i](q)
            out = self._dot_attention(k_, v_, q_)
            outs.append(out)
        outs = torch.stack(outs, dim=-1)
        outs = outs.view(outs.shape[0], outs.shape[1], -1)
        rep = self._W(outs)
        return rep

# ### Encoder models
class DeterministicEncoder(nn.Module):
    
    def __init__(self, 
                 rep_dim,
                 attention_type="dot"):
        super(DeterministicEncoder, self).__init__()
        layers = [nn.Linear(2, 128),
                  nn.ReLU(inplace=True),
                  nn.Linear(128, 128),
                  nn.ReLU(inplace=True),
                  nn.Linear(128, 128),
                  nn.ReLU(inplace=True),
                  nn.Linear(128, 128),
                  nn.ReLU(inplace=True),
                  nn.Linear(128, 128),
                  nn.ReLU(inplace=True),
                  nn.Linear(128, 128),
                  nn.ReLU(inplace=True),
                  nn.Linear(128, 128),
                  nn.ReLU(inplace=True),
                  nn.Linear(128, rep_dim)
                  ]
        self.mlp = nn.Sequential(*layers)

        if attention_type is None:
            self.attention =  None
        else:
            self.attention = Attention(rep_dim, attention_type)

        self._target_transform = nn.Sequential(*[nn.Linear(1, 32), nn.ReLU(inplace=True)], nn.Linear(32, rep_dim))
        self._context_transform = nn.Sequential(*[nn.Linear(1, 32), nn.ReLU(inplace=True)], nn.Linear(32, rep_dim))

    def forward(self, context_x, context_y, target_x):
        xy = torch.cat([context_x,context_y], dim=-1)
        ri = self.mlp(xy)
        if self.attention is not None:
            x_star = self._target_transform(target_x)
            xi = self._context_transform(context_x)
            ri = self.attention(xi, ri, x_star) 
            return ri
        else:
            ri = torch.mean(ri, dim=1)
            return ri.unsqueeze(1).repeat(1, target_x.shape[1], 1)

class LatentEncoder(nn.Module):
    
    def __init__(self, 
                 latent_dim=32, 
                 ):
        super(LatentEncoder, self).__init__()
        layers = [nn.Linear(2, 128),
                  nn.ReLU(inplace=True),
                  nn.Linear(128, 128),
                  nn.ReLU(inplace=True),
                  nn.Linear(128, 128),
                  ]
        self.mlp = nn.Sequential(*layers)
        self.small_mlp = nn.Sequential(*[nn.Linear(128, 96), nn.ReLU(inplace=True)])
        self.mean = nn.Sequential(*[nn.Linear(96, 96), nn.ReLU(inplace=True), nn.Linear(96, latent_dim)])
        self.log_sigma =  nn.Sequential(*[nn.Linear(96, 96), nn.ReLU(inplace=True), nn.Linear(96, latent_dim)])

    def forward(self, x, y):
        xy = torch.cat([x,y], dim=-1)
        si = self.mlp(xy)
        sc = si.mean(dim=1)
        
        sc = self.small_mlp(sc)
        mean = self.mean(sc)
        log_sigma = self.log_sigma(sc)
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)

        return Normal(mean, sigma)

# ### Decoder module
class Decoder(nn.Module):
    
    def __init__(self,  
                 latent_dim=32,
                 rep_dim = 64 
                 ):
        super(Decoder, self).__init__()
        layers = [nn.Linear(1+rep_dim+latent_dim, 64),
                  nn.ReLU(inplace=True),
                  nn.Linear(64, 64),           
                  ]
        self.mlp = nn.Sequential(*layers)
        self.mean = nn.Sequential(*[nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Linear(64, 1)])
        self.log_sigma = nn.Sequential(*[nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Linear(64, 1)])
        
    def forward(self, r, z, x):
        rzx = torch.cat([r, z, x], dim=-1)
        rzx = self.mlp(rzx)

        mean = self.mean(rzx)
        log_sigma = self.log_sigma(rzx)
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)

        return Normal(mean, sigma)

# ### Attention-Neural Process Model
class NeuralProcessModel(nn.Module):
    
    def __init__(self,
                 rep_dim=32, 
                 latent_dim=32, 
                 attn_type="dot"
                 ):
        
        super().__init__()
        self.r_encoder = DeterministicEncoder(rep_dim=rep_dim,
                                              attention_type=attn_type,
                                              )
                
        self.z_encoder = LatentEncoder(latent_dim=latent_dim)
        
        self.decoder = Decoder(latent_dim=latent_dim, rep_dim=rep_dim)
        
    def forward(self, context_x, context_y, target_x, target_y=None):
        num_targets = target_x.size(1)
        rc = self.r_encoder(context_x, context_y, target_x)
        q_context = self.z_encoder(context_x, context_y)
        
        if target_y is None:
            z = q_context.rsample()
        else:
            q_target = self.z_encoder(target_x, target_y)
            z = q_target.rsample()

        z = z.unsqueeze(1).repeat(1,num_targets,1)
        dist = self.decoder(rc, z, target_x)
        if target_y is not None:
            log_likelihood = dist.log_prob(target_y)
            kl_loss = kl_divergence(q_target, q_context)
            kl_loss = torch.sum(kl_loss, dim=-1, keepdim=True)
            kl_loss = kl_loss.repeat(1, num_targets).unsqueeze(-1)
            loss = -torch.mean((log_likelihood - kl_loss)/num_targets)
            
            return dist, log_likelihood, kl_loss, loss
        else:
            return dist
        

# ### Plotting utilities
def plot_functions(target_x, target_y, context_x, context_y, pred_y, std):
    """Plots the predicted mean and variance and the context points.
  
  Args: 
    target_x: An array of shape [B,num_targets,1] that contains the
        x values of the target points.
    target_y: An array of shape [B,num_targets,1] that contains the
        y values of the target points.
    context_x: An array of shape [B,num_contexts,1] that contains 
        the x values of the context points.
    context_y: An array of shape [B,num_contexts,1] that contains 
        the y values of the context points.
    pred_y: An array of shape [B,num_targets,1] that contains the
        predicted means of the y values at the target points in target_x.
    std: An array of shape [B,num_targets,1] that contains the
        predicted std dev of the y values at the target points in target_x.
      """
  # Plot everything
    fig, ax = plt.subplots()
    ax.plot(target_x[0], pred_y[0], 'b', linewidth=2)
    ax.plot(target_x[0], target_y[0], 'k:', linewidth=2)
    ax.plot(context_x[0], context_y[0], 'ko', markersize=10)
    ax.fill_between(
          target_x[0, :, 0],
          pred_y[0, :, 0] - std[0, :, 0],
          pred_y[0, :, 0] + std[0, :, 0],
          alpha=0.2,
          facecolor='#65c9f7',
          interpolate=True)
    
    return fig, ax


