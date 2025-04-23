import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from random import randint
from torch.distributions.kl import kl_divergence

import pdb 

from perceiver_pytorch import PerceiverIO 
from .np import MuSigmaEncoder, NeuralProcess, Decoder
import numpy as np

class PositionEmbedder(nn.Module):
    def __init__(self, num_freq=16, sigma=1):
        super(PositionEmbedder, self).__init__()

        self.num_freq = num_freq
        self.sigma = sigma

        self.freq = nn.Linear(in_features=1, out_features=self.num_freq)
        print('non-learnable frequencies: {}'.format(self.sigma))
        with torch.no_grad(): # fix these weights
            self.freq.weight = nn.Parameter(torch.normal(mean=0, std=self.sigma, 
                                            size=(self.num_freq, 1)), 
                                            requires_grad=False
                                            )
            self.freq.bias = nn.Parameter(torch.zeros(self.num_freq), requires_grad=False)

        self.layers = nn.Sequential(
            nn.Linear(2*self.num_freq, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )

        return
    
    def forward(self, x):
        nb, ns, _ = x.shape
        x = self.freq(x)
        x = torch.cat([torch.sin(2 * np.pi * x), torch.cos(2 * np.pi * x)], dim=-1)
        x = self.layers(x.view(nb*ns, 2*self.num_freq))

        return x

class Encoder(nn.Module):
    def __init__(self, emb, r_dim, n_blocks):
        super().__init__()

        self.input_to_hidden = PerceiverIO(
            dim=16,  
            queries_dim=1,
            depth=n_blocks,  
            num_latents=r_dim,  
            latent_dim=1  
        )
        self.emb = emb

    def forward(self, x, y):
        ns = x.shape[0]
        xe = self.emb(x.view(1, ns, 1))
        latent = self.input_to_hidden(xe.view(1, ns, 16), queries=y.view(1, ns, 1))
        pdb.set_trace()
        return latent 

class PerceiverIONeuralProcess(NeuralProcess):
    """
    Implements Neural Process for functions of arbitrary dimensions.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    """
    def __init__(self, r_dim, z_dim, h_dim, n_blocks):
        super().__init__(r_dim, z_dim, h_dim, n_blocks)
        # Initialize networks
        emb = PositionEmbedder()
        self.xy_to_r = Encoder(emb, self.r_dim, n_blocks=self.n_blocks)
        self.r_to_mu_sigma = MuSigmaEncoder(self.r_dim, self.z_dim)
        # self.xz_to_y = Decoder(emb, self.z_dim, self.h_dim, n_blocks=self.n_blocks)
        self.xz_to_y = Decoder(1, z_dim, self.h_dim, 1, n_blocks=self.n_blocks)
