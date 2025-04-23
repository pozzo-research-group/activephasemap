import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from random import randint
from torch.distributions.kl import kl_divergence

import numpy as np            
import matplotlib.pyplot as plt
import pdb 

def context_target_split(x, y, num_context, num_extra_target):
    """Given inputs x and their value y, return random subsets of points for
    context and target. Note that following conventions from "Empirical
    Evaluation of Neural Process Objectives" the context points are chosen as a
    subset of the target points.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, x_dim)

    y : torch.Tensor
        Shape (batch_size, num_points, y_dim)

    num_context : int
        Number of context points.

    num_extra_target : int
        Number of additional target points.
    """
    num_points = x.shape[1]
    # Sample locations of context and target points
    locations = np.random.choice(num_points,
                                 size=num_context + num_extra_target,
                                 replace=False)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations, :]
    y_target = y[:, locations, :]
    return x_context, y_context, x_target, y_target

class PositionEmbedder(nn.Module):
    def __init__(self, basis = "bessel", n_freq=32, sigma=1, n_latents = 8):
        super(PositionEmbedder, self).__init__()

        if basis=="bessel":
            self.basis = self.bessel_functions_basis 
            self.n_basis = 3
        else:
            self.basis = self.fourier_basis
            self.n_basis = 2

        self.n_freq = n_freq
        self.sigma = sigma 
        self.n_latents = n_latents

        self.freq = nn.Linear(in_features=1, out_features=self.n_freq)
        with torch.no_grad(): # fix these weights
            wts = torch.normal(mean=0,std=self.sigma, size=(self.n_freq, 1))
            self.freq.weight = nn.Parameter(torch.exp(wts), requires_grad=False)
            self.freq.bias = nn.Parameter(torch.zeros(self.n_freq), requires_grad=False)

        self.layers = nn.Sequential(
            nn.Linear(self.n_basis*self.n_freq, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_latents)
        )

        return
    
    def forward(self, x):
        nb, ns, _ = x.shape
        xn = self.normalize(x)
        xf = self.freq(xn)
        xb = self.basis(xf)
        xe = self.layers(xb)

        return xe

    def normalize(self, x):
        lower, upper = x.min(1, keepdim=True)[0], x.max(1, keepdim=True)[0]

        return (x-lower)/(upper-lower)
    
    def unnormalize(self, x, lower, upper):

        return (upper-lower)*x+lower

    def bessel_functions_basis(self, x):
        eps = 1e-4
        x = self.unnormalize(x, 0.01, 20.0)
        j0 = torch.sin(x)/(x+eps)
        j1 = ( torch.sin(x)/(x**2 + eps) ) - (torch.cos(x)/(x+eps))
        j2 = ( (3.0/x**2) - 1.0 )*torch.sin(x)/(x+eps) - (3*torch.cos(x)/(x**2+eps))
 
        return torch.cat([j0, j1, j2], dim=-1)

    def fourier_basis(self, x):
        return torch.cat([torch.sin(2 * np.pi * x), torch.cos(2 * np.pi * x)], dim=-1)


class Encoder(nn.Module):
    """Maps an (x_i, y_i) pair to a representation r_i.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer.

    r_dim : int
        Dimension of output representation r.
    """
    def __init__(self, emb, x_dim, y_dim, h_dim, r_dim, n_blocks=5):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.r_dim = r_dim
        self.emb = emb

        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Linear(self.h_dim, self.h_dim))
            blocks.append(nn.ReLU())
        head = [nn.Linear(self.emb.n_latents + self.y_dim, self.h_dim), nn.ReLU()]
        tail = [nn.Linear(self.h_dim, self.r_dim)]
        layers = []
        layers.append(head)
        layers.append(blocks)
        layers.append(tail)
        layers = [x for xs in layers for x in xs]

        self.input_to_hidden = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        xe = self.emb(x)
        input_pairs = torch.cat((xe, y), dim=-1)
        return self.input_to_hidden(input_pairs)

class MuSigmaEncoder(nn.Module):
    """
    Maps a representation r to mu and sigma which will define the normal
    distribution from which we sample the latent variable z.

    Parameters
    ----------
    r_dim : int
        Dimension of output representation r.

    z_dim : int
        Dimension of latent variable z.
    """
    def __init__(self, r_dim, z_dim):
        super(MuSigmaEncoder, self).__init__()

        self.r_dim = r_dim
        self.z_dim = z_dim

        self.r_to_hidden = nn.Linear(self.r_dim, self.r_dim)
        self.hidden_to_mu = nn.Linear(self.r_dim, self.z_dim)
        self.hidden_to_sigma = nn.Linear(self.r_dim, self.z_dim)

    def forward(self, r):
        """
        r : torch.Tensor
            Shape (batch_size, r_dim)
        """
        hidden = self.r_to_hidden(r)
        mu = self.hidden_to_mu(hidden)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * torch.sigmoid(self.hidden_to_sigma(hidden))
        return mu, sigma


class Decoder(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    z_dim : int
        Dimension of latent variable z.

    h_dim : int
        Dimension of hidden layer.

    y_dim : int
        Dimension of y values.
    """
    def __init__(self, emb, x_dim, z_dim, h_dim, y_dim, n_blocks=5):
        super(Decoder, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.emb = emb

        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Linear(self.h_dim, self.h_dim))
            blocks.append(nn.ReLU())
        head = [nn.Linear(self.emb.n_latents + self.z_dim, self.h_dim), nn.ReLU()]
        layers = []
        layers.append(head)
        layers.append(blocks)
        layers = [x for xs in layers for x in xs]

        self.xz_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(self.h_dim, self.y_dim)
        self.hidden_to_sigma = nn.Linear(self.h_dim, self.y_dim)

    def forward(self, x, z):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        z : torch.Tensor
            Shape (batch_size, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        batch_size, num_points, _ = x.size()
        
        # Embed x into frequency domain
        xe = self.emb(x)
        # Repeat z, so it can be concatenated with every x. This changes shape
        # from (batch_size, z_dim) to (batch_size, num_points, z_dim)
        z = z.unsqueeze(1).repeat(1, num_points, 1)
        # Flatten x and z to fit with linear layer
        x_flat = xe.view(batch_size, num_points, xe.shape[-1])
        z_flat = z.view(batch_size, num_points, self.z_dim)
        # Input is concatenation of z with every row of x
        input_pairs = torch.cat((x_flat, z_flat), dim=-1)

        hidden = self.xz_to_hidden(input_pairs)

        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)

        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        return mu, sigma

class NeuralProcess(nn.Module):
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
    def __init__(self, r_dim, z_dim, h_dim, n_blocks=3, pos_basis="bessel"):
        super(NeuralProcess, self).__init__()
        self.x_dim = 1
        self.y_dim = 1
        self.r_dim = r_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.n_blocks = n_blocks

        # Initialize networks
        emb = PositionEmbedder(basis=pos_basis)
        self.xy_to_r = Encoder(emb, self.x_dim, self.y_dim, self.h_dim, self.r_dim, n_blocks=self.n_blocks)
        self.r_to_mu_sigma = MuSigmaEncoder(self.r_dim, self.z_dim)
        self.xz_to_y = Decoder(emb, self.x_dim, self.z_dim, self.h_dim, self.y_dim, n_blocks=self.n_blocks)

    def aggregate(self, r_i):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        """
        return torch.mean(r_i, dim=1)

    def xy_to_mu_sigma(self, x, y):
        """
        Maps (x, y) pairs into the mu and sigma parameters defining the normal
        distribution of the latent variables z.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        y : torch.Tensor
            Shape (batch_size, num_points, y_dim)
        """
        batch_size, num_points, _ = x.size()
        # Flatten tensors, as encoder expects one dimensional inputs
        x_flat = x.view(batch_size, num_points, self.x_dim)
        y_flat = y.contiguous().view(batch_size, num_points, self.y_dim)
        # Encode each point into a representation r_i
        r_i_flat = self.xy_to_r(x_flat, y_flat)
        # Reshape tensors into batches
        r_i = r_i_flat.view(batch_size, num_points, self.r_dim)
        # Aggregate representations r_i into a single representation r
        r = self.aggregate(r_i)
        # Return parameters of distribution
        return self.r_to_mu_sigma(r)

    def forward(self, x_context, y_context, x_target, y_target=None):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.

        Parameters
        ----------
        x_context : torch.Tensor
            Shape (batch_size, num_context, x_dim). Note that x_context is a
            subset of x_target.

        y_context : torch.Tensor
            Shape (batch_size, num_context, y_dim)

        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)

        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.

        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        batch_size, num_context, x_dim = x_context.size()
        _, num_target, _ = x_target.size()
        _, _, y_dim = y_context.size()

        if self.training:
            # Encode target and context (context needs to be encoded to
            # calculate kl term)
            # print('class:NeuralProcess:forward: ', x_target.dtype, y_target.dtype)
            mu_target, sigma_target = self.xy_to_mu_sigma(x_target, y_target)
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from encoded distribution using reparameterization trick
            q_target = Normal(mu_target, sigma_target)
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_target.rsample()
            # Get parameters of output distribution
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred, q_target, q_context
        else:
            # At testing time, encode only context
            mu_context, sigma_context = self.xy_to_mu_sigma(x_context, y_context)
            # Sample from distribution based on context
            q_context = Normal(mu_context, sigma_context)
            z_sample = q_context.rsample()
            # Predict target points based on context
            y_pred_mu, y_pred_sigma = self.xz_to_y(x_target, z_sample)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred

def neural_process_loss(p_y_pred, y_target, q_target, q_context):
    """
    Computes Neural Process loss.

    Parameters
    ----------
    p_y_pred : one of torch.distributions.Distribution
        Distribution over y output by Neural Process.

    y_target : torch.Tensor
        Shape (batch_size, num_target, y_dim)

    q_target : one of torch.distributions.Distribution
        Latent distribution for target points.

    q_context : one of torch.distributions.Distribution
        Latent distribution for context points.
    """
    # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
    # over batch and sum over number of targets and dimensions of y
    log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
    # KL has shape (batch_size, r_dim). Take mean over batch and sum over
    # r_dim (since r_dim is dimension of normal distribution)
    kl = kl_divergence(q_target, q_context).mean(dim=0).sum()

    return -log_likelihood + 0.01*kl

def train_neural_process(model, data_loader, optimizer, **kwargs):
    loss_value = 0
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()

        x, y = data
        n_domain = x.shape[1]
        # Sample number of context and target points
        num_context = randint(3, int((n_domain/2)-3))
        num_extra_target = randint(int(n_domain/2), int(n_domain/2)+2)

        x_context, y_context, x_target, y_target = \
            context_target_split(x, y, num_context, num_extra_target)
        p_y_pred, q_target, q_context = \
            model(x_context, y_context, x_target, y_target)

        loss = neural_process_loss(p_y_pred, y_target, q_target, q_context)
        loss.backward()
        optimizer.step()

        loss_value += loss.item()

    return model, optimizer, loss_value/len(data_loader)      
