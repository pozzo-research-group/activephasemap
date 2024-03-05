import numpy as np 
import torch
import matplotlib.pyplot as plt
import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create synthetic data
class SimulatorTestFunction:
    r"""Base class for phasemapping test functions

    """

    def __init__(self, sim, bounds, num_objectives=3):
        r"""
        Args:
            sim : simulator class for the experiment
            bounds : bounds of input dimensions.
            num_objectives : number of latent dimensions
        """
        self.sim = sim 
        self.dim = len(bounds)
        self.bounds = torch.tensor(bounds).transpose(-1, -2).to(device)
        self.num_objectives = num_objectives

    def evaluate_true(self, np_model, X):
        spectra = torch.zeros((X.shape[0], self.sim.n_domain)).to(device)
        for i, xi in enumerate(X):
            si = self.sim.simulate(xi.cpu().numpy())
            spectra[i] = torch.tensor(si).to(device)
        t = torch.from_numpy(self.sim.t)
        t = t.repeat(X.shape[0], 1).to(device)
        with torch.no_grad():
            z, _ = np_model.xy_to_mu_sigma(t.unsqueeze(2), spectra.unsqueeze(2))
        return z, spectra  

    __call__ = evaluate_true  


class ExperimentalTestFunction:
    r""" test function class for experiments.

    """

    def __init__(self, sim, bounds, num_objectives=3):
        r"""
        Args:
            sim : simulator class for the experiment
            dim : The (input) dimension.
            num_objectives : number of latent dimensions
        """
        self.sim = sim 
        self.dim = len(bounds)
        self.bounds = torch.tensor(bounds).transpose(-1, -2).to(device)
        self.num_objectives = num_objectives
