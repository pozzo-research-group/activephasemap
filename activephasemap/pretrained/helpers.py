import numpy as np
import torch
from torch.distributions import Normal
from torch.utils.data import Dataset
RNG = np.random.default_rng()
import glob, pdb
import matplotlib.pyplot as plt
from scipy import interpolate 
from activephasemap.models.np import context_target_split
from activephasemap.utils import inset_spectra
from activephasemap.simulators import MinMaxScaler, scaled_tickformat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sasmodels.special import sas_sinx_x
from scipy.integrate import simpson
from scipy.ndimage import gaussian_filter1d

class UVVisDataset(Dataset):
    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Directory with all the data.
        """
        self.dir = root_dir
        self.files = glob.glob(self.dir+'/*.npz')
        self.xrange = [0,1]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            npzfile = np.load(self.files[i])
        except Exception as e:
            print('%s Could not load %s'%(type(e).__name__, self.files[i]))
        wl, I = npzfile['wl'], npzfile['I']
        wl = (wl-min(wl))/(max(wl)-min(wl))
        wl_ = torch.tensor(wl).unsqueeze(1).to(torch.double)
        I_ = torch.tensor(I).unsqueeze(1).to(torch.double)

        return wl_, I_

class SAXSLogLog(Dataset):
    def __init__(self, root_dir):
        """
        Arguments:
            root_dir (string): Directory with all the data.
        """
        self.dir = root_dir
        data = np.load(self.dir+"sasmodels.npz")
        self.q = data["x"]
        self.Iq = data["y"]
        self.xrange = [-3, 0]
        self.n_domain = 100
        self.q_grid = np.linspace(self.xrange[0], self.xrange[1], self.n_domain)

    def __len__(self):
        return self.Iq.shape[0]

    def __getitem__(self, i):
        Iq = self.Iq[i,:]
        spline = interpolate.splrep(np.log10(self.q), np.log10(Iq), s=0)
        I_grid = interpolate.splev(self.q_grid, spline, der=0)

        domain = torch.tensor(self.q_grid).unsqueeze(1).to(torch.double)
        codomain = torch.tensor(I_grid).unsqueeze(1).to(torch.double)

        return domain, codomain 

def plot_dataset_samples(dataset, n_samples=100):
    fig, ax = plt.subplots()

    for i in np.random.randint(len(dataset), size=n_samples):
        xi, yi = dataset[i]
        xi_np = xi.detach().cpu().squeeze().numpy()
        yi_np = yi.detach().cpu().squeeze().numpy()
        ax.plot(xi_np, yi_np, c='tab:blue', alpha=0.5)
    
    return fig, ax

def plot_samples(ax, dataset, model, x_target, z_dim, num_samples=100):
    z_sample = torch.randn((num_samples, z_dim))
    with torch.no_grad():
        for zi in z_sample:
            mu, _ = model.xz_to_y(x_target, zi.to(device))
            y = mu.detach().cpu().squeeze().numpy()
            x = x_target.squeeze().cpu().numpy()
            ax.plot(x, y, c='tab:blue', alpha=0.5)

    return 

def plot_posterior_samples(x_target, dataset, model):
    fig, axs = plt.subplots(2,5, figsize=(4*5, 4*2))
    rids = np.random.randint(len(dataset), size=10)
    for i, ax in enumerate(axs.flatten()):
        xi, yi = dataset[rids[i]]
        n_domain = xi.shape[0]
        # Sample locations of context and target points
        locations = np.random.choice(n_domain, size=n_domain, replace=False)

        x_context = xi[locations[:int(n_domain/2)], :].reshape(1,int(n_domain/2),1).to(device)
        y_context = yi[locations[:int(n_domain/2)], :].reshape(1,int(n_domain/2),1).to(device)

        with torch.no_grad():
            for _ in range(200):
                # Neural process returns distribution over y_target
                p_y_pred = model(x_context, y_context, x_target)

                # Extract mean of distribution
                y = p_y_pred.loc.detach().cpu().squeeze().numpy()
                x = x_target.squeeze().cpu().numpy()
                ax.plot(x, y, alpha=0.5, c='tab:blue')
            ax.scatter(x_context.cpu().numpy(), y_context.cpu().numpy(), c='tab:red')
            ax.plot(xi.cpu().squeeze().numpy(), yi.cpu().squeeze().numpy(), c='tab:red')

    return fig, axs
