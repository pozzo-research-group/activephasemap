import numpy as np
import torch
from torch.distributions import Normal
from torch.utils.data import Dataset
RNG = np.random.default_rng()
import glob, pdb
import matplotlib.pyplot as plt
from scipy import interpolate 
from activephasemap.models.np import context_target_split
from activephasemap.utils import _inset_spectra
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

class SAXSPairWise(Dataset):
    def __init__(self, root_dir, n_sub_sample = 1000):
        """
        Arguments:
            root_dir (string): Directory with all the data.
        """
        self.dir = root_dir
        self.models = ["sphere", 
                       "cylinder", 
                       "ellipsoid", 
                       "elliptical_cylinder"
                       ]
        self.ns = n_sub_sample 

        q_values = np.load(self.dir+"/q_values.npy")
        self.flags = q_values<=1e-1
        self.q = q_values[self.flags]

        self.xrange = [1e-3, 1000]
        self.n_domain = 300
        self.r = np.logspace(np.log10(self.xrange[0]), 
                             np.log10(self.xrange[1]), 
                             self.n_domain
                             ) 

    def __len__(self):
        return len(self.models)*self.ns

    def __getitem__(self, i):
        quotient, remainder = divmod(i, self.ns)
        data = np.load(self.dir+"/data_by_models/%s.npy"%self.models[quotient])
        Iq = data[remainder,self.flags]
        pr = self.convert_to_pairwise(Iq)

        r_scaled = (self.r-self.xrange[0])/(self.xrange[1]-self.xrange[0])
        domain = torch.tensor(r_scaled).unsqueeze(1).to(torch.double)
        codomain = torch.tensor(pr).unsqueeze(1).to(torch.double)

        return domain, codomain 

    def convert_to_pairwise(self, I):
        '''Converts the scattering intensity back to the pairwise distribution function.
        
        inputs: 
        - r: the pairwise distances of the two randomly sampled coordinates of the structure
        - I: the scattering intensity 
        - q: the momentum transfer vector
        
        outputs:
        - p_r: the pairwise distribution which is a function of (r).'''
        pr = []
        for r_val in self.r:
            integrand = (I * self.q * r_val * np.sin(self.q * r_val))/((2 * np.pi**2))
            pr.append(simpson(integrand, x=self.q))

        return np.asarray(pr)

    def convert_to_intensity(self, pr):
        '''Converts the pairwise distribution function into the scattering intensity as a function of q. 
        
        inputs:
        - q: the momentum transfer vector (q) 
        - self.p_r: the pairwise distribution function  
        - self.r: the pairwise distances of the two randomly sampled coordinates of the structure
        
        results:
        - self.I_q: the scattering intensity curve as a function of q.'''
        Iq = []
        for q_val in self.q:
            integrand = 4 * np.pi * pr * sas_sinx_x(q_val * self.r)
            Iq.append(simpson(integrand, x=self.r))
        
        return gaussian_filter1d(np.asarray(Iq), sigma=1.25)

class SAXSLogLogV1(Dataset):
    def __init__(self, root_dir, n_sub_sample = 1000):
        """
        Arguments:
            root_dir (string): Directory with all the data.
        """
        self.dir = root_dir
        self.models = ["sphere", 
                       "cylinder", 
                       "ellipsoid", 
                       "elliptical_cylinder"
                       ]
        self.ns = n_sub_sample 

        q_values = np.load(self.dir+"/q_values.npy")
        self.q = np.log10(q_values)
        self.n_domain = 100
        self.xrange = [-3, 0]

    def __len__(self):
        return len(self.models)*self.ns

    def __getitem__(self, i):
        quotient, remainder = divmod(i, self.ns)
        data = np.load(self.dir+"/data_by_models/%s.npy"%self.models[quotient])
        Iq = np.log10(data[remainder,:])
        spline = interpolate.splrep(self.q, Iq, s=0)
        q_grid = np.linspace(self.xrange[0], self.xrange[1], self.n_domain)
        I_grid = interpolate.splev(q_grid, spline, der=0)

        domain = torch.tensor(q_grid).unsqueeze(1).to(torch.double)
        codomain = torch.tensor(I_grid).unsqueeze(1).to(torch.double)

        return domain, codomain 

class SAXSPorodV1(Dataset):
    def __init__(self, root_dir, n_sub_sample = 1000):
        """
        Arguments:
            root_dir (string): Directory with all the data.
        """
        self.dir = root_dir
        self.models = ["sphere", 
                       "cylinder", 
                       "ellipsoid", 
                       "elliptical_cylinder"
                       ]
        self.ns = n_sub_sample 

        self.q = np.load(self.dir+"/q_values.npy")
        self.xrange = [-3, 0]
        self.n_domain = 100
        self.q_grid = np.linspace(self.xrange[0], self.xrange[1], self.n_domain)
        self.rids = np.random.randint(16000, size=self.ns)

    def __len__(self):
        return len(self.models)*self.ns

    def __getitem__(self, i):
        model_id, curve_id = divmod(i, self.ns)
        data = np.load(self.dir+"/data_by_models/%s.npy"%self.models[model_id])
        Iq = self.transform(self.q, data[self.rids[curve_id],:])
        spline = interpolate.splrep(np.log10(self.q), Iq, s=0)
        I_grid = interpolate.splev(self.q_grid, spline, der=0)

        domain = torch.tensor(self.q_grid).unsqueeze(1).to(torch.double)
        codomain = torch.tensor(I_grid).unsqueeze(1).to(torch.double)

        return domain, codomain 

    def transform(self, x, y):
        return 1e3 * y * (x**4)

    def inverse_transform(self, x, y):
        # convert q from log transform and apply the porod scaling by 4
        return y/(1e3 * 10 **(4*x)) 

class SAXSPorod(Dataset):
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
        Iq = self.transform(self.q, self.Iq[i,:])
        spline = interpolate.splrep(np.log10(self.q), Iq, s=0)
        I_grid = interpolate.splev(self.q_grid, spline, der=0)

        domain = torch.tensor(self.q_grid).unsqueeze(1).to(torch.double)
        codomain = torch.tensor(I_grid).unsqueeze(1).to(torch.double)

        return domain, codomain 

    def transform(self, x, y):
        return 1e3 * y * (x**4)

    def inverse_transform(self, x, y):
        # convert q from log transform and apply the porod scaling by 4
        return y/(1e3 * 10 **(4*x)) 

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
    if isinstance(dataset, SAXSLogLog):
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(1,2, figsize=(4*2, 4))

    for i in np.random.randint(len(dataset), size=n_samples):
        xi, yi = dataset[i]
        xi_np = xi.detach().cpu().squeeze().numpy()
        yi_np = yi.detach().cpu().squeeze().numpy()
        if isinstance(dataset, SAXSPairWise):
            pr = yi.cpu().squeeze().numpy()
            Iq = dataset.convert_to_intensity(pr)
            ax[0].plot(dataset.r, pr, c='tab:blue', alpha=0.5)
            ax[1].loglog(dataset.q, Iq, c='tab:blue', alpha=0.5)
        elif isinstance(dataset, SAXSPorod):
            Iq = dataset.inverse_transform(xi_np, yi_np)
            ax[0].plot(xi_np, yi_np, c='tab:blue', alpha=0.5)
            ax[1].loglog(10**xi_np, Iq, c='tab:blue', alpha=0.5)        
        else:
            ax.plot(xi_np, yi_np, c='tab:blue', alpha=0.5)
    
    return fig, ax

def plot_samples(ax, dataset, model, x_target, z_dim, num_samples=100):
    z_sample = torch.randn((num_samples, z_dim))
    with torch.no_grad():
        for zi in z_sample:
            mu, _ = model.xz_to_y(x_target, zi.to(device))
            y = mu.detach().cpu().squeeze().numpy()
            x = x_target.squeeze().cpu().numpy()
            if isinstance(dataset, SAXSPairWise):
                Iq = dataset.convert_to_intensity(y)
                ax[0].plot(dataset.r, pr, c='tab:blue', alpha=0.5)
                ax[1].loglog(dataset.q, Iq, c='tab:blue', alpha=0.5)
            elif isinstance(dataset, SAXSPorod):
                Iq = dataset.inverse_transform(x, y)
                ax[0].plot(x, y, c='tab:blue', alpha=0.5)
                ax[1].loglog(10**x, Iq, c='tab:blue', alpha=0.5)
            else:
                ax.plot(x, y, c='tab:blue', alpha=0.5)

    return 

def plot_posterior_samples(x_target, dataset, model):
    fig, axs = plt.subplots(2,5, figsize=(4*5, 4*2))
    rids = np.random.randint(len(dataset), size=10)
    for i, ax in enumerate(axs.flatten()):
        xi, yi = dataset[i]
        n_domain = xi.shape[0]
        # Sample locations of context and target points
        locations = np.random.choice(n_domain, size=n_domain, replace=False)

        x_context = xi[locations[:int(n_domain/2)], :].reshape(1,int(n_domain/2),1).to(device)
        y_context = yi[locations[:int(n_domain/2)], :].reshape(1,int(n_domain/2),1).to(device)
        x_target = xi[locations, :].reshape(1,n_domain,1).to(device)
        y_target = yi[locations, :].reshape(1,n_domain,1).to(device)

        with torch.no_grad():
            for _ in range(200):
                # Neural process returns distribution over y_target
                p_y_pred = model(x_context, y_context, x_target)

                # Extract mean of distribution
                y = p_y_pred.loc.detach().cpu().squeeze().numpy()
                x = x_target.squeeze().cpu().numpy()
                if isinstance(dataset, SAXSPairWise):
                    Iq = dataset.convert_to_intensity(y)
                    ax.loglog(dataset.q, Iq, c='tab:blue', alpha=0.5)
                elif isinstance(dataset, SAXSPorod):
                    Iq = dataset.inverse_transform(x, y)
                    ax.loglog(10**x, Iq, c='tab:blue', alpha=0.5)
                else:
                    ax.plot(x, y, alpha=0.05, c='tab:blue')

            if isinstance(dataset, SAXSPairWise):
                Iq = dataset.convert_to_intensity(yi.detach().cpu().squeeze().numpy())
                ax.plot(dataset.q, Iq, c='tab:red')
            elif isinstance(dataset, SAXSPorod):
                x_np = xi.detach().cpu().squeeze().numpy()
                Iq = dataset.inverse_transform(x_np, 
                                               yi.detach().cpu().squeeze().numpy()
                                               )
                ax.loglog(10**x_np, Iq, c='tab:red')
            else:
                ax.scatter(x_context.cpu().numpy(), y_context.cpu().numpy(), c='tab:red')
                ax.plot(xi.cpu().squeeze().numpy(), yi.cpu().squeeze().numpy(), c='tab:red')

    return fig, axs
