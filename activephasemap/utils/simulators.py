import numpy as np 
import torch
import glob
from scipy.spatial.distance import cdist
import pandas as pd 
from scipy import interpolate 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from activephasemap.utils.visuals import MinMaxScaler, scaled_tickformat

class GNPPhases:
    def __init__(self, dir):
        comps = pd.read_csv(dir+'/grid.csv').to_numpy()
        files = glob.glob(dir+'/Grid_*.xlsx')
        self.spectra_files = [pd.read_excel(file) for file in files]
        AG_x = comps[:,0]*0.00064/350*10**5
        AA_x = comps[:,1]*0.00630/350*10**4
        self.points = np.hstack((AG_x.reshape(-1,1), AA_x.reshape(-1,1)))
        self.wl = self.spectra_files[0]['Wavelength'].values.astype('double')
        self.t = (self.wl - min(self.wl))/(max(self.wl) - min(self.wl))
        self.n_domain = len(self.wl)
        
    def simulate(self, c):
        rid = np.random.choice(len(self.spectra_files))
        lookup_dist = cdist(c.reshape(1,-1), self.points)
        lookup_cid = np.argmin(lookup_dist)
        y = self.spectra_files[rid].iloc[:,lookup_cid+1].values.astype('double')

        return y 
    
    def generate(self):
        self.F = [self.simulate(ci) for ci in self.points] 
        self.comps = self.points 
        self.spectra = np.asarray(self.F)

        return

class UVVisExperiment:
    """Experiment class to facilatate interaction between activephasemap and UV-Vis plate reader data

    For UV-Vis Spectrascopy, the data directory should contain 
        1. comps_x.npy - compositions in (num_comps x dim) shaped numpy array (.npy)
        2. spectra_x.npy - spectra in a numpy array with rows corresponds to the composition (.npy)
        3. wav.npy - wavelength vector in (num_wavelengths x ) shaped numpy array (.npy)
    """
    def __init__(self, bounds, iter, dir):
        self.dim = len(bounds)
        self.bounds = torch.tensor(bounds).transpose(-1, -2).to(device)
        self.dir = dir 
        comps, spectra = [], []
        for k in range(iter):
            comps.append(np.load(self.dir+'comps_%d.npy'%k))
            spectra.append(np.load(self.dir+'spectra_%d.npy'%k))
            print('Loading data from iteration %d with shapes:'%k, comps[k].shape, spectra[k].shape)
        self.comps = np.vstack(comps)
        self.points = self.comps
        self.spectra = np.vstack(spectra)
        self.wav = np.load(self.dir+'wav.npy')

    def generate(self, use_spline=False):
        if use_spline:
            self.wl = np.linspace(min(self.wav), max(self.wav), num=100)
            self.F = [self.spline_interpolate(self.wl, self.spectra[i,:]) for i in range(len(self.comps))]
            self.t = (self.wl - min(self.wl))/(max(self.wl) - min(self.wl))
        else:
            self.F = [self.normalize(self.spectra[i,:]) for i in range(len(self.comps))]
            self.wl = self.wav.copy()
            self.t = (self.wav - min(self.wav))/(max(self.wav) - min(self.wav))

        self.n_domain = len(self.t)
        self.spectra_normalized = np.asarray(self.F)

        return 

    def normalize(self, f):
        norm = np.sqrt(np.trapz(f**2, self.wav))

        return f/norm 

    def plot(self, ax, bounds):
        if len(bounds)==2:
            bounds = np.asarray(bounds).T
            scaler_x = MinMaxScaler(bounds[0,0], bounds[1,0])
            scaler_y = MinMaxScaler(bounds[0,1], bounds[1,1])
            ax.xaxis.set_major_formatter(lambda x, pos : scaled_tickformat(scaler_x, x, pos))
            ax.yaxis.set_major_formatter(lambda y, pos : scaled_tickformat(scaler_y, y, pos))
            t = np.linspace(0,1, self.spectra.shape[1])
            for i, (ci, si) in enumerate(zip(self.comps, self.spectra)):
                norm_ci = np.array([scaler_x.transform(ci[0]), scaler_y.transform(ci[1])])
                self._inset_spectra(norm_ci,t, si, ax)
            ax.set_xlabel('C1', fontsize=20)
            ax.set_ylabel('C2', fontsize=20) 
        else:
            for si in self.spectra:
                ax.plot(self.wav, si)
            ax.set_xlabel('wavelength')
            ax.set_ylabel('intensity') 

        return ax

    def _inset_spectra(self, c, t, ft, ax, **kwargs):
        loc_ax = ax.transLimits.transform(c)
        ins_ax = ax.inset_axes([loc_ax[0],loc_ax[1],0.1,0.1])
        ins_ax.plot(t, ft, **kwargs)
        ins_ax.axis('off')
        
        return 

    def spline_interpolate(self, wl_, y):
        spline = interpolate.splrep(self.wav, y, s=0)
        I_grid = interpolate.splev(wl_, spline, der=0)

        return I_grid