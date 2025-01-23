"""
GNPPhases and UVVisExperiment Classes for Spectroscopic Data Simulation and Analysis.

This module contains classes for generating, normalizing, and visualizing 
spectroscopic data for compositional studies in materials science. It supports 
data from experiments and simulations, with tools to process, normalize, 
and interpolate spectral data.

Classes
-------
MinMaxScaler
    Scales data to a specified range using min-max normalization.

GNPPhases
    Simulates and generates spectroscopic data based on compositional grids.

UVVisExperiment
    Facilitates interaction between experimental UV-Vis data and active 
    phase map models.

Functions
---------
scaled_tickformat(scaler, x, pos)
    Formats tick values using inverse transformation of a MinMaxScaler.

"""

import numpy as np 
import torch
import glob
from scipy.spatial.distance import cdist
import pandas as pd 
from scipy import interpolate 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MinMaxScaler:
    """
    Min-Max Scaler for normalizing data to a specified range.

    Parameters
    ----------
    min : float
        Minimum value of the scaling range.
    max : float
        Maximum value of the scaling range.

    Methods
    -------
    transform(x)
        Scales the input `x` to the range [0, 1].
    inverse(xt)
        Inverses the scaled data `xt` back to the original range.
    """
    def __init__(self, min, max):
        self.min = min 
        self.max = max 
        self.range = max-min

    def transform(self, x):
        """
        Scale the input data.

        Parameters
        ----------
        x : numpy.ndarray
            Input data to be scaled.

        Returns
        -------
        numpy.ndarray
            Scaled data in the range [0, 1].
        """
        return (x-self.min)/self.range
    
    def inverse(self, xt):
        """
        Inverse the scaling operation.

        Parameters
        ----------
        xt : numpy.ndarray
            Scaled data.

        Returns
        -------
        numpy.ndarray
            Data transformed back to the original range.
        """
        return (self.range*xt)+self.min

def scaled_tickformat(scaler, x, pos):
    return '%.1f'%scaler.inverse(x)
    
class GNPPhases:
    """
    Simulates spectroscopic data for material compositions based on pre-defined grids.

    Parameters
    ----------
    dir : str
        Path to the directory containing grid and spectral data.

    Methods
    -------
    simulate(c)
        Simulates a spectrum for a given composition `c`.
    generate()
        Generates spectroscopic data for all compositions in the grid.
    """
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
        """
        Simulate a spectrum for a given composition.

        Parameters
        ----------
        c : numpy.ndarray
            Composition array.

        Returns
        -------
        numpy.ndarray
            Simulated spectral data.
        """
        rid = np.random.choice(len(self.spectra_files))
        lookup_dist = cdist(c.reshape(1,-1), self.points)
        lookup_cid = np.argmin(lookup_dist)
        y = self.spectra_files[rid].iloc[:,lookup_cid+1].values.astype('double')

        return y 
    
    def generate(self):
        """
        Generate spectra for all compositions in the grid.

        Returns
        -------
        None
        """
        self.F = [self.simulate(ci) for ci in self.points] 
        self.comps = self.points 
        self.spectra = np.asarray(self.F)

        return

class UVVisExperiment:
    """
    Facilitates interaction between UV-Vis spectroscopic data and active phase map models.

    Parameters
    ----------
    bounds : list of tuple
        Bounds for the compositional space.
    direc : str
        Directory containing experimental UV-Vis data.

    Methods
    -------
    read_iter_data(iter)
        Loads compositional and spectral data from specified iterations.
    generate(use_spline=False)
        Processes and normalizes spectral data.
    normalize(f)
        Normalizes a spectrum.
    plot(ax, bounds)
        Visualizes spectral data with compositional bounds.
    spline_interpolate(wl_, y)
        Interpolates spectral data using splines.
    """
    def __init__(self, bounds, direc):
        self.dim = len(bounds)
        self.bounds = torch.tensor(bounds).transpose(-1, -2).to(device)
        self.dir = direc

    def read_iter_data(self, iter): 
        """
        Load compositional and spectral data from multiple iterations.

        Parameters
        ----------
        iter : int
            Number of iterations to load data from.

        Returns
        -------
        None
        """
        comps, spectra = [], []
        for k in range(iter):
            comps.append(np.load(self.dir+'comps_%d.npy'%k).astype(np.double))
            spectra.append(np.load(self.dir+'spectra_%d.npy'%k))
            print('Loading data from iteration %d with shapes:'%k, comps[k].shape, spectra[k].shape)
        self.comps = np.vstack(comps)
        self.points = self.comps
        self.spectra = np.vstack(spectra)
        self.wav = np.load(self.dir+'wav.npy')

    def generate(self, use_spline=False):
        """
        Normalize and process spectral data.

        Parameters
        ----------
        use_spline : bool, optional
            Whether to use spline interpolation (default is False).

        Returns
        -------
        None
        """
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
        """
        Normalize a spectrum using its L2 norm.

        Parameters
        ----------
        f : numpy.ndarray
            Input spectrum, a 1D array of intensity values.

        Returns
        -------
        numpy.ndarray
            Normalized spectrum where the integral of the squared values is 1.

        Notes
        -----
        - The normalization is performed using the L2 norm, calculated with the trapezoidal rule.

        """
        norm = np.sqrt(np.trapz(f**2, self.wav))

        return f/norm 

    def plot(self, ax, bounds):
        """
        Visualize spectral data with compositional bounds or wavelength-intensity plots.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axis object to plot the data.
        bounds : list of tuple
            Compositional bounds for the visualization. If `len(bounds) == 2`, the spectra
            will be plotted within a 2D composition space. Otherwise, wavelength-intensity
            plots will be shown.

        Returns
        -------
        matplotlib.axes.Axes
            The axis object with the plotted data.

        Notes
        -----
        - Inset spectra are displayed for 2D composition plots.
        - If bounds do not have 2 elements, a standard wavelength-intensity plot is generated.

        """
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

    def _inset_spectra(self, c, t, ft, ax, uniform_yscale=None, **kwargs):
        """
        Add an inset plot of a spectrum at a specific location in a composition plot.

        Parameters
        ----------
        c : numpy.ndarray
            Normalized composition coordinates, a 1D array with two elements.
        t : numpy.ndarray
            Time or wavelength coordinates for the spectrum.
        ft : numpy.ndarray
            Intensity values of the spectrum.
        ax : matplotlib.axes.Axes
            Matplotlib axis object for the main plot.
        uniform_yscale : tuple of float, optional
            Fixed y-axis limits for the inset plots. If `None`, each inset is scaled individually.
        **kwargs : dict
            Additional keyword arguments for the plot.

        Returns
        -------
        None

        Notes
        -----
        - Inset plots are positioned based on normalized coordinates `c`.
        - Each inset plot is scaled or unscaled based on `uniform_yscale`.

        """
        loc_ax = ax.transLimits.transform(c)
        ins_ax = ax.inset_axes([loc_ax[0],loc_ax[1],0.1,0.1])
        ins_ax.plot(t, ft, **kwargs)
        ins_ax.axis('off')
        if uniform_yscale is not None:
            ins_ax.set_ylim([uniform_yscale[0], uniform_yscale[1]])
        
        return 

    def spline_interpolate(self, wl_, y):
        """
        Interpolate spectral data using cubic splines.

        Parameters
        ----------
        wl_ : numpy.ndarray
            Target wavelengths for interpolation.
        y : numpy.ndarray
            Intensity values at the original wavelengths.

        Returns
        -------
        numpy.ndarray
            Interpolated intensity values at the target wavelengths.

        Notes
        -----
        - The method uses `scipy.interpolate.splrep` and `splev` for spline interpolation.
        - The smoothing parameter `s=0` ensures an exact fit to the input data.

        """
        spline = interpolate.splrep(self.wav, y, s=0)
        I_grid = interpolate.splev(wl_, spline, der=0)

        return I_grid