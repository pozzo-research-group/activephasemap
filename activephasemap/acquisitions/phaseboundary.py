import torch
from autophasemap import compute_elastic_kmeans, BaseDataSet, compute_BIC
from autophasemap.geometry import SquareRootSlopeFramework, WarpingManifold
from autophasemap.diffusion import DiffusionMaps
import numpy as np
from botorch.utils.transforms import normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _from_comp_to_spectrum(test_function, gp_model, np_model, c):
    with torch.no_grad():
        t_ = test_function.sim.t
        c = torch.tensor(c).to(device)
        gp_model.eval()
        normalized_x = normalize(c, test_function.bounds.to(c))
        posterior = gp_model.posterior(normalized_x)  # based on https://github.com/pytorch/botorch/issues/1110
        t = torch.from_numpy(t_).to(device)
        t = t.repeat(c.shape[0]).view(c.shape[0], len(t_), 1)
        mu, std = np_model.xz_to_y(t, posterior.mean)

        return mu, std  

def _get_twod_grid(n_grid, bounds):
    x = np.linspace(bounds[0,0],bounds[1,0], n_grid)
    y = np.linspace(bounds[0,1],bounds[1,1], n_grid)
    X,Y = np.meshgrid(x,y)
    points = np.vstack([X.ravel(), Y.ravel()]).T 

    return points

class AutoPhaseMapDataSet(BaseDataSet):
    def __init__(self, C, q, Iq, n_domain = 100):
        super().__init__(n_domain=n_domain)
        self.t = np.linspace(0,1, num=self.n_domain)
        self.q = q
        self.N = C.shape[0]
        self.Iq = Iq
        self.C = C
    
    def generate(self, process=None):
        if process=="normalize":
            self.F = [self.Iq[i,:]/self.l2norm(self.q, self.Iq[i,:]) for i in range(self.N)]
        elif process=="smoothen":
            self.F = [self._smoothen(self.Iq[i,:]/self.l2norm(self.q, self.Iq[i,:]), window_length=7, polyorder=3) for i in range(self.N)]
        elif process is None:
            self.F = [self.Iq[i,:] for i in range(self.N)]
            
        return


class PhaseBoundaryPenalty(torch.nn.Module):
    r"""A penalty funcion based on phase boundaries to be added to any arbitrary acquisition function
    to construct a PenalizedAcquisitionFunction."""

    def __init__(self, test_function, gp_model, np_model, min_clusters=2, max_clusters=5):
        super().__init__()
        self.test_function = test_function
        self.gp_model = gp_model
        self.np_model = np_model
        bounds = self.test_function.bounds.cpu().numpy()
        self.grid_comps = _get_twod_grid(10, bounds = bounds)
        n_grid_samples = self.grid_comps.shape[0]
        n_spectra_dim =  self.test_function.sim.t.shape[0]
        grid_spectra = np.zeros((n_grid_samples, n_spectra_dim))
        with torch.no_grad():
            for i in range(n_grid_samples):
                mu, _ = _from_comp_to_spectrum(self.test_function, self.gp_model,
                self.np_model, self.grid_comps[i,:].reshape(1, 2)
                )
                grid_spectra[i,:] = mu.cpu().squeeze().numpy()

        self.data = AutoPhaseMapDataSet(self.grid_comps, self.test_function.sim.t, grid_spectra, n_domain=n_spectra_dim)
        self.data.generate()
        self.sweep_n_clusters = np.arange(min_clusters,max_clusters)
        self.BIC = []
        for n_clusters in self.sweep_n_clusters:
            out = compute_elastic_kmeans(self.data, n_clusters, max_iter=5, verbose=0, smoothen=False)
            self.BIC.append(compute_BIC(self.data, out.fik_gam, out.qik_gam, out.delta_n))

        self.min_bic_clusters = self.sweep_n_clusters[np.argmin(self.BIC)]
        self.out = compute_elastic_kmeans(self.data, self.min_bic_clusters, max_iter=10, verbose=0, smoothen=True)

    def get_entropy(self, c):
        SRSF = SquareRootSlopeFramework(self.data.t)
        diffmap = DiffusionMaps(self.data.C)
        print(c)
        mu, _ = _from_comp_to_spectrum(self.test_function, self.gp_model, self.np_model, c.reshape(1, 2))
        F = mu.cpu().squeeze().numpy()
        q = SRSF.to_srsf(F)
        dists = np.zeros(self.min_bic_clusters)
        for k in range(self.min_bic_clusters):
            gamma = SRSF.get_gamma(self.out.templates[k], q)
            f_gam = SRSF.warp_f_gamma(F, gamma)
            q_gam = SRSF.to_srsf(f_gam)
            dists[k] = np.sqrt(np.trapz((self.out.templates[k] - q_gam)**2, self.data.t))
        
        s_norm, s_hat, d_smoothened = diffmap.get_asymptotic_function(dists)
        entropy = [-d*np.log(d) for d in d_smoothened]
        
        return np.sum(np.asarray(entropy).T, axis=1)

    def forward(self, X):
        _, num_samples, c_dim = X.shape
        entropies = torch.zeros(num_samples).to(X)
        for i in range(num_samples):
            entropies[i] = self.get_entropy(X[:,i,:])
        
        return entropies