import torch
from autophasemap import compute_elastic_kmeans, BaseDataSet, compute_BIC
from autophasemap.geometry import SquareRootSlopeFramework, WarpingManifold
from autophasemap.diffusion import DiffusionMaps
import numpy as np
from botorch.utils.transforms import normalize
from activephasemap.utils.settings import from_comp_to_spectrum, get_twod_grid


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def __init__(self, test_function, gp_model, np_model, min_clusters=2, max_clusters=5, max_iter_search=10, max_iter = 20):
        super().__init__()
        self.test_function = test_function
        self.gp_model = gp_model
        self.np_model = np_model
        self.bounds = self.test_function.bounds.cpu().numpy()
        self.grid_comps = get_twod_grid(10, bounds = self.bounds)
        n_grid_samples = self.grid_comps.shape[0]
        n_spectra_dim =  self.test_function.sim.t.shape[0]
        self.grid_spectra = np.zeros((n_grid_samples, n_spectra_dim))
        with torch.no_grad():
            for i in range(n_grid_samples):
                mu, _ = from_comp_to_spectrum(self.test_function, self.gp_model,
                self.np_model, self.grid_comps[i,:].reshape(1, 2)
                )
                self.grid_spectra[i,:] = mu.cpu().squeeze().numpy()

        self.data = AutoPhaseMapDataSet(self.grid_comps, self.test_function.sim.t, self.grid_spectra, n_domain=n_spectra_dim)
        self.data.generate()
        self.sweep_n_clusters = np.arange(min_clusters,max_clusters+1)
        self.BIC = []
        for n_clusters in self.sweep_n_clusters:
            out = compute_elastic_kmeans(self.data, n_clusters, max_iter=max_iter_search, verbose=0, smoothen=False)
            self.BIC.append(compute_BIC(self.data, out.fik_gam, out.qik_gam, out.delta_n))

        self.min_bic_clusters = self.sweep_n_clusters[np.argmin(self.BIC)]
        self.out = compute_elastic_kmeans(self.data, self.min_bic_clusters, max_iter=max_iter, verbose=0, smoothen=True)

    def get_entropy(self, c):
        SRSF = SquareRootSlopeFramework(self.data.t)
        diffmap = DiffusionMaps(self.data.C)
        print(c)
        mu, _ = from_comp_to_spectrum(self.test_function, self.gp_model, self.np_model, c.reshape(1, 2))
        F = mu.cpu().squeeze().numpy()
        q = SRSF.to_srsf(F)
        dists = np.zeros(self.min_bic_clusters)
        for k in range(self.min_bic_clusters):
            gamma = SRSF.get_gamma(self.out.templates[k], q)
            f_gam = SRSF.warp_f_gamma(F, gamma)
            q_gam = SRSF.to_srsf(f_gam)
            dists[k] = np.sqrt(np.trapz((self.out.templates[k] - q_gam)**2, self.data.t))
        
        _, _, d_smoothened = diffmap.get_asymptotic_function(dists)
        entropy = [-d*np.log(d) for d in d_smoothened]
        
        return np.sum(np.asarray(entropy).T, axis=1)

    def forward(self, X):
        _, num_samples, _ = X.shape
        entropies = torch.zeros(num_samples).to(X)
        for i in range(num_samples):
            entropies[i] = self.get_entropy(X[:,i,:])
        
        return entropies