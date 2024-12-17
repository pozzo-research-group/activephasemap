from abc import ABC, abstractmethod
import torch 
import gpytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)
from botorch.utils.transforms import normalize
import pdb, time, datetime
from typing import Union

class BaseAcquisiton(torch.nn.Module):
    def __init__(self, expt, bounds, z2y, c2z, **kwargs):
        super().__init__()
        self.expt = expt
        self.z_to_y = z2y 
        self.c_to_z = c2z 
        self.bounds = bounds
        self.input_dim = len(bounds)
        self.nz = kwargs.get("num_z_sample", 128) 

        self.train_x = torch.from_numpy(expt.comps).to(device)
        self.train_y = torch.from_numpy(expt.spectra_normalized).to(device)


    @abstractmethod
    def forward(self, x):
        """
        This method must be implemented by any subclass of BaseClass.
        """
        pass

    def optimize(self, batch_size, num_restarts=8, n_iterations=200):
        X = torch.rand(num_restarts, batch_size, len(self.bounds)).to(device)
        X = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * X
        X.requires_grad_(True)
        optimizer = torch.optim.Adam([X], lr=0.1)

        start = time.time()
        for i in range(n_iterations):
            optimizer.zero_grad()
            acqv = -self(X)
            loss = acqv.sum()
            loss.backward() 
            optimizer.step()

            # clamp values to the feasible set
            for j, (lb, ub) in enumerate(zip(*self.bounds)):
                X.data[..., j].clamp_(lb, ub)  # need to do this on the data not X itself

            if (i + 1) % 50 == 0:
                end = time.time()
                time_str =  str(datetime.timedelta(seconds=end-start))
                print(f"({time_str:>s}) Iteration {i+1:>3}/{n_iterations:>3} - Loss: {loss.item():>4.3f}")
        
        return X[acqv.sum(dim=1).argmin(),...].clone().detach()

class CompositeModelUncertainity(BaseAcquisiton):
    def __init__(self, expt, bounds, NP, MLP, **kwargs):
        super().__init__(expt, bounds, NP, MLP, **kwargs)

    def forward(self, x):
        z_mu, z_std = self.c_to_z.mlp(x)       
        nr, nb, d = z_mu.shape
        z_dist = torch.distributions.Normal(z_mu, z_std)
        z = z_dist.rsample(torch.Size([self.nz])).view(self.nz*nr*nb, d)
        t = torch.from_numpy(self.expt.t).repeat(self.nz*nr*nb, 1, 1).to(device)
        t = torch.swapaxes(t, 1, 2)
        y_samples, _ = self.z_to_y.xz_to_y(t, z)
        sigma_pred = y_samples.view(self.nz, nr, nb, len(self.t), 1).std(dim=0)
        acqv = sigma_pred.mean(dim=-2).squeeze(-1)

        return acqv

class XGBUncertainity(BaseAcquisiton):
    """
    Uncertainty-Weighted Exploration:

        This method incorporates both the residuals (r(x)) and model uncertainty (sigma(x)) to guide active learning
        in regression tasks. The goal is to explore regions where the model's predictions exhibit significant uncertainty
        and where the model's predictions are likely to have high error.

        The acquisition function is defined as:
            Acquisition(x) = alpha*r(x) + (1-alpha) * sigma(x) alpha \in [0, 1]
        
        Where:
            - r(x) is the residual at point x scaled by sigma(x), estimated using Kernel Density Estimation (KDE) 
              based on the residuals from the training set.
            - sigma(x) is the model's predicted uncertainty (standard deviation) at point x.
            - beta is a scaling factor that balances the contribution of residuals and uncertainty in the acquisition function.

        The residuals are min-max normalized to allow acquisiton function optimization to fairly optimize 
        both the components.

        The method prioritizes sampling at points where either:
            1. The residual (r(x)) indicates high prediction error at known locations, or
            2. The uncertainty (sigma(x)) is high, suggesting areas where the model is less confident.

        This acquisition function encourages exploration in regions of the input space where the model is uncertain 
        and where the residuals indicate that further learning could significantly improve the model's performance.
    """

    def __init__(self, expt, bounds, NP, XGB, **kwargs):
        super().__init__(expt, bounds, NP, XGB, **kwargs)

    def c2y(self, x):
        z_mu, z_std = self.c_to_z.predict(x)   
        nr, nb, dz = z_mu.shape
        z_dist = torch.distributions.Normal(z_mu, z_std)
        z = z_dist.rsample(torch.Size([self.nz])).view(self.nz*nr*nb, dz)
        t = torch.from_numpy(self.expt.t).repeat(self.nz*nr*nb, 1, 1).to(device)
        t = torch.swapaxes(t, 1, 2)
        y_samples, _ = self.z_to_y.xz_to_y(t, z)

        return y_samples

    def _min_max_normalize(self, y: torch.Tensor):
        """Normalize a vector of shape (n_samples, ) to [0,1]
        """
        y_min = torch.min(y)
        y_max = torch.max(y)
        y_norm = (y - y_min) / (y_max - y_min + 1e-8)

        return y_norm

    def forward(self, x, alpha=0.5, return_rx_sigma = False):
        nr, nb, dx = x.shape

        # Compute uncertainity of spectrum prediction
        y_samples = self.c2y(x)
        mu_pred = y_samples.view(self.nz, nr, nb, len(self.expt.t), 1).mean(dim=0)
        sigma_pred = y_samples.view(self.nz, nr, nb, len(self.expt.t), 1).std(dim=0)
        sigma_x = (sigma_pred/mu_pred).mean(dim=-2).squeeze()

        # Compute error distribition on spectrum prediction
        y_samples_train = self.c2y(self.train_x.unsqueeze(-2))
        mu_pred_train = y_samples_train.view(self.nz, self.train_x.shape[0], len(self.expt.t)).mean(dim=0)
        sigma_pred_train = y_samples_train.view(self.nz, self.train_x.shape[0], len(self.expt.t)).std(dim=0)
        res = (self.train_y-mu_pred_train)/(sigma_pred_train+1e-8)
        res_ = torch.mean(torch.abs(res), dim=1)
        kde = KDEResidualEstimator(self.train_x, res_)
        rx = kde(x.view(nr*nb, dx))

        rx_norm = self._min_max_normalize(rx)
        sigma_x_norm = self._min_max_normalize(sigma_x.flatten())

        T1 = alpha*rx_norm.view(nr, nb)
        T2 = (1-alpha)*sigma_x_norm.view(nr, nb)

        if not return_rx_sigma:
            return T1+T2 
        else:
            return res_, T1, T2

class KDEResidualEstimator(torch.nn.Module):
    """Kernel Density Estimate (KDE) for smoothing residuals with automatic bandwidth selection.

    Attributes:
        X (torch.Tensor): Sampled points of shape (n_samples, dim).
        residuals (torch.Tensor): Residual values at sampled points of shape (n_samples,).
        bandwidth (float): The computed bandwidth for kernel density estimation.
        inv_bandwidth_sq (float): The inverse of the squared bandwidth (used for efficiency in kernel computation).

    Methods:
        __init__(self, X: torch.Tensor, residuals: torch.Tensor, bandwidth: str = "scott"):
            Initializes the KDE residual estimator with given data, residuals, and bandwidth selection method.
        
        _compute_bandwidth(self, method: str) -> float:
            Computes the bandwidth using Scott's or Silverman's rule based on the given method.
        
        estimate(self, x: torch.Tensor) -> torch.Tensor:
            Estimates the residual at a given location using kernel density estimation.
            
        Bandwidth Calculation Methods:
            Scott's Rule: h = sigma * n^(-1 / (d + 4))
            Silverman's Rule: h = 0.9 * min(sigma, IQR / 1.34) * n^(-1 / (d + 4))
    """
    def __init__(self, X: torch.Tensor, residuals: torch.Tensor, bandwidth: Union[str, float] = "scott"):
        super().__init__()
        self.X = X  # Shape: (n_samples, dim)
        self.residuals = residuals  # Shape: (n_samples,)
        self.bandwidth = self._compute_bandwidth(bandwidth)  # Auto bandwidth
        self.inv_bandwidth_sq = 1.0 / (2.0 * self.bandwidth**2)

    def _compute_bandwidth(self, method: Union[str, float]) -> float:
        """
        Compute bandwidth using Scott's or Silverman's rule.

        Args:
            method (str or float): "scott" or "silverman" or a value.

        Returns:
            float: Bandwidth value.
        """
        n_samples, dim = self.X.shape
        std_devs = self.X.std(dim=0)  # Standard deviations along each dimension
        if method == "scott":
            bandwidth = torch.mean(std_devs) * (n_samples ** (-1 / (dim + 4)))
        elif method == "silverman":
            iqr = torch.subtract(*torch.quantile(self.X, 
                                                 torch.tensor([0.75, 0.25], dtype=torch.double, requires_grad=True), 
                                                 dim=0
                                                 )
                                )
            sigma = torch.min(std_devs, iqr / 1.34)
            bandwidth = 0.9 * sigma * (n_samples ** (-1 / (dim + 4)))
        elif type(method)==float:
            bandwidth = torch.tensor([method], dtype=torch.double, requires_grad=True).to(device)
        else:
            raise ValueError("Invalid bandwidth method. Choose 'scott' or 'silverman' or a value.")

        return bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate the residual at a given location.

        Args:
            x (torch.Tensor): Query points (m_samples, dim).

        Returns:
            torch.Tensor: Estimated residuals (m_samples,).
        """
        # Compute pairwise squared distances between x and self.X
        pairwise_distances = torch.cdist(x, self.X, p=2)**2  # Shape: (m_samples, n_samples)

        # Compute Gaussian kernel weights
        kernel_weights = torch.exp(-pairwise_distances * self.inv_bandwidth_sq)  # Shape: (m_samples, n_samples)

        # Weighted residuals
        weighted_residuals = kernel_weights * self.residuals.unsqueeze(0)  # Shape: (m_samples, n_samples)

        # Normalized estimate
        residual_estimates = weighted_residuals.sum(dim=1) / (kernel_weights.sum(dim=1) + 1e-8)

        return residual_estimates
