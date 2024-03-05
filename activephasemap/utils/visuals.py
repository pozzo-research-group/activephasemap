import torch 
import numpy as np 
from scipy import stats
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colormaps 
from matplotlib.cm import ScalarMappable
RNG = np.random.default_rng()
from activephasemap.np.utils import context_target_split 
from botorch.utils.transforms import normalize, unnormalize
from autophasemap import compute_elastic_kmeans, BaseDataSet, compute_BIC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def from_comp_to_spectrum(test_function, gp_model, np_model, c):
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

def get_twod_grid(n_grid, bounds):
    x = np.linspace(bounds[0,0],bounds[1,0], n_grid)
    y = np.linspace(bounds[0,1],bounds[1,1], n_grid)
    X,Y = np.meshgrid(x,y)
    points = np.vstack([X.ravel(), Y.ravel()]).T 

    return points

# plot samples in the composition grid of p(y|c)
def _inset_spectra(c, t, mu, sigma, ax, show_sigma=False, **kwargs):
        loc_ax = ax.transLimits.transform(c)
        ins_ax = ax.inset_axes([loc_ax[0],loc_ax[1],0.1,0.1])
        ins_ax.plot(t, mu, **kwargs)
        if show_sigma:
            ins_ax.fill_between(t,mu-sigma, mu+sigma,
            alpha=0.2, color='grey')
        ins_ax.axis('off')
        
        return 

class MinMaxScaler:
    def __init__(self, min, max):
        self.min = min 
        self.max = max 
        self.range = max-min

    def transform(self, x):
        return (x-self.min)/self.range
    
    def inverse(self, xt):
        return (self.range*xt)+self.min

def scaled_tickformat(scaler, x, pos):
    return '%.1f'%scaler.inverse(x)

def plot_gpmodel_grid(ax, test_function, gp_model, np_model,num_grid_spacing=10, **kwargs):
    bounds = test_function.bounds.cpu().numpy()
    c1 = np.linspace(bounds[0,0], bounds[1,0], num=num_grid_spacing)
    c2 = np.linspace(bounds[0,1], bounds[1,1], num=num_grid_spacing)
    scaler_x = MinMaxScaler(bounds[0,0], bounds[1,0])
    scaler_y = MinMaxScaler(bounds[0,1], bounds[1,1])
    ax.xaxis.set_major_formatter(lambda x, pos : scaled_tickformat(scaler_x, x, pos))
    ax.yaxis.set_major_formatter(lambda y, pos : scaled_tickformat(scaler_y, y, pos))
    with torch.no_grad():
        for i in range(10):
            for j in range(10):
                ci = np.array([c1[i], c2[j]]).reshape(1, 2)
                mu, sigma = from_comp_to_spectrum(test_function, gp_model, np_model, ci)
                mu_ = mu.cpu().squeeze().numpy()
                sigma_ = sigma.cpu().squeeze().numpy()
                norm_ci = np.array([scaler_x.transform(c1[i]), scaler_y.transform(c2[j])])
                _inset_spectra(norm_ci, test_function.sim.t, mu_, sigma_, ax, **kwargs)
    ax.set_xlabel('C1', fontsize=20)
    ax.set_ylabel('C2', fontsize=20)

    return  

def plot_experiment(t, bounds, data):
    fig, ax = plt.subplots()
    scaler_x = MinMaxScaler(bounds[0][0], bounds[0][1])
    scaler_y = MinMaxScaler(bounds[1][0], bounds[1][1])
    ax.xaxis.set_major_formatter(lambda x, pos : scaled_tickformat(scaler_x, x, pos))
    ax.yaxis.set_major_formatter(lambda y, pos : scaled_tickformat(scaler_y, y, pos))
    for ci, si in zip(data.x.cpu().numpy(), data.y.cpu().numpy()):
        norm_ci = np.array([scaler_x.transform(ci[0]), scaler_y.transform(ci[1])])
        _inset_spectra(norm_ci, t, si,[], ax, show_sigma=False)
    ax.set_xlabel('C1', fontsize=20)
    ax.set_ylabel('C2', fontsize=20) 
    ax.spines[['right', 'top']].set_visible(False)

    return 

def plot_iteration(query_idx, test_function, train_x, gp_model, np_model, acquisition, z_dim):
    layout = [['A1','A2', 'C', 'C'], 
              ['B1', 'B2', 'C', 'C']
              ]
    
    # plot selected points
    C_train = test_function.sim.points
    C_grid = get_twod_grid(20, test_function.bounds.cpu().numpy())
    fig, axs = plt.subplot_mosaic(layout, figsize=(4*4, 4*2))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    x_ = train_x.cpu().numpy()
    axs['A1'].scatter(x_[:,0], x_[:,1], marker='x', color='k')
    axs['A1'].set_xlabel('C1', fontsize=20)
    axs['A1'].set_ylabel('C2', fontsize=20)    
    axs['A1'].set_title('C sampling')

    # plot acqf
    normalized_C_grid = normalize(torch.tensor(C_grid).to(train_x), test_function.bounds.to(train_x))
    with torch.no_grad():
        acq_values = acquisition(normalized_C_grid.reshape(len(C_grid),1,2)).cpu().numpy()
    cmap = colormaps["magma"]
    norm = Normalize(vmin=min(acq_values), vmax = max(acq_values))
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    axs['A2'].tricontourf(C_grid[:,0], C_grid[:,1], acq_values, cmap=cmap, norm=norm)
    divider = make_axes_locatable(axs["A2"])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.set_ylabel('Acqusition value')
    axs['A2'].set_title('utility')
    axs['A2'].set_xlabel('C1', fontsize=20)
    axs['A2'].set_ylabel('C2', fontsize=20) 

    with torch.no_grad():
        for _ in range(5):
            c_dim = C_train.shape[1]
            ci = RNG.choice(C_train).reshape(1, c_dim)
            mu, _ = from_comp_to_spectrum(test_function, gp_model, np_model, ci)
            t_ = test_function.sim.t
            axs['B2'].plot(t_, mu.cpu().squeeze(), color='grey')
            axs['B2'].set_title('random sample p(y|c)')
            axs['B2'].set_xlabel('t', fontsize=20)
            axs['B2'].set_ylabel('f(t)', fontsize=20) 

            z_sample = torch.randn((1, z_dim)).to(device)
            t = torch.from_numpy(t_)
            t = t.view(1, t_.shape[0], 1).to(device)
            mu, _ = np_model.xz_to_y(t, z_sample)
            axs['B1'].plot(t_, mu.cpu().squeeze(), color='grey')
            axs['B1'].set_title('random sample p(y|z)')
            axs['B1'].set_xlabel('t', fontsize=20)
            axs['B1'].set_ylabel('f(t)', fontsize=20) 

    plot_gpmodel_grid(axs['C'], test_function, gp_model, np_model, show_sigma=False)

    return 

def plot_gpmodel(test_function, gp_model, np_model, fname):
    # plot comp to z model predictions and the GP covariance
    z_dim = np_model.z_dim
    fig, axs = plt.subplots(2,z_dim*2, figsize=(4*z_dim*2, 4*2))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    C_train = test_function.sim.points
    y_train = np.asarray(test_function.sim.F)
    t_ = test_function.sim.t
    n_train = len(C_train)
    with torch.no_grad():
        c = torch.tensor(C_train).to(device)
        normalized_c = normalize(c, test_function.bounds.to(device))
        posterior = gp_model.posterior(normalized_c)
        z_pred = posterior.mean.cpu().numpy()

        t = torch.from_numpy(t_)
        t = t.repeat(n_train, 1).to(device)
        y =  torch.from_numpy(y_train).to(device)
        z_true_mu, _ = np_model.xy_to_mu_sigma(t.unsqueeze(2),y.unsqueeze(2))
        z_true_mu = z_true_mu.cpu().numpy()

        # compare z values from GP and NP models
        for i in range(z_dim):
            sns.kdeplot(z_true_mu[:,i], ax=axs[0,i], fill=True, label='NP Model')
            sns.kdeplot(z_pred[:,i], ax=axs[0,i],fill=True, label='GP Model')
            axs[0,i].set_xlabel('z_%d'%(i+1)) 
            axs[0,i].legend()

        # plot the covariance matrix      
        X,Y = np.meshgrid(np.linspace(min(C_train[:,0]),max(C_train[:,0]),10), 
        np.linspace(min(C_train[:,1]),max(C_train[:,1]),10))
        c_grid_np = np.vstack([X.ravel(), Y.ravel()]).T 
        c_grid = torch.tensor(c_grid_np).to(device)
        # plot covariance of randomly selected points
        idx = RNG.choice(range(n_train),size=z_dim, replace=False)  
        for i, id_ in enumerate(idx):
            ci = C_train[id_,:].reshape(1, 2)
            ci = torch.Tensor(ci).to(device)
            Ki = gp_model.get_covaraince(ci, c_grid)
            axs[1,i].tricontourf(c_grid_np[:,0], c_grid_np[:,1], Ki, cmap='plasma')
            axs[1,i].scatter(C_train[id_,0], C_train[id_,1], marker='x', s=50, color='k')
            axs[1,i].set_xlabel('C1')
            axs[1,i].set_ylabel('C2')    

        # plot predicted z values as contour plots
        for i in range(z_dim):
            norm=plt.Normalize(z_pred[:,i].min(),z_pred[:,i].max())
            axs[1,z_dim+i].tricontourf(C_train[:,0], C_train[:,1], 
            z_pred[:,i], cmap='bwr', norm=norm)        
            axs[1,z_dim+i].set_xlabel('C1')
            axs[1,z_dim+i].set_ylabel('C2') 
            axs[1,z_dim+i].set_title('Predicted z_%d'%(i+1))

        # plot true z values as contour plots
        for i in range(z_dim):
            norm=plt.Normalize(z_true_mu[:,i].min(),z_true_mu[:,i].max())
            axs[0,z_dim+i].tricontourf(C_train[:,0], C_train[:,1], z_true_mu[:,i], cmap='bwr', norm=norm)        
            axs[0,z_dim+i].set_xlabel('C1')
            axs[0,z_dim+i].set_ylabel('C2') 
            axs[0,z_dim+i].set_title('True z_%d'%(i+1))        

        plt.savefig(fname)
        plt.close()        
    return 

# plot phase map predition

def plot_gpmodel_recon(ax, gp_model, np_model, test_function, c):
    with torch.no_grad():
        mu, sigma = from_comp_to_spectrum(test_function, gp_model, np_model, c)
        mu_ = mu.cpu().squeeze()
        sigma_ = sigma.cpu().squeeze()
        ax.plot(test_function.sim.t, mu_, label="GP pred.")
        ax.fill_between(test_function.sim.t,mu_-sigma_,mu_+sigma_,
        alpha=0.2, color='grey', label="GP Unc.")

    return 

def plot_npmodel_recon(ax, np_model, x, y):
    xt = torch.from_numpy(x.reshape(1,len(x),1)).to(device)
    yt =  torch.from_numpy(y.reshape(1,len(x),1)).to(device)
    z_true_mu, z_true_sigma = np_model.xy_to_mu_sigma(xt,yt)
    mu, std = np_model.xz_to_y(xt, z_true_mu)
    mu_ = mu.cpu().squeeze()
    sigma_ = std.cpu().squeeze()
    ax.plot(x, mu_, label='NP pred.')
    ax.fill_between(x,mu_-sigma_, mu_+sigma_,alpha=0.2, color='grey', label="NP Unc.")    

    return 

def plot_npmodel_recon_sample(ax, np_model, x, y):
    xt = torch.from_numpy(x.reshape(1,len(x),1)).to(device)
    yt =  torch.from_numpy(y.reshape(1,len(x),1)).to(device)
    x_context, y_context, _, _ = context_target_split(xt, yt, 25, 25)
    ax.scatter(x_context.squeeze().cpu().numpy(), y_context.squeeze().cpu().numpy(), c='tab:red')
    ax.plot(x, y, color='tab:red', lw=1.0, label='Data')
    for i in range(200):
        # Neural process returns distribution over y_target
        p_y_pred = np_model(x_context, y_context, xt)
        ax.plot(x,p_y_pred.loc.cpu().numpy()[0], alpha=0.05, c='b') 

    return 

def plot_phasemap_pred(test_function, gp_model, np_model, fname):
    c_dim = test_function.sim.points.shape[1]
    with torch.no_grad():
        idx = RNG.choice(range(len(test_function.sim.points)),size=3, replace=False)
        # plot comparision of predictions with actual
        fig, axs = plt.subplots(2,3, figsize=(4*3, 4*2))
        for i, id_ in enumerate(idx):
            ci = test_function.sim.points[id_,:].reshape(1, c_dim)        
            plot_gpmodel_recon(axs[0,i], gp_model, np_model, test_function, ci)
            axs[0, i].scatter(test_function.sim.t, test_function.sim.F[id_], color='k', label="Data")
            plot_npmodel_recon_sample(axs[1,i], np_model, test_function.sim.t, test_function.sim.F[id_])
            for j in [0,1]:
                axs[j, i].legend()
        plt.savefig(fname)
        plt.close() 


""" Visualization tools customized for experimental campaigns """ 
def plot_gpmodel_expt(test_function, gp_model, np_model, fname):
    # plot comp to z model predictions and the GP covariance
    z_dim = np_model.z_dim
    fig, axs = plt.subplots(2,z_dim, figsize=(4*z_dim, 4*2))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    C_train = test_function.sim.points
    y_train = np.asarray(test_function.sim.F)
    t_ = test_function.sim.t
    n_train = len(C_train)
    X,Y = np.meshgrid(np.linspace(min(C_train[:,0]),max(C_train[:,0]),10), 
    np.linspace(min(C_train[:,1]),max(C_train[:,1]),10))
    c_grid_np = np.vstack([X.ravel(), Y.ravel()]).T 
    c_grid = torch.tensor(c_grid_np).to(device)    
    with torch.no_grad():
        # predict z distribution of p(z|c) approximated by GP
        normalized_c_grid = normalize(c_grid, test_function.bounds.to(device))
        posterior = gp_model.posterior(normalized_c_grid)
        z_pred = posterior.mean.cpu().numpy() 

        t = torch.from_numpy(t_)
        t = t.repeat(n_train, 1).to(device)
        y =  torch.from_numpy(y_train).to(device)
        # predict z distribution estimated by NP 
        z_true_mu, z_true_sigma = np_model.xy_to_mu_sigma(t.unsqueeze(2),y.unsqueeze(2))
        z_true_mu = z_true_mu.cpu().numpy()
        z_true_sigma = z_true_sigma.cpu().numpy() 

        # compare z values from GP and NP models
        for i in range(z_dim):
            sns.kdeplot(z_true_mu[:,i], ax=axs[0,i], fill=True, label='NP Model')
            sns.kdeplot(z_pred[:,i], ax=axs[0,i],fill=True, label='GP Model')
            axs[0,i].set_xlabel('z_%d'%(i+1)) 
            axs[0,i].legend()

        # plot covariance of randomly selected points
        idx = RNG.choice(range(n_train),size=z_dim, replace=False)  
        for i, id_ in enumerate(idx):
            ci = C_train[id_,:].reshape(1, 2)
            ci = torch.tensor(ci).to(device)
            Ki = gp_model.get_covaraince(ci, c_grid)
            axs[1,i].tricontourf(c_grid_np[:,0], c_grid_np[:,1], Ki, cmap='plasma')
            axs[1,i].scatter(C_train[id_,0], C_train[id_,1], marker='x', s=50, color='k')
            axs[1,i].set_xlabel('C1')
            axs[1,i].set_ylabel('C2')    

        plt.savefig(fname)
        plt.close()        
    return 


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

def plot_autophasemap(pbp, fname):
    scaler_x = MinMaxScaler(bounds[0,0], bounds[1,0])
    scaler_y = MinMaxScaler(bounds[0,1], bounds[1,1])

    fig, axs = plt.subplots(1,2, figsize=(2*4, 4))
    axs[0].plot(pbp.sweep_n_clusters, pbp.BIC, marker='o')
    axs[0].axvline(x = pbp.sweep_n_clusters[pbp.min_bic_clusters], ls='--', color='tab:red')

    ax = axs[1]
    cmap = plt.get_cmap("tab10") 
    ax.xaxis.set_major_formatter(lambda x, pos : scaled_tickformat(scaler_x, x, pos))
    ax.yaxis.set_major_formatter(lambda y, pos : scaled_tickformat(scaler_y, y, pos))
    for i in range(pbp.grid_comps.shape[0]):
        ci = np.array([scaler_x.transform(pbp.grid_comps[i,0]), scaler_y.transform(pbp.grid_comps[i,1])])
        _inset_spectra(ci, pbp.test_function.sim.t, pbp.grid_spectra[i,:], [], ax, 
        color=cmap(pbp.out.delta_n[i]))
    ax.set_xlabel('C1', fontsize=20)
    ax.set_ylabel('C2', fontsize=20)    
    plt.savefig(fname)
    plt.close() 

    return 
    