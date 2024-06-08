import os, shutil
import numpy as np 
RNG = np.random.default_rng()

import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colormaps 
from matplotlib.cm import ScalarMappable

import torch 
from botorch.utils.transforms import normalize
from activephasemap.models.np import context_target_split 
from activephasemap.utils.settings import from_comp_to_spectrum, get_twod_grid
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# plot samples in the composition grid of p(y|c)
def _inset_spectra(c, t, mu, sigma, ax, show_sigma=False, **kwargs):
        loc_ax = ax.transLimits.transform(c)
        ins_ax = ax.inset_axes([loc_ax[0],loc_ax[1],0.1,0.1])
        ins_ax.plot(t, mu, **kwargs)
        if show_sigma:
            ins_ax.fill_between(t,mu-sigma, mu+sigma,
            color='grey')
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

def plot_gpmodel_grid(ax, expt, gp_model, np_model,num_grid_spacing=10, **kwargs):
    bounds = expt.bounds.cpu().numpy()
    c1 = np.linspace(bounds[0,0], bounds[1,0], num=num_grid_spacing)
    c2 = np.linspace(bounds[0,1], bounds[1,1], num=num_grid_spacing)
    scaler_x = MinMaxScaler(bounds[0,0], bounds[1,0])
    scaler_y = MinMaxScaler(bounds[0,1], bounds[1,1])
    if kwargs.pop("scale_axis", True):
        ax.xaxis.set_major_formatter(lambda x, pos : scaled_tickformat(scaler_x, x, pos))
        ax.yaxis.set_major_formatter(lambda y, pos : scaled_tickformat(scaler_y, y, pos))
    with torch.no_grad():
        for i in range(num_grid_spacing):
            for j in range(num_grid_spacing):
                ci = np.array([c1[i], c2[j]]).reshape(1, 2)
                mu, sigma = from_comp_to_spectrum(expt, gp_model, np_model, ci)
                mu_ = mu.cpu().squeeze().numpy()
                sigma_ = sigma.cpu().squeeze().numpy()
                norm_ci = np.array([scaler_x.transform(c1[i]), scaler_y.transform(c2[j])])
                _inset_spectra(norm_ci, expt.t, mu_, sigma_, ax, **kwargs)
    ax.set_xlabel('C1', fontsize=20)
    ax.set_ylabel('C2', fontsize=20)

    return  

def plot_experiment(t, bounds, data):
    fig, ax = plt.subplots(figsize=(4,4))
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

def plot_iteration(query_idx, expt, train_x, gp_model, np_model, acquisition, z_dim):
    layout = [['A1','A2', 'C', 'C'], 
              ['B1', 'B2', 'C', 'C']
              ]
    
    # plot selected points
    C_train = expt.points
    bounds =  expt.bounds.cpu().numpy()
    C_grid = get_twod_grid(20, bounds)
    fig, axs = plt.subplot_mosaic(layout, figsize=(4*4, 4*2))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    if torch.is_tensor(train_x):
        x_ = train_x.cpu().numpy()
    else:
        x_ = train_x
    axs['A1'].scatter(x_[:,0], x_[:,1], marker='x', color='k')
    axs['A1'].set_xlabel('C1', fontsize=20)
    axs['A1'].set_ylabel('C2', fontsize=20)    
    axs['A1'].set_title('C sampling')
    axs['A1'].set_xlim([bounds[0,0], bounds[1,0]])
    axs['A1'].set_ylim([bounds[0,1], bounds[1,1]])

    # plot acqf
    normalized_C_grid = normalize(torch.tensor(C_grid).to(device), expt.bounds.to(device))
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
            mu, _ = from_comp_to_spectrum(expt, gp_model, np_model, ci)
            t_ = expt.t
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

    plot_gpmodel_grid(axs['C'], expt, gp_model, np_model, show_sigma=False)

    return fig, axs

def plot_gpmodel(expt, gp_model, np_model, fname):
    # plot comp to z model predictions and the GP covariance
    z_dim = np_model.z_dim
    fig, axs = plt.subplots(2,z_dim*2, figsize=(4*z_dim*2, 4*2))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    C_train = expt.points
    y_train = np.asarray(expt.F)
    t_ = expt.t
    n_train = len(C_train)
    with torch.no_grad():
        c = torch.tensor(C_train).to(device)
        normalized_c = normalize(c, expt.bounds.to(device))
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

def plot_gpmodel_recon(ax, gp_model, np_model, expt, c):
    with torch.no_grad():
        mu, sigma = from_comp_to_spectrum(expt, gp_model, np_model, c)
        mu_ = mu.cpu().squeeze()
        sigma_ = sigma.cpu().squeeze()
        ax.plot(expt.wl, mu_, label="GP pred.")
        ax.fill_between(expt.wl,mu_-sigma_,mu_+sigma_,
        color='grey', label="GP Unc.")

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

def plot_phasemap_pred(expt, gp_model, np_model, fname):
    c_dim = expt.shape[1]
    with torch.no_grad():
        idx = RNG.choice(range(len(expt.points)),size=3, replace=False)
        # plot comparision of predictions with actual
        fig, axs = plt.subplots(2,3, figsize=(4*3, 4*2))
        for i, id_ in enumerate(idx):
            ci = expt.points[id_,:].reshape(1, c_dim)        
            plot_gpmodel_recon(axs[0,i], gp_model, np_model, expt, ci)
            axs[0, i].scatter(expt.t, expt.F[id_], color='k', label="Data")
            plot_npmodel_recon_sample(axs[1,i], np_model, expt.t, expt.F[id_])
            for j in [0,1]:
                axs[j, i].legend()
        plt.savefig(fname)
        plt.close() 


""" Visualization tools customized for experimental campaigns """ 
def plot_gpmodel_expt(expt, gp_model, np_model, fname):
    # plot comp to z model predictions and the GP covariance
    z_dim = np_model.z_dim
    fig, axs = plt.subplots(2,z_dim, figsize=(4*z_dim, 4*2))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    C_train = expt.points
    y_train = np.asarray(expt.F)
    t_ = expt.t
    n_train = len(C_train)
    X,Y = np.meshgrid(np.linspace(min(C_train[:,0]),max(C_train[:,0]),10), 
    np.linspace(min(C_train[:,1]),max(C_train[:,1]),10))
    c_grid_np = np.vstack([X.ravel(), Y.ravel()]).T 
    c_grid = torch.tensor(c_grid_np).to(device)    
    with torch.no_grad():
        # predict z distribution of p(z|c) approximated by GP
        normalized_c_grid = normalize(c_grid, expt.bounds.to(device))
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

def plot_model_accuracy(direc, gp_model, np_model, expt):
    """ Plot accuract of model predictions of experimental data

    """
    num_samples, c_dim = expt.comps.shape
    if os.path.exists(direc+'preds/'):
        shutil.rmtree(direc+'preds/')
    os.makedirs(direc+'preds/')
    for i in range(num_samples):
        fig, ax = plt.subplots()
        ci = expt.comps[i,:].reshape(1, c_dim)
        plot_gpmodel_recon(ax, gp_model, np_model, expt, ci)
        ax.scatter(expt.wl, expt.F[i], color='k')
        plt.savefig(direc+'preds/%d.png'%(i))
        plt.close()
