import os, shutil, json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)

from botorch.utils.sampling import draw_sobol_samples

from activephasemap.models.np import NeuralProcess
from activephasemap.models.utils import finetune_neural_process
from activephasemap.models.xgb import XGBoost
from activephasemap.models.acquisition import XGBUncertainity
from activephasemap.simulators import  MinMaxScaler, scaled_tickformat
from apdist.distances import AmplitudePhaseDistance as dist

RNG = np.random.default_rng()

from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colormaps 
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches

import importlib.resources as pkg_resources

PRETRAIN_LOC = "./pretrained/model.pt"

with pkg_resources.open_text("activephasemap.pretrained", "best_config.json") as file:    
    best_np_config = json.load(file)
N_LATENT = best_np_config["z_dim"]
N_Z_DRAWS = 256
xgb_model_args = {"objective": "reg:squarederror",
                  "max_depth": 3,
                  "eta": 0.1,
                  "eval_metric": "rmse"
                  }

np_model_args = {"num_iterations": 1000, 
                 "verbose":250, 
                 "lr":best_np_config["lr"], 
                 "batch_size": best_np_config["batch_size"]
                 }

def initialize_points(bounds, n_init_points, device):
    """
    Initialize a set of points using Sobol sequences within the specified bounds.

    Parameters
    ----------
    bounds : torch.Tensor
        The bounds within which the initial points should be generated.
    n_init_points : int
        The number of initial points to generate.
    device : torch.device
        The device (CPU or GPU) where the points will be stored.

    Returns
    -------
    torch.Tensor
        The initialized points.
    """
    if n_init_points < 1:
        init_x = torch.zeros(1, 1).to(device)
    else:
        bounds = bounds.to(device, dtype=torch.double)
        init_x = draw_sobol_samples(bounds=bounds, n=n_init_points, q=1).squeeze(-2)

    return init_x

def get_twod_grid(n_grid, bounds):
    """
    Generate a 2D grid of points within the specified bounds.

    Parameters
    ----------
    n_grid : int
        The number of grid points along each dimension.
    bounds : np.ndarray
        The bounds of the grid in 2D space.

    Returns
    -------
    np.ndarray
        A 2D array of points in the grid.
    """
    x = np.linspace(bounds[0,0],bounds[1,0], n_grid)
    y = np.linspace(bounds[0,1],bounds[1,1], n_grid)
    X,Y = np.meshgrid(x,y)
    points = np.vstack([X.ravel(), Y.ravel()]).T 

    return points 

def _inset_spectra(c, t, mu, sigma, ax, show_sigma=False, uniform_yscale = None, **kwargs):
    """
    Create an inset plot for the spectra at a given location on the main plot.

    Parameters
    ----------
    c : tuple
        The coordinates where the inset plot will be positioned.
    t : np.ndarray
        The time or wavelength values.
    mu : np.ndarray
        The mean values of the spectra.
    sigma : np.ndarray
        The standard deviation values of the spectra.
    ax : matplotlib.axes.Axes
        The main axes of the plot to which the inset will be added.
    show_sigma : bool, optional
        Whether to show the sigma as a shaded region, by default False.
    uniform_yscale : tuple, optional
        The y-axis limits for the inset plot, by default None.

    Returns
    -------
    None
    """
    loc_ax = ax.transLimits.transform(c)
    ins_ax = ax.inset_axes([loc_ax[0],loc_ax[1],0.1,0.1])
    ins_ax.plot(t, mu, **kwargs)
    if show_sigma:
        ins_ax.fill_between(t,mu-sigma, mu+sigma,
        color='grey')
    ins_ax.axis('off')
    if uniform_yscale is not None:
        ins_ax.set_ylim([uniform_yscale[0], uniform_yscale[1]])
    
    return 

@torch.no_grad
def print_matrix(A):
    """
    Print the matrix A in a human-readable format.

    Parameters
    ----------
    A : torch.Tensor
        The matrix to be printed.

    Returns
    -------
    None
    """
    A = pd.DataFrame(A.cpu().numpy())
    A.columns = ['']*A.shape[1]
    print(A.to_string())

""" Helper functions """
@torch.no_grad()
def featurize_spectra(np_model, comps_all, spectra_all):
    """
    Generate latent space embeddings from spectra using a Neural Process model.

    Parameters
    ----------
    np_model : NeuralProcess
        The trained Neural Process model.
    comps_all : np.ndarray
        The compositions of the samples.
    spectra_all : np.ndarray
        The spectra of the samples.

    Returns
    -------
    torch.Tensor
        The compositions in the latent space.
    torch.Tensor
        The mean values of the latent space representations.
    torch.Tensor
        The standard deviation values of the latent space representations.
    """
    num_samples, n_domain = spectra_all.shape
    spectra = torch.zeros((num_samples, n_domain)).to(device)
    for i, si in enumerate(spectra_all):
        spectra[i] = torch.tensor(si).to(device)
    t = torch.linspace(0, 1, n_domain).repeat(num_samples, 1).to(device)

    inds = torch.randint(0, n_domain, (int(0.95*n_domain),))
    x_context = t[:,inds].unsqueeze(-1) 
    y_context = spectra[:,inds].unsqueeze(-1)
    mu_context, sigma_context = np_model.xy_to_mu_sigma(x_context, y_context)
    q_context = torch.distributions.Normal(mu_context, sigma_context)
        
    train_y = q_context.rsample(torch.Size([N_Z_DRAWS]))
    
    z_mean = train_y.mean(dim=0)
    z_std = train_y.std(dim=0)
    train_x = torch.from_numpy(comps_all).to(device)

    return train_x, z_mean, z_std

def run_iteration(expt, config):
    """
    Perform a single iteration of active phasemapping using a surrogate model.

    Parameters
    ----------
    expt : UVVisExperiment
        The current experimental data.
    config : dict
        The configuration settings for the iteration.

    Returns
    -------
    dict
        A dictionary containing the results of the iteration.
    """
    # assemble data for surrogate model training  
    comps_all = expt.comps 
    spectra_all = expt.spectra_normalized # use normalized spectra
    print('Data shapes : ', comps_all.shape, spectra_all.shape)
    result = {"comps_all" : comps_all,
              "spectra_all" : spectra_all,
    }

    # Specify the Neural Process model
    np_model = NeuralProcess(best_np_config["r_dim"], N_LATENT, best_np_config["h_dim"]).to(device)
    with pkg_resources.path("activephasemap.pretrained", "np_model.pt") as model_path:
        np_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    print("Finetuning Neural Process model: ")
    np_model, np_loss = finetune_neural_process(expt.t, spectra_all, np_model, **np_model_args)
    torch.save(np_model.state_dict(), config["save_dir"]+'np_model_%d.pt'%config["iteration"])
    np.save(config["save_dir"]+'np_loss_%d.npy'%config["iteration"], np_loss)
    result["np_model"] = np_model
    result["np_loss"] = np_loss

    train_x, train_z_mean, train_z_std = featurize_spectra(np_model, comps_all, spectra_all)
    train_y = torch.cat((train_z_mean, train_z_std), dim=1)
    comp_model = XGBoost(xgb_model_args)
    print("Training comosition model p(z|C): ")
    comp_train_loss, comp_eval_loss = comp_model.train(train_x, train_y)

    np.save(config["save_dir"]+'comp_train_loss_%d.npy'%config["iteration"], comp_train_loss)
    np.save(config["save_dir"]+'comp_eval_loss_%d.npy'%config["iteration"], comp_eval_loss)
    comp_model.save( config["save_dir"]+'comp_model_%d.json'%config["iteration"])
    result["comp_model"] = comp_model
    result["comp_train_loss"] = comp_train_loss
    result["comp_eval_loss"] = comp_eval_loss

    print("Collecting next data points to sample by acqusition optimization...")
    bounds = torch.tensor(config["bounds"]).transpose(-1, -2).to(device)
    acqf = XGBUncertainity(expt, bounds, np_model, comp_model)
    new_x = acqf.optimize(config["batch_size"])
    print_matrix(new_x)

    torch.save(train_x.cpu(), config["save_dir"]+"train_x_%d.pt" %config["iteration"])
    torch.save(train_z_mean.cpu(), config["save_dir"]+"train_z_mean_%d.pt" %config["iteration"])
    torch.save(train_z_std.cpu(), config["save_dir"]+"train_z_std_%d.pt" %config["iteration"])

    result["acqf"] = acqf
    result["comps_new"] = new_x.cpu().numpy()
    result["train_x"] = train_x
    result["train_z_mean"] = train_z_mean 
    result["train_z_std"] = train_z_std

    return result

def from_comp_to_spectrum(t, c, comp_model, np_model, return_mlp_outputs=False):
    """
    Generate a spectrum prediction from composition using a composition model and Neural Process model.

    Parameters
    ----------
    t : np.ndarray
        The time or wavelength values.
    c : np.ndarray
        The composition values.
    comp_model : XGBoost
        The trained composition model.
    np_model : NeuralProcess
        The trained Neural Process model.
    return_mlp_outputs : bool, optional
        Whether to return the MLP outputs, by default False.

    Returns
    -------
    np.ndarray
        The predicted mean spectrum.
    np.ndarray
        The predicted standard deviation of the spectrum.
    """
    ci = torch.tensor(c).to(device)
    z_mu, z_std = comp_model.predict(ci.view(1,1,ci.shape[0]))   
    z_dist = torch.distributions.Normal(z_mu.squeeze(), z_std.squeeze())
    z = z_dist.sample(torch.Size([100]))
    t = torch.from_numpy(t).repeat(100, 1, 1).to(device)
    t = torch.swapaxes(t, 1, 2)

    # sample NP model given the z distribution and time domain
    y_samples, _ = np_model.xz_to_y(t, z)

    mean_pred = y_samples.mean(dim=0, keepdim=True)
    sigma_pred = y_samples.std(dim=0, keepdim=True)
    mu_ = mean_pred.cpu().squeeze()
    sigma_ = sigma_pred.cpu().squeeze() 
    if return_mlp_outputs:
        return mu_, sigma_, z_mu.squeeze(), z_std.squeeze() 
    else:
        return mu_, sigma_   

@torch.no_grad()
def plot_model_accuracy(expt, config, result):
    """
    Plot the accuracy of model predictions against the experimental data.

    Parameters
    ----------
    expt : UVVisExperiment
        The current experimental data.
    config : dict
        The configuration settings for the plot.
    result : dict
        The results of the model predictions.

    Returns
    -------
    None
    """

    print("Creating plots to visualize training data predictions...")
    iter_plot_dir = config["plot_dir"]+'preds_%d/'%config["iteration"]
    if os.path.exists(iter_plot_dir):
        shutil.rmtree(iter_plot_dir)
    os.makedirs(iter_plot_dir)

    fig, ax = plt.subplots()
    ax.plot(range(len(result["comp_train_loss"])), result["comp_train_loss"])
    ax.plot(range(len(result["comp_eval_loss"])), result["comp_eval_loss"])
    plt.savefig(iter_plot_dir+'c2z_loss.png')
    plt.close() 

    num_samples, c_dim = expt.comps.shape
    for i in range(num_samples):
        fig, axs = plt.subplots(1,3, figsize=(3*4, 4))

        # Plot MLP model predictions of the spectra
        mu, sigma, z_mu, z_std = from_comp_to_spectrum(expt.t, 
                                                        expt.comps[i,:], 
                                                        result["comp_model"], 
                                                        result["np_model"],
                                                        return_mlp_outputs = True
                                                        )
        axs[0].plot(expt.wl, mu)
        minus = (mu-sigma)
        plus = (mu+sigma)
        axs[0].fill_between(expt.wl, minus, plus, color='grey')
        axs[0].scatter(expt.wl, expt.spectra_normalized[i,:], color='k', s=10)
        mu_norm = (mu - min(mu))/(max(mu)-min(mu))
        spectra_i_norm =  (expt.spectra_normalized[i,:] - min(expt.spectra_normalized[i,:])) \
        /(max(expt.spectra_normalized[i,:])-min(expt.spectra_normalized[i,:]))
        amplitude, phase = dist(expt.t, spectra_i_norm, mu_norm)
        error = 0.5*(amplitude+phase)
        axs[0].set_title("(%.2f, %.2f) : %.2f"%(expt.comps[i,0], expt.comps[i,1], error))
        
        # Plot the Z values comparision between trained and MLP predictions
        labels = []
        def add_label(violin, label):
            color = violin["bodies"][0].get_facecolor().flatten()
            labels.append((mpatches.Patch(color=color), label))

        pred = torch.distributions.Normal(z_mu, z_std).sample(torch.Size([100]))
        add_label(axs[1].violinplot(pred.cpu().numpy(), showmeans=True), label="XGB")

        train_z_samples = torch.distributions.Normal(result["train_z_mean"][i,:], 
                                                     result["train_z_std"][i,:]
                                                    ).sample(torch.Size([100]))
        add_label(axs[1].violinplot(train_z_samples.cpu().numpy(), showmeans=True), label="NP")
        axs[1].legend(*zip(*labels))

        # Plot NP model prediction from trained Z values
        q_train = torch.distributions.Normal(result["train_z_mean"][i,:], 
                                             result["train_z_std"][i,:]
                                            )
        x_target = torch.from_numpy(expt.t).view(1,len(expt.t),1).to(device)
        for j in range(200):
            y_pred_mu, y_pred_sigma = result["np_model"].xz_to_y(x_target, q_train.rsample())
            p_y_pred = torch.distributions.Normal(y_pred_mu, y_pred_sigma)
            mu = p_y_pred.loc.detach()
            axs[2].plot(expt.wl, mu.squeeze().cpu().numpy(), alpha=0.05, c='tab:orange')
        axs[2].scatter(expt.wl, expt.spectra_normalized[i,:], color='k', s=10)
        axs[2].set_title("NP Model")

        plt.savefig(iter_plot_dir+'%d.png'%(i))
        plt.close()

@torch.no_grad()
def plot_iteration(expt, config, result):
    """
    Plot the current iteration of the active phasemapping process.

    Parameters
    ----------
    expt : UVVisExperiment
        The current experimental data.
    config : dict
        The configuration settings for the iteration.
    result : dict
        The results of the current iteration.

    Returns
    -------
    tuple
        The figure and axes of the plot.
    """
    layout = [['A1','A2', 'C', 'C'], 
              ['B1', 'B2', 'C', 'C']
              ]
    bounds = torch.tensor(config["bounds"]).transpose(-1, -2).to(device)

    # plot selected points
    bounds =  expt.bounds.cpu().numpy()
    C_grid = get_twod_grid(20, bounds)
    fig, axs = plt.subplot_mosaic(layout, figsize=(4*4, 4*2))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    axs['A1'].scatter(expt.comps[:,0], expt.comps[:,1], marker='x', color='k')
    axs['A1'].scatter(result["comps_new"][:,0], result["comps_new"][:,1], color='tab:red')
    axs['A1'].set_xlabel('C1')
    axs['A1'].set_ylabel('C2')    
    axs['A1'].set_title('C sampling')
    axs['A1'].set_xlim([bounds[0,0], bounds[1,0]])
    axs['A1'].set_ylim([bounds[0,1], bounds[1,1]])

    # plot acqf
    C_grid_ = torch.tensor(C_grid).to(device).reshape(len(C_grid),1,2)
    rx_train_, rx_, sigma_x_ = result["acqf"](C_grid_, return_rx_sigma=True)
    rx_train = rx_train_.squeeze().cpu().numpy()
    rx = rx_.squeeze().cpu().numpy()
    sigma_x = sigma_x_.squeeze().cpu().numpy()

    cmap = colormaps["magma"]
    norm = Normalize(vmin=min(rx), vmax = max(rx))
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    axs['B1'].tricontourf(C_grid[:,0], C_grid[:,1], rx, cmap=cmap, norm=norm)
    axs['B1'].scatter(expt.points[:, 0], expt.points[:, 1], s=rx_train*10, edgecolor='w', facecolors='none')
    axs['B1'].scatter(result["comps_new"][:,0], result["comps_new"][:,1], marker='x', color='w')    
    divider = make_axes_locatable(axs["B1"])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.set_ylabel(r'$ \alpha r(x)$')
    axs['B1'].set_xlabel('C1')
    axs['B1'].set_ylabel('C2') 

    norm = Normalize(vmin=min(sigma_x), vmax = max(sigma_x))
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    axs['B2'].tricontourf(C_grid[:,0], C_grid[:,1], sigma_x, cmap=cmap, norm=norm)
    axs['B2'].scatter(result["comps_new"][:,0], result["comps_new"][:,1], marker='x', color='w')    
    divider = make_axes_locatable(axs["B2"])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.set_ylabel(r"$(1-\alpha) \sigma(x)$")
    axs['B2'].set_xlabel('C1')
    axs['B2'].set_ylabel('C2')     

    acqv = rx + sigma_x
    norm = Normalize(vmin=0, vmax = 1)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    axs['A2'].tricontourf(C_grid[:,0], C_grid[:,1], acqv, cmap=cmap, norm=norm)
    axs['A2'].scatter(result["comps_new"][:,0], result["comps_new"][:,1], marker='x', color='w')    
    divider = make_axes_locatable(axs["A2"])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.set_ylabel(r"$\alpha r_(x) + (1-\alpha) \sigma(x)$")
    axs['A2'].set_xlabel('C1')
    axs['A2'].set_ylabel('C2')    

    # plot the full evaluation on composition space
    ax = axs['C']
    bounds = expt.bounds.cpu().numpy()
    scaler_x = MinMaxScaler(bounds[0,0], bounds[1,0])
    scaler_y = MinMaxScaler(bounds[0,1], bounds[1,1])
    
    ax.xaxis.set_major_formatter(lambda x, pos : scaled_tickformat(scaler_x, x, pos))
    ax.yaxis.set_major_formatter(lambda y, pos : scaled_tickformat(scaler_y, y, pos))
    C_grid = get_twod_grid(10, bounds)

    for ci in C_grid:
        mu, sigma = from_comp_to_spectrum(expt.t, ci, result["comp_model"], result["np_model"])
        mu_ = mu.cpu().squeeze().numpy()
        sigma_ = sigma.cpu().squeeze().numpy()
        norm_ci = np.array([scaler_x.transform(ci[0]), scaler_y.transform(ci[1])])
        _inset_spectra(norm_ci, expt.t, mu_, sigma_, ax, show_sigma=True)
    ax.set_xlabel('C1')
    ax.set_ylabel('C2')

    return fig, axs
