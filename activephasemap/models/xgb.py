import torch
import xgboost
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split

class XGBoost(torch.nn.Module):
    """
    A PyTorch-compatible XGBoost model with training, prediction, saving, loading, 
    and backward pass support.

    This class wraps the XGBoost model to integrate it with PyTorch, enabling 
    its use in workflows requiring forward and backward passes.

    Parameters
    ----------
    params : dict
        Dictionary of hyperparameters for the XGBoost model.

    Attributes
    ----------
    params : dict
        Hyperparameters for the XGBoost model.
    model : xgboost.Booster or None
        Trained XGBoost model. None if the model has not been trained yet.
    evals_result : dict
        Dictionary storing evaluation results during training.

    Methods
    -------
    train(inputs, targets)
        Train the XGBoost model with cross-validation and hyperparameter tuning.
    predict(inputs)
        Predict outputs for given inputs using the trained model.
    save(path)
        Save the trained model to the specified path.
    load(path)
        Load a trained model from the specified path.
    forward(inputs)
        Forward pass: call predict internally.
    """

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model = None
        self.evals_result = {}

    def train(self, inputs, targets):
        """
        Train the XGBoost model with cross-validation and hyperparameter tuning.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (n_samples, n_features).
        targets : torch.Tensor
            Target tensor of shape (n_samples,).

        Returns
        -------
        train_loss : list of float
            Training loss values for each iteration.
        eval_loss : list of float
            Evaluation loss values for each iteration.

        Notes
        -----
        - Hyperparameter tuning is performed using a parameter grid.
        - Early stopping is applied based on evaluation loss.

        Raises
        ------
        ValueError
            If no hyperparameter combination improves the performance.

        """
        inputs_np = inputs.clone().detach().cpu().numpy()
        targets_np = targets.clone().detach().cpu().numpy()
        
        dtrain = xgboost.DMatrix(inputs_np, label=targets_np)

        # Use cross-validation to evaluate the best parameters if param_grid is provided
        # param_grid = {"eta":[0.002, 0.4, 0.8], "max_depth": [2, 4, 6, 10]}
        param_grid = {'max_depth': [3, 5, 7, 15, 25],
                      'learning_rate': [0.01, 0.1, 0.2],
                      'subsample': [0.8, 1],
                      'colsample_bytree': [0.8, 1],
                      'gamma': [0, 0.1, 0.2]
                      }
        if param_grid:
            best_params = None
            best_score = float("inf")
            for params in ParameterGrid(param_grid):
                # Merge given params with default ones
                params = {**self.params, **params}
                cv_results = xgboost.cv(
                    params,
                    dtrain,
                    num_boost_round=100,
                    nfold=5,
                    metrics=["rmse"],
                    early_stopping_rounds=10,
                    seed=42
                )
                mean_rmse = cv_results["test-rmse-mean"].min()
                if mean_rmse < best_score:
                    best_score = mean_rmse
                    best_params = params
            print(f"Best Parameters: {best_params}, Best RMSE: {best_score}")
            self.params = best_params  # Update the model's parameters with the best ones

        # Train the model using the best parameters
        train_inputs_np, eval_inputs_np, train_targets_np, eval_targets_np = train_test_split(inputs_np, 
                                                                                              targets_np, 
                                                                                              test_size=0.2, 
                                                                                              random_state=42
                                                                                              )

        # Create DMatrix for train and eval sets
        dtrain = xgboost.DMatrix(train_inputs_np, label=train_targets_np)
        deval = xgboost.DMatrix(eval_inputs_np, label=eval_targets_np)
        evals = [(dtrain, "train"), (deval, "eval")]
        self.evals_result = {}
        self.model = xgboost.train(self.params, 
                                   dtrain, 
                                   num_boost_round=100, 
                                   evals=evals, 
                                   evals_result=self.evals_result, 
                                   early_stopping_rounds=10
                                   )
        train_loss = self.evals_result["train"][self.params["eval_metric"]]
        eval_loss = self.evals_result["eval"][self.params["eval_metric"]]

        return train_loss, eval_loss

    def predict(self, inputs):
        """
        Predict outputs for given inputs using the trained model.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (n_samples, n_features).

        Returns
        -------
        z_mu : torch.Tensor
            Predicted mean tensor of shape (n_samples, n_outputs).
        z_std : torch.Tensor
            Predicted standard deviation tensor of shape (n_samples, n_outputs).

        Raises
        ------
        ValueError
            If the model has not been trained.

        """

        preds = XGBoostAutoGrad.apply(self.model, inputs)
        _, _, dz = preds.shape
        z_mu = preds[...,:int(dz/2)]
        z_std = torch.abs(preds[..., int(dz/2):]) 

        return z_mu, z_std

    def save(self, path):
        """
        Save the trained model to the specified path.

        Parameters
        ----------
        path : str
            File path where the model should be saved.

        Raises
        ------
        ValueError
            If the model has not been trained.

        Examples
        --------
        >>> model.save("xgboost_model.json")
        """
        if self.model is None:
            raise ValueError("The model has not been trained yet.")
        self.model.save_model(path)
    
    def load(self, path):
        """
        Load a trained model from the specified path.

        Parameters
        ----------
        path : str
            File path from which the model should be loaded.

        Examples
        --------
        >>> model.load("xgboost_model.json")
        """
        self.model = xgboost.Booster()
        self.model.load_model(path)

    def forward(self, inputs):
        """
        Forward pass: call predict internally.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (n_samples, n_features).

        Returns
        -------
        z_mu : torch.Tensor
            Predicted mean tensor of shape (n_samples, n_outputs).
        z_std : torch.Tensor
            Predicted standard deviation tensor of shape (n_samples, n_outputs).
        """
        return self.predict(inputs)

class XGBoostAutoGrad(torch.autograd.Function):
    """
    Custom autograd Function for integrating XGBoost with PyTorch.

    This class provides forward and backward methods to enable gradient 
    computation for XGBoost predictions.

    Methods
    -------
    forward(ctx, model, x)
        Predict outputs for given inputs using the trained model.
    backward(ctx, grad_output)
        Compute gradients w.r.t. the inputs using finite differences.
    """
    @staticmethod
    def forward(ctx, model, x):
        """
        Predict outputs for given inputs using the trained model.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object to save information for backward computation.
        model : xgboost.Booster
            Trained XGBoost model.
        x : torch.Tensor
            Input tensor of shape (n_batches, batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Predicted outputs tensor of shape (n_batches, batch_size, n_outputs).

        Raises
        ------
        ValueError
            If the model is not trained or input dimensions are incorrect.
        """
        ctx.save_for_backward(x)
        ctx.model = model
        nr, nb, dx = x.shape
        x_2d = x.view(nr*nb, dx)
        x_np = x_2d.clone().detach().cpu().numpy()
        dmatrix = xgboost.DMatrix(x_np)
        preds = model.predict(dmatrix)
        if preds.ndim==1:
            ny = 1
        else:
            ny = preds.shape[-1]
        
        preds_tensor = torch.tensor(preds, dtype=x.dtype, device=x.device).view(nr, nb, ny)

        return preds_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients w.r.t. the inputs using finite differences.

        Parameters
        ----------
        ctx : torch.autograd.Function
            Context object containing saved tensors and model.
        grad_output : torch.Tensor
            Gradient of the loss w.r.t. the outputs.

        Returns
        -------
        None
            Gradients w.r.t. the model (not computed).
        torch.Tensor
            Gradients w.r.t. the inputs.

        Notes
        -----
        - Gradients are approximated using central finite differences.
        - The computation is resource-intensive and may be slow for large datasets.

        Raises
        ------
        ValueError
            If input dimensions are inconsistent with the forward pass.
        """
        x, = ctx.saved_tensors
        nr, nb, dx = x.shape
        _, _, dz = grad_output.shape

        x_np = x.view(nr*nb, dx).detach().cpu().numpy().astype(np.float64)

        # Finite differences for gradient approximation
        epsilon = 0.5*torch.norm(x, dim=(1,2)).mean().item()

        # Initialize gradient
        jacobian = torch.zeros(nr*nb, dz, dx).to(x.device)

        # Compute central differences for each input dimension
        for i in range(dx):
            perturb = np.zeros_like(x_np)
            perturb[:,i] = epsilon
            x_plus = xgboost.DMatrix(x_np + perturb)
            x_minus = xgboost.DMatrix(x_np - perturb)
            f_plus = ctx.model.predict(x_plus)  
            f_minus = ctx.model.predict(x_minus) 
            # Central difference approximation of the derivative
            f_grad = (f_plus - f_minus) / (2 * epsilon + 1e-3)
            f_grad_tensor = torch.tensor(f_grad, dtype=x.dtype, device=x.device)
            jacobian[..., i] = f_grad_tensor.view(nr*nb, dz) 
        
        grad = torch.einsum('bij,bjk->bik', jacobian.permute(0,2,1), grad_output.view(nr*nb, dz, 1))
        
        return None, grad.view(nr, nb, dx)