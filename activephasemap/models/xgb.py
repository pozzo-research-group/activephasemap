import torch
import xgboost
import numpy as np
from sklearn.model_selection import ParameterGrid
import pdb

class XGBoost(torch.nn.Module):
    """
    A PyTorch-compatible XGBoost model with training, prediction, saving, loading, 
    and backward pass support.
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model = None
        self.evals_result = {}

    def train(self, inputs, targets):
        """
        Train the XGBoost model with cross-validation and hyperparameter tuning.
        """
        inputs_np = inputs.clone().detach().cpu().numpy()
        targets_np = targets.clone().detach().cpu().numpy()
        
        dtrain = xgboost.DMatrix(inputs_np, label=targets_np)

        # Use cross-validation to evaluate the best parameters if param_grid is provided
        param_grid = {"eta":[0.002, 0.4, 0.8], "max_depth": [2, 4, 6, 10]}
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
        num_samples = inputs_np.shape[0]
        split_idx = int(0.8 * num_samples)
        train_inputs_np, eval_inputs_np = inputs_np[:split_idx], inputs_np[split_idx:]
        train_targets_np, eval_targets_np = targets_np[:split_idx], targets_np[split_idx:]
        
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
        """
        preds = XGBoostAutoGrad.apply(self.model, inputs)
        _, _, dz = preds.shape
        z_mu = preds[...,:int(dz/2)]
        z_std = torch.abs(preds[..., int(dz/2):]) 

        return z_mu, z_std

    def save(self, path):
        """
        Save the trained model to the specified path.
        """
        if self.model is None:
            raise ValueError("The model has not been trained yet.")
        self.model.save_model(path)
    
    def load(self, path):
        """
        Load a trained model from the specified path.
        """
        self.model = xgboost.Booster()
        self.model.load_model(path)

    def forward(self, inputs):
        """
        Forward pass: call predict internally.
        """
        return self.predict(inputs)

class XGBoostAutoGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, model, x):
        """
        Predict outputs for given inputs using the trained model.
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
        """
        x, = ctx.saved_tensors
        _, _, dx = x.shape
        nr, nb, dz = grad_output.shape

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
            f_grad = (f_plus - f_minus) / (2 * epsilon)
            f_grad_tensor = torch.tensor(f_grad, dtype=x.dtype, device=x.device)
            jacobian[..., i] = f_grad_tensor.view(nr*nb, dz) 
        
        grad = torch.einsum('bij,bjk->bik', jacobian.permute(0,2,1), grad_output)
        
        return None, grad.permute(0,2,1)