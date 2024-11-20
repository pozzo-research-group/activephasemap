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
        inputs_np = inputs.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
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
        if self.model is None:
            raise ValueError("The model has not been trained yet.")
        nr, nb, dx = inputs.shape
        inputs_2d = inputs.view(nr*nb, dx)
        inputs_np = inputs_2d.detach().cpu().numpy()
        dmatrix = xgboost.DMatrix(inputs_np)
        preds = self.model.predict(dmatrix)
        preds_tensor = torch.tensor(preds, dtype=inputs.dtype, device=inputs.device)
        _, dz = preds_tensor.shape
        z_mu = preds_tensor.view(nr, nb, dz)[...,:int(dz/2)]
        z_std = torch.abs(preds_tensor.view(nr, nb, dz)[..., int(dz/2):])  

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

    def backward(self, grad_output, inputs):
        """
        Backward pass: compute gradients w.r.t. the inputs using finite differences.
        """
        if self.model is None:
            raise ValueError("The model has not been trained yet.")
        
        inputs_np = inputs.detach().cpu().numpy()
        dtest = xgboost.DMatrix(inputs_np)
        preds = self.model.predict(dtest, output_margin=True)  # Get raw predictions (margins)

        # Finite differences for gradient approximation
        epsilon = 1e-5
        grads = np.zeros_like(inputs_np)
        for i in range(inputs_np.shape[1]):  # Loop over each feature
            perturbed_inputs = inputs_np.copy()
            perturbed_inputs[:, i] += epsilon  # Apply small perturbation to the i-th feature
            perturbed_dmatrix = xgboost.DMatrix(perturbed_inputs)
            perturbed_preds = self.model.predict(perturbed_dmatrix, output_margin=True)
            
            # Compute gradient approximation for the i-th feature
            grads[:, i] = (perturbed_preds - preds) / epsilon

        # Convert gradients to PyTorch tensors
        grad_input = torch.tensor(grads, dtype=inputs.dtype, device=inputs.device)
        
        # Scale by chain rule with grad_output from subsequent layers
        grad_input = grad_input * grad_output.unsqueeze(1)

        return grad_input
