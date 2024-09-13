import torch
import numpy as np 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pdb 

class MLPModel(torch.nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()
        layers = [torch.nn.Linear(x_dim, 128),
                  torch.nn.Sigmoid(),
                  torch.nn.Linear(128, 128),
                  torch.nn.Sigmoid(),
                  ]

        self.x_to_hidden = torch.nn.Sequential(*layers)
        self.hidden_to_mu = torch.nn.Linear(128, z_dim)
        self.hidden_to_std = torch.nn.Linear(128, z_dim)

    def forward(self, x):
        h = self.x_to_hidden(x)
        mu = self.hidden_to_mu(h)
        std = torch.sigmoid(self.hidden_to_std(h))

        return mu, std

class MLP(torch.nn.Module):
    def __init__(self, c, z_mu, z_std, **kwargs):
        super().__init__()
        self.c = c.to(device) 
        self.z_mu = z_mu.to(device) 
        self.z_std = z_std.to(device)
        self.learning_rate = kwargs.get("learning_rate",0.01)
        self.num_epochs = kwargs.get("num_epochs", 16)
        self.debug = kwargs.get("debug", False)
        self.verbose = kwargs.get("verbose", 1)
        self.mlp = MLPModel(c.shape[-1], z_mu.shape[-1]).to(device)

    def loss(self, z_mu_train, z_std_train, z_mu_pred, z_std_pred):
        error_mean = torch.nn.functional.mse_loss(z_mu_train, z_mu_pred)
        error_std = torch.nn.functional.mse_loss(z_std_train, z_std_pred)

        return (error_mean+error_std).mean()

    def fit(self, use_early_stoping=True):
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.learning_rate)
        self.mu_scaler = StandardScaler()
        self.std_scaler = StandardScaler()
        train_ind = np.random.randint(0, len(self.c), int(0.8*len(self.c)))
        test_ind = np.setdiff1d(np.arange(len(self.c)), train_ind)
        self.train_c, self.train_z_mu, self.train_z_std = self.c[train_ind,:], self.z_mu[train_ind,:], self.z_std[train_ind,:]
        self.train_z_mu  = self.mu_scaler.fit_transform(self.train_z_mu) 
        self.train_z_std = self.std_scaler.fit_transform(self.train_z_std)
        self.test_c, self.test_z_mu, self.test_z_std = self.c[test_ind,:], self.z_mu[test_ind,:], self.z_std[test_ind,:]
        self.test_z_mu  = self.mu_scaler.transform(self.test_z_mu) 
        self.test_z_std = self.std_scaler.transform(self.test_z_std)

        train_dataset = torch.utils.data.TensorDataset(self.train_c, self.train_z_mu, self.train_z_std)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

        eval_dataset = torch.utils.data.TensorDataset(self.test_c, self.test_z_mu, self.test_z_std)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=4, shuffle=False)

        self.mlp.train()
        train_loss, eval_loss = [], []
        early_stopper = EarlyStopper(patience=100, min_delta=0.1)
        for epoch in range(self.num_epochs):
            epoch_train_loss = []
            for xb, yb_mu, yb_std in train_loader:
                optimizer.zero_grad()
                z_mu_, z_std_ = self.mlp(xb)
                loss = self.loss(yb_mu, yb_std, z_mu_, z_std_)
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss.item())   

            train_loss.append(sum(epoch_train_loss)/len(epoch_train_loss))

            with torch.no_grad():
                epoch_val_loss = []
                for xb, yb_mu, yb_std in eval_loader:
                    z_mu_, z_std_ = self.mlp(xb)
                    loss = self.loss(yb_mu, yb_std, z_mu_, z_std_)
                    epoch_val_loss.append(loss.item())
                
                eval_loss.append(sum(epoch_val_loss)/len(epoch_val_loss))

            if ((epoch) % self.verbose == 0) or (epoch==self.num_epochs-1):
                print(
                    f"Epoch {epoch+1:>3}/{self.num_epochs} - Loss: {train_loss[-1]:>4.3f} ", end=""
                )
                print(f" Evaluation Loss: {eval_loss[-1]:>4.3f} ", end="")
                print(" Early stopping counter: %d"%early_stopper.counter)

            if use_early_stoping and early_stopper.early_stop(eval_loss[-1]):
                print("Early stopping...")
                print(
                    f"Epoch {epoch+1:>3}/{self.num_epochs} - Loss: {train_loss[-1]:>4.3f} ", end=""
                )
                print(f" Evaluation Loss: {eval_loss[-1]:>4.3f} ")
                break

        return train_loss, eval_loss        

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class StandardScaler:

    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.

        from : https://gist.github.com/aryan-f/8a416f33a27d73a149f92ce4708beb40
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)
    
    def inverse_transform(self, values):
        return (self.std + self.epsilon)*(values) + self.mean 