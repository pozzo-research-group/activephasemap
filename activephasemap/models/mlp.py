import torch
import numpy as np 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLPModel(torch.nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()
        layers = [torch.nn.Linear(x_dim, 16),
                  torch.nn.ReLU(),
                  torch.nn.Linear(16, 32),
                  torch.nn.ReLU(),
                  torch.nn.Linear(32, 16),
                  torch.nn.ReLU()
                  ]

        self.x_to_hidden = torch.nn.Sequential(*layers)
        self.hidden_to_mu = torch.nn.Linear(16, z_dim)
        self.hidden_to_std = torch.nn.Linear(16, z_dim)

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

    def fit(self):
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.learning_rate)

        train_ind = np.random.randint(0, len(self.c), int(0.9*len(self.c)))
        test_ind = np.setdiff1d(np.arange(len(self.c)), train_ind)
        self.train_c, self.train_z_mu, self.train_z_std = self.c[train_ind,:], self.z_mu[train_ind,:], self.z_std[train_ind,:]
        self.test_c, self.test_z_mu, self.test_z_std = self.c[test_ind,:], self.z_mu[test_ind,:], self.z_std[test_ind,:]

        train_dataset = torch.utils.data.TensorDataset(self.train_c, self.train_z_mu, self.train_z_std)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

        eval_dataset = torch.utils.data.TensorDataset(self.test_c, self.test_z_mu, self.test_z_std)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=4, shuffle=False)

        self.mlp.train()
        train_loss = []
        for epoch in range(self.num_epochs):
            epoch_loss = []
            for xb, yb_mu, yb_std in train_loader:
                optimizer.zero_grad()
                z_mu_, z_std_ = self.mlp(xb)
                loss = self.loss(yb_mu, yb_std, z_mu_, z_std_)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())   

            train_loss.append(sum(epoch_loss)/len(epoch_loss))
            if ((epoch) % self.verbose == 0) or (epoch==self.num_epochs-1):
                print(
                    f"Epoch {epoch+1:>3}/{self.num_epochs} - Loss: {train_loss[-1]:>4.3f} ", end=""
                )
                with torch.no_grad():
                    eval_loss = []
                    for xb, yb_mu, yb_std in eval_loader:
                        z_mu_, z_std_ = self.mlp(xb)
                        loss = self.loss(yb_mu, yb_std, z_mu_, z_std_)
                        eval_loss.append(loss.item())
                    print(f" Evaluation Loss: {sum(eval_loss)/len(eval_loss):>4.3f} ")

        return train_loss        


