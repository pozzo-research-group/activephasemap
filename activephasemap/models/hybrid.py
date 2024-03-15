import torch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
from torch.utils.data import DataLoader, Dataset
import numpy as np
from activephasemap.np.neural_process import NeuralProcess
from activephasemap.np.training import NeuralProcessTrainer
from activephasemap.np.utils import context_target_split

class NPModelDataset(Dataset):
    def __init__(self, time, y):
        self.data = []
        for yi in y:
            xi = torch.from_numpy(time).to(device)
            xi = xi.view(xi.shape[0],1).to(device)
            yi = yi.view(yi.shape[0],1).to(device)
            self.data.append((xi,yi))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def update_npmodel(time, np_model, data, **kwargs):
    batch_size = kwargs.get('batch_size',  2)
    num_context = kwargs.get('num_context',  25)
    num_target = kwargs.get('num_target',  25)
    num_iterations = kwargs.get('num_iterations',  30)
    # print('func:update_npmodel: input spectra shape :', data.y.shape)
    dataset = NPModelDataset(time, data.y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for name, param in np_model.named_parameters():
        if 'hidden_to' in name:
            param.requires_grad = False
        elif 'r_to_hidden' in name:
            param.requires_grad = False   
    optimizer = torch.optim.Adam(np_model.parameters(), lr=kwargs.get('lr',  1e-3))
    trainer = NeuralProcessTrainer(device, np_model, optimizer,
    num_context_range=(num_context, num_context),
    num_extra_target_range=(num_target, num_target),
    print_freq=kwargs.get('print_freq',  10)
    )

    np_model.training = True
    trainer.train(data_loader, num_iterations, verbose = kwargs.get("verbose", False))
    loss = trainer.epoch_loss_history[-1]
    print('func:update_npmodel: NP model loss : %.2f'%loss)

    # freeze model training
    np_model.training = False

    return np_model, loss 