import torch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
from torch.utils.data import DataLoader, Dataset
from activephasemap.np.training import NeuralProcessTrainer

class NPModelDataset(Dataset):
    def __init__(self, time, y):
        self.data = []
        for yi in y:
            xi = torch.from_numpy(time).to(device)
            yi = torch.from_numpy(yi).to(device)
            self.data.append((xi.unsqueeze(1),yi.unsqueeze(1)))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def update_np(time, data,np_model, **kwargs):
    num_domain = len(time)
    num_context_min = 3
    num_context_max = int((num_domain/2)-3) 
    num_target_extra_min = int(num_domain/2) 
    num_target_extra_max = int((num_domain/2) +3)
    batch_size = kwargs.get('batch_size',  16)
    num_iterations = kwargs.get('num_iterations',  30)
    # print('func:update_npmodel: input spectra shape :', data.y.shape)
    dataset = NPModelDataset(time, data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # for name, param in np_model.named_parameters():
    #     if 'hidden_to' in name:
    #         param.requires_grad = False
    #     elif 'r_to_hidden' in name:
    #         param.requires_grad = False   
    optimizer = torch.optim.Adam(np_model.parameters(), lr=kwargs.get('lr',  1e-3))
    trainer = NeuralProcessTrainer(device, np_model, optimizer,
    num_context_range=(num_context_min, num_context_max),
    num_extra_target_range=(num_target_extra_min, num_target_extra_max),
    print_freq=kwargs.get('print_freq',  10)
    )

    np_model.training = True
    trainer.train(data_loader, num_iterations, verbose = kwargs.get("verbose", False))
    loss = trainer.epoch_loss_history[-1]
    print('func:update_npmodel: NP model loss : %.2f'%loss)

    # freeze model training
    np_model.training = False

    return np_model, trainer.epoch_loss_history 