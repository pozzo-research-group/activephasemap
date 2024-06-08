import torch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
from torch.utils.data import DataLoader, Dataset
from activephasemap.models.np import train_neural_process

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

def finetune_neural_process(x, y, model, **kwargs):
    batch_size = kwargs.get('batch_size',  16)
    num_iterations = kwargs.get('num_iterations',  30)
    dataset = NPModelDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    freeze_params, finetune_params = [], []
    for name, param in model.named_parameters():
        if 'hidden_to' in name:
            param.requires_grad = False
            # if "weight" in name:
            #     torch.nn.init.xavier_uniform_(param)
            finetune_params.append(param)
        else:
            freeze_params.append(param)

    model.training = True
    lr = kwargs.get('learning_rate',  1e-3)
    optimizer = torch.optim.Adam([{'params': freeze_params, "lr":lr},
                                  {'params': finetune_params, 'lr': lr*20}],
                                  lr=lr
                                )
    epoch_loss = []
    for epoch in range(num_iterations):
        model, optimizer, loss_value = train_neural_process(model, data_loader,optimizer)

        if epoch%kwargs.get("verbose", 1)==0:
            print("Epoch: %d, Loss value : %2.4f"%(epoch, loss_value))
        epoch_loss.append(loss_value)

    # freeze model training
    model.training = False

    return model, epoch_loss