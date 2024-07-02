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
    finetune_tags = ["_to_hidden.4.weight", "hidden_to"]
    for name, param in model.named_parameters():
        if "_to_hidden.4.weight" in name:
            torch.nn.init.xavier_uniform_(param)
            print("Finetuning %s..."%name)
            finetune_params.append(param)
        elif "hidden_to" in name:
            if "weight" in name:
                torch.nn.init.xavier_uniform_(param)
                finetune_params.append(param)
                print("Finetuning %s..."%name)
        else:
            freeze_params.append(param)
    
    # look for possible parallelization on multiple GPUs
    if device=="cuda":
        n_devices = torch.cuda.device_count()
        print('Running NP-finetuning on %d GPUs.'%n_devices)
        if n_devices>1:
            model = torch.nn.DataParallel(model)
    else:
        print("Using %s device with no parallelization"%device)
    model.to(device)
    model.training = True
    lr = kwargs.get('learning_rate',  1e-3)
    optimizer = torch.optim.Adam([{'params': freeze_params, "lr":lr},
                                  {'params': finetune_params, 'lr': lr*10}],
                                  lr=lr
                                )
    epoch_loss = []
    verbose = kwargs.get("verbose", 1)
    for epoch in range(num_iterations):
        model, optimizer, loss_value = train_neural_process(model, data_loader,optimizer)

        if (epoch%verbose==0) or (epoch==num_iterations-1):
            print("Epoch: %d, Loss value : %2.4f"%(epoch, loss_value))
        epoch_loss.append(loss_value)

    # freeze model training
    model.training = False

    return model, epoch_loss