import torch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double)
from torch.utils.data import DataLoader, Dataset
from activephasemap.models.np import train_neural_process

class NPModelDataset(Dataset):
    """
    Dataset class for preparing data for Neural Process models.

    This class prepares the data in the format required for Neural Process models by 
    converting input features and target values into PyTorch tensors and storing them 
    as pairs of `(x, y)`.

    Parameters
    ----------
    time : numpy.ndarray
        Array of time or domain points.
    y : numpy.ndarray
        Array of target values corresponding to `time`.

    Attributes
    ----------
    data : list of tuple
        A list of `(x, y)` pairs where both `x` and `y` are PyTorch tensors.

    Methods
    -------
    __getitem__(index)
        Retrieves the data pair `(x, y)` at the specified index.
    __len__()
        Returns the number of data pairs in the dataset.

    """
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
    """
    Fine-tunes a Neural Process model on a given dataset.

    This function fine-tunes a Neural Process model on input data (`x`, `y`) by training specific 
    layers (e.g., `_to_hidden.4.weight` and `hidden_to`) while freezing other model parameters. 
    It supports multi-GPU training, customizable learning rates, batch size, and the number of iterations.

    Parameters
    ----------
    x : numpy.ndarray
        Array of input features, typically representing time or domain points.
    y : numpy.ndarray
        Array of target values, corresponding to the input features.
    model : torch.nn.Module
        Neural Process model to be fine-tuned.
    **kwargs : dict, optional
        Additional parameters for fine-tuning:
        - `batch_size` (int): Size of each batch for training. Default is 16.
        - `num_iterations` (int): Number of fine-tuning epochs. Default is 30.
        - `learning_rate` (float): Base learning rate for optimizer. Default is 1e-3.
        - `verbose` (int): Verbosity level for logging epoch losses. Default is 1.

    Returns
    -------
    model : torch.nn.Module
        Fine-tuned Neural Process model.
    epoch_loss : list of float
        List of average loss values for each epoch.

    Notes
    -----
    - The function initializes certain parameters (`_to_hidden.4.weight` and weights in `hidden_to`) 
      using Xavier uniform initialization before fine-tuning.
    - Fine-tuned parameters are trained with a learning rate 10 times larger than other parameters.
    - If multiple GPUs are available, the function leverages `torch.nn.DataParallel` for parallel training.

    """
    batch_size = kwargs.get('batch_size',  16)
    num_iterations = kwargs.get('num_iterations',  30)
    dataset = NPModelDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    freeze_params, finetune_params = [], []
    finetune_tags = ["_to_hidden.4.weight", "hidden_to"]
    for name, param in model.named_parameters():
        if "_to_hidden.4.weight" in name:
            torch.nn.init.xavier_uniform_(param)
            # print("Finetuning %s..."%name)
            finetune_params.append(param)
        elif "hidden_to" in name:
            if "weight" in name:
                torch.nn.init.xavier_uniform_(param)
                finetune_params.append(param)
                # print("Finetuning %s..."%name)
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