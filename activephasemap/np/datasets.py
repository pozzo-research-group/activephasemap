import glob
import numpy as np
import torch
from math import pi
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
RNG = np.random.default_rng()
import pdb

class MultiPeakGaussians(Dataset):
    def __init__(self, num_samples=1000, num_points=100, warping=False):
        self.N = num_samples
        self.num_points = num_points
        self.warping = warping
        x = torch.linspace(0,1,num_points).unsqueeze(1)
        self.data = []
        for i in range(self.N):
            t = np.linspace(0,1, self.num_points)
            if self.warping:
                t = self.gam(t)
            y = self.g(t).astype(np.double)
            self.data.append((x, torch.from_numpy(y).unsqueeze(1))) 

    def g(self, t):
        out = np.ones(t.shape)
        p = RNG.choice([1,2,3])
        for i in range(1,p+1):
            zi = np.random.normal(1, 0.1)
            mean = (2*i-1)/(2*p)
            std = 1/(3*p)
            out += zi*self.phi(t, mean, std)

        return out

    def gamma(self, t):
        a = np.random.uniform(-3, 3)
        if a==0:
            gam = t
        else:
            gam = (np.exp(a*t)-1)/(np.exp(a)-1)

        return gam  

    def phi(self, t, mu, sigma):
        factor = 1/(2*(sigma**2))
        return np.exp(-factor*(t-mu)**2)

    def __getitem__(self, i):
        return self.data[i]
    
    def __len__(self):
        return self.N

class GaussiansData(Dataset):
    def __init__(self, num_samples=1000, num_points=100):
        self.N = num_samples
        self.t = torch.linspace(0,1,num_points).unsqueeze(1)
        # Generate synthetic dataset with randomly 
        # shifted noisy 1D signal
        torch.manual_seed(1)  # for reproducibility
        x = torch.linspace(-12, 12, num_points).expand(self.N, num_points)
        noise = torch.randint(1, 100, (self.N, 1)) / 1e5
        mu = torch.randint(-30, 30, size=(self.N, 1)) / 10
        sig = torch.randint(50, 500, size=(self.N, 1)) / 1e2
        train_data = self.gaussian(x, mu, sig) + noise * torch.randn(size=(self.N, num_points))
        # Normalize to (0, 1)
        train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min()) 
        self.data = []
        for i in range(self.N):
            self.data.append((self.t, train_data[i].unsqueeze(1))) 
        
    
    def gaussian(self, x, mu, sig):
        return torch.exp(-torch.pow(x - mu, 2.) / (2 * torch.pow(sig, 2.)))
  
    def __getitem__(self, i):
        return self.data[i]
    
    def __len__(self):
        return self.N


class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.

    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled.

    num_samples : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """
    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Shape (num_points, x_dim)
            x = torch.linspace(-pi, pi, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            y = a * torch.sin(x - b)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


def mnist(batch_size=16, size=28, path_to_data='../../mnist_data'):
    """MNIST dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def celeba(batch_size=16, size=32, crop=89, path_to_data='../celeba_data',
           shuffle=True):
    """CelebA dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image.

    crop : int
        Size of center crop. This crop happens *before* the resizing.

    path_to_data : string
        Path to CelebA data files.
    """
    transform = transforms.Compose([
        transforms.CenterCrop(crop),
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    celeba_data = CelebADataset(path_to_data,
                                transform=transform)
    celeba_loader = DataLoader(celeba_data, batch_size=batch_size,
                               shuffle=shuffle)
    return celeba_loader


class CelebADataset(Dataset):
    """CelebA dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        path_to_data : string
            Path to CelebA data files.

        subsample : int
            Only load every |subsample| number of images.

        transform : torchvision.transforms
            Torchvision transforms to be applied to each image.
        """
        self.img_paths = glob.glob(path_to_data + '/*.jpg')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = Image.open(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0
