# activephasemap
Active sampling-based strategies for constructing 'phase' maps using spectroscopic measurements

## Installation instructions:
Create a conda environment
```bash
conda create --prefix ./location/envs/activephasemap/ python==3.11
```
and activate it in a command shell
```bash
conda activate activephasemap
```

1. Install botorch (this installs majority of the packages required)
```bash
pip install botorch
```

2. Install other common packages:
```bash
pip install matplotlib pandas openpyxl seaborn torchvision
```

3. Install the `apdist` package from [here](https://github.com/kiranvad/Amplitude-Phase-Distance/tree/main)

4. Install `activephasemap` from the repo:
```bash
pip install -e .
```
or directly from github:
```bash
pip install git+https://github.com/pozzo-research-group/activephasemap.git
``` 

For an application to creating a differentiable self-driving labs of gold nanoparticles, see [here](https://github.com/pozzo-research-group/papers/tree/activephasemap-preprint/seed-AuNP-phasemaps)