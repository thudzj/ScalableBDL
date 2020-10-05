"""This is the code featured in the 60-seconds blitz to Laplace approximation found in the readme."""

# Standard imports
import torch
import torchvision
import tqdm

# From the repository
from fisher import KFAC
from lenet5 import lenet5
from sampling import invert_factors

# Change this to 'cuda' if you have a working GPU.
device = 'cpu'

# We will use the provided LeNet-5 variant pre-trained on MNIST.
model = lenet5(pretrained='mnist', device=device).to(device)

# Standard PyTorch dataset location
torch_data = "~/.torch/datasets"
mnist = torchvision.datasets.MNIST(root=torch_data,
                                   train=True,
                                   transform=torchvision.transforms.ToTensor(),
                                   download=True)
train_data = torch.utils.data.DataLoader(mnist, batch_size=100, shuffle=True)

# Decide which loss criterion and curvature approximation to use.
criterion = torch.nn.CrossEntropyLoss().to(device)
kfac = KFAC(model)

# Standard PyTorch training loop:
for images, labels in tqdm.tqdm(train_data):
    logits = model(images.to(device))

    # We compute the 'true' Fisher information matrix (FIM),
    # by taking the expectation over the model distribution.
    # To obtain the empirical FIM, just use the labels from
    # the data distribution directly.
    dist = torch.distributions.Categorical(logits=logits)
    sampled_labels = dist.sample()

    loss = criterion(logits, sampled_labels)
    model.zero_grad()
    loss.backward()

    # We call 'estimator.update' here instead of 'optimizer.step'.
    kfac.update(batch_size=images.size(0))

# Access and invert the curvature information to perform Bayesian inference.
# 'Norm' (tau) and 'scale' (N) are the two hyperparameters of Laplace approximation.
# See the tutorial notebook for for an in-depth example and explanation.
factors = list(kfac.state.values())
inv_factors = invert_factors(factors, norm=0.5, scale=1, estimator='kfac')
