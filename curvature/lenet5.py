import os
import torch


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def lenet5(pretrained="", device=None):
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 6, 5, padding=2),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Conv2d(6, 16, 5),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        Flatten(),
        torch.nn.Linear(16 * 5 * 5, 120),
        torch.nn.ReLU(),
        torch.nn.Linear(120, 84),
        torch.nn.ReLU(),
        torch.nn.Linear(84, 10)
    )

    if pretrained:
        state_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'lenet5_mnist.pth')
        state_dict = torch.load(state_path, map_location=device)
        model.load_state_dict(state_dict)

    return model
