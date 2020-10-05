import os

import torch
import torchvision
import tqdm

from resnet import resnet18
from datasets import gtsrb
from utils import setup, ram, vram, accuracy
from evaluate import eval_nn


def main():
    args = setup()

    print("Loading model")
    """
    model_class = getattr(torchvision.models, args.model)
    if args.model in ['googlenet', 'inception_v3']:
        model = model_class(pretrained=True, aux_logits=False)
    else:
        model = model_class(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 43)
    """
    model = resnet18(num_classes=43)
    model.to(args.device).train()
    if args.parallel:
        model = torch.nn.parallel.DataParallel(model)
    train_loader, val_loader = gtsrb(args.data_dir, batch_size=args.batch_size, workers=args.workers)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = torch.nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, optimizer, criterion, args.epochs, args.lr, args.device)

    path = os.path.join(args.root_dir, 'weights', f"{args.model}_{args.data}.pth")
    torch.save(model.state_dict(), path)


def train(model, train_loader, val_loader, optimizer, criterion, epochs, learning_rate, device):
    train_loss = 0
    for epoch in range(epochs):
        model.train()
        train_loader = tqdm.tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]")
        for batch, (images, labels) in enumerate(train_loader):
            train_loader.set_postfix({'Train loss': train_loss / ((batch + 1) + (epoch * len(train_loader))),
                                      'Train acc.': train_acc if batch > 10 else 0,
                                      'RAM': ram(),
                                      'VRAM': vram()})

            logits = model(images.to(device, non_blocking=True))
            loss = criterion(logits, labels.to(device, non_blocking=True))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % (len(train_loader) // 10) == 0:
                train_loss += loss.detach().cpu().numpy()
                train_acc = accuracy(logits.detach().cpu().numpy(), labels.numpy())

        eval_nn(model, val_loader, device, verbose=True)
        # adjust_learning_rate(optimizer, epoch, every=5, init_lr=learning_rate)


def adjust_learning_rate(optimizer, epoch, every=1, init_lr=0.001):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // every))
    print(f"Changing learning rate from {init_lr} to {lr}.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()
