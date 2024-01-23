import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from utils import compute_accuracy
from model import WildfireBinClassifier
from test_model import test_single_epoch, test_model
from eval_model import eval_single_epoch
import matplotlib

matplotlib.use('agg')  # Cambiado a 'agg'
import matplotlib.pyplot as plt

# this ensures that the current MacOS version is at least 12.3+
mps_available = torch.backends.mps.is_available()
# this ensures that the current PyTorch installation was built with MPS activated.
mps_built = torch.backends.mps.is_built()

if mps_available and mps_built:
    device = torch.device("mps")


def train_single_epoch(train_loader: torch.utils.data.DataLoader,
                       my_model: torch.nn.Module,
                       optimizer: torch.optim,
                       criterion: torch.nn.functional,
                       epoch: int,
                       log_interval: int,
                       ) -> Tuple[float, float]:
    # Activate the train = True flag inside the model
    my_model.train()

    train_loss = []
    acc = 0.
    avg_weight = 0.1
    for batch_idx, (data, target) in enumerate(train_loader):

        # Move input data and labels to the device
        data, target = data.to(device), target.to(device)

        # Set my_model gradients to 0.
        optimizer.zero_grad()

        # Forward batch of images through the my_model
        output = my_model(data)

        # Compute loss
        loss = criterion(output, target)

        # Compute backpropagation
        loss.backward()

        # Update parameters of the my_model
        optimizer.step()

        # Compute metrics
        acc += compute_accuracy(output, target)
        train_loss.append(loss.item())

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    avg_acc = 100. * acc / len(train_loader.dataset)

    return np.mean(train_loss), avg_acc


def train_model(train_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                eval_loader: torch.utils.data.DataLoader,
                hparams: dict
                ) -> WildfireBinClassifier:
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    val_losses = []
    val_accs = []

    my_model = WildfireBinClassifier().to(device)

    optimizer = torch.optim.Adam(my_model.parameters(), lr=hparams['learning_rate'],
                                 weight_decay=hparams['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(hparams['num_epochs']):
        # Compute & save the average training loss for the current epoch
        train_loss, train_acc = train_single_epoch(train_loader, my_model, optimizer, criterion, epoch,
                                                   hparams["log_interval"])
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Compute & save the average test loss & accuracy for the current epoch
        test_loss, test_acc = test_single_epoch(
            test_loader=test_loader,
            my_model=my_model,
            criterion=criterion
        )
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        val_loss, val_acc = eval_single_epoch(
            eval_loader=eval_loader,
            my_model=my_model,
            criterion=criterion
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    final_test_loss, final_test_acc = test_model(
        test_loader=test_loader,
        my_model=my_model,
        criterion=criterion
    )

    # Plot the plots of the learning curves
    _plot_learning_curves(train_losses, train_accs, val_losses, val_accs)

    # Print or plot final test results
    print('Final Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(
        final_test_loss, final_test_acc))

    return my_model


def _plot_learning_curves(train_losses, train_accs, val_losses, val_accs):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='eval')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.plot(train_accs, label='train')
    plt.plot(val_accs, label='eval')
    plt.savefig('learning_curves.png')  # save instead of show
    # plt.show()  # Not needed when using 'agg'
