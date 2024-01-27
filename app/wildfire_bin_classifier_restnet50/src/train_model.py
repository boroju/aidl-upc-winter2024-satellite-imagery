import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from utils import compute_accuracy
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import resnet50
from test_model import test_single_epoch, test_model
from val_model import val_single_epoch
import matplotlib

matplotlib.use('agg')  # Cambiado a 'agg'
import matplotlib.pyplot as plt

device = torch.device("cpu")


def train_single_epoch(train_loader: torch.utils.data.DataLoader,
                       model: torch.nn.Module,
                       optimizer: torch.optim,
                       criterion: torch.nn.functional,
                       epoch: int,
                       log_interval: int,
                       ) -> Tuple[float, float]:
    # Activate the train=True flag inside the model
    model.train()

    train_loss = []
    acc = 0.
    avg_weight = 0.1
    for batch_idx, (data, target) in enumerate(train_loader):

        # Move input data and labels to the device
        data, target = data.to(device), target.to(device)

        # Set model gradients to 0.
        optimizer.zero_grad()

        # Forward batch of images through the model
        output = model(data).to(device)

        # Compute loss
        loss = criterion(output.squeeze(), target.float())

        # Compute backpropagation
        loss.backward()

        # Update parameters of the model
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
                val_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                hparams: dict
                ) -> torch.nn.Module:
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # Define and load pre-trained ResNet50 model
    model = resnet50(pretrained=True)

    # Change the output layer for binary classification
    num_filters = model.fc.in_features
    model.fc = nn.Linear(num_filters, 1)

    # Define the optimizer and the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(hparams['num_epochs']):
        # Compute & save the average training loss for the current epoch
        train_loss, train_acc = train_single_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            log_interval=hparams["log_interval"]
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = val_single_epoch(
            val_loader=val_loader,
            model=model,
            criterion=criterion
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

    final_test_loss, final_test_acc = test_model(
        test_loader=test_loader,
        model=model,
        criterion=criterion
    )

    # Plot the plots of the learning curves
    _plot_learning_curves(train_losses, train_accs, val_losses, val_accs)

    # Print or plot final test results
    print('Final Test set: Average loss: {:.4f}, Accuracy: {:.2f}%'.format(
        final_test_loss, final_test_acc))

    return model


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
