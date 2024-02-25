import numpy as np
import torch
from typing import Tuple
from utils import compute_accuracy

device = torch.device("cpu")


@torch.no_grad()  # decorator: avoid computing gradients
def test_single_epoch(
        test_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        criterion: torch.nn.functional,
) -> Tuple[float, float]:
    # Dectivate the train=True flag inside the model
    model.eval()

    test_loss = []
    acc = 0
    for data, target in test_loader:
        data, target = data.to(device), target

        output = model(data)

        # Apply the loss criterion and accumulate the loss
        test_loss.append(criterion(output.squeeze(), target.float()).item())

        # compute number of correct predictions in the batch
        acc += compute_accuracy(output, target)

    # Average accuracy across all correct predictions batches now
    test_acc = 100. * acc / len(test_loader.dataset)
    test_loss = np.mean(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, acc, len(test_loader.dataset), test_acc,
    ))
    return test_loss, test_acc


@torch.no_grad()  # decorator: avoid computing gradients
def test_model(test_loader: torch.utils.data.DataLoader,
               model: torch.nn.Module,
               criterion: torch.nn.functional) -> Tuple[float, float]:
    model.eval()

    loss_list = []
    acc = 0
    for data, target in test_loader:
        data, target = data.to(device), target

        output = model(data)

        loss_list.append(criterion(output.squeeze(), target.float()).item())

        acc += compute_accuracy(output, target)

    avg_loss = np.mean(loss_list)
    avg_acc = 100. * acc / len(test_loader.dataset)

    return avg_loss, avg_acc
