import numpy as np
import torch
from typing import Tuple
from utils import compute_accuracy

# this ensures that the current MacOS version is at least 12.3+
mps_available = torch.backends.mps.is_available()
# this ensures that the current PyTorch installation was built with MPS activated.
mps_built = torch.backends.mps.is_built()

if mps_available and mps_built:
    device = torch.device("mps")


@torch.no_grad()  # decorator: avoid computing gradients
def test_single_epoch(
        test_loader: torch.utils.data.DataLoader,
        my_model: torch.nn.Module,
        criterion: torch.nn.functional,
) -> Tuple[float, float]:
    # Dectivate the train=True flag inside the model
    my_model.eval()

    test_loss = []
    acc = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = my_model(data)

        # Apply the loss criterion and accumulate the loss
        test_loss.append(criterion(output, target).item())

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
               my_model: torch.nn.Module,
               criterion: torch.nn.functional) -> Tuple[float, float]:
    my_model.eval()

    loss_list = []
    acc = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = my_model(data)

        loss_list.append(criterion(output, target).item())

        acc += compute_accuracy(output, target)

    avg_loss = np.mean(loss_list)
    avg_acc = 100. * acc / len(test_loader.dataset)

    return avg_loss, avg_acc
