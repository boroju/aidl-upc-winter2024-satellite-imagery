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
def val_single_epoch(
        val_loader: torch.utils.data.DataLoader,
        my_model: torch.nn.Module,
        criterion: torch.nn.functional
        ) -> Tuple[float, float]:

    # Dectivate the train=True flag inside the model
    my_model.eval()

    eval_loss = []
    acc = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)

        output = my_model(data)

        # Apply the loss criterion and accumulate the loss
        eval_loss.append(criterion(output, target).item())

        # compute number of correct predictions in the batch
        acc += compute_accuracy(output, target)

    # Average accuracy across all correct predictions batches now
    eval_acc = 100. * acc / len(val_loader.dataset)
    eval_loss = np.mean(eval_loss)
    print('\Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        eval_loss, acc, len(val_loader.dataset), eval_acc,
        ))
    return eval_loss, eval_acc
