import torch


def compute_accuracy(predicted_batch: torch.Tensor, label_batch: torch.Tensor) -> float:
    pred = predicted_batch.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acum = pred.eq(label_batch.view_as(pred)).sum().item()
    return acum


def accuracy(labels, outputs):
    preds = outputs.argmax(-1)
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc


def save_model(model, path):
    torch.save(model.state_dict(), path)
