from torchvision.models import resnet50
import torch.nn as nn
import torch
from main import hparams

# Define and load pre-trained ResNet50 model
model = resnet50(pretrained=True)

# Change the output layer for binary classification
num_filters = model.fc.in_features
model.fc = nn.Linear(num_filters, 1)

# Define the optimizer and the loss function
optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
criterion = nn.BCEWithLogitsLoss()