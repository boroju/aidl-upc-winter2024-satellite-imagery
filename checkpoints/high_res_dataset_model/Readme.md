Model Accuraccy: 0.75 (epoch 3)

hparams:

ignore: weights
model: deeplabv3+
backbone: resnet50
in_channels: 5
num_classes: 2
num_filters: 3
loss: focal
class_weights: null
ignore_index: null
lr: 0.0001
patience: 5
freeze_backbone: true
freeze_decoder: false
batch_size: 6
patch_size: 256
length: 4000
num_workers: 64
paths: /opt/ml/processing/input/data
