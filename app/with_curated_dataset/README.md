# Dealing with Satellite Imagery data is not easy at all

In the initial phase of the project, we were uncertain about the feasibility of building our own dataset. During this stage, we explored various satellite products, such as:

- [MODIS Thermal Anomalies/Fire 8-Day](https://planetarycomputer.microsoft.com/dataset/modis-14A2-061) for Fire Mask
- [Landsat Collection 2 Level-1](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l1) and [Landsat Collection 2 Level-2](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2) for Land Cover

We realized that constructing our satellite imagery dataset would be quite challenging. So, we decided to split the work force into two groups. One group would continue investigating the feasibility of building our own dataset, while the other group would find a curated dataset.

# Experiment A: Working with a curated dataset

## Contents

1. [Goal](#Goal)
2. [Dataset](#Dataset)
3. [Source](#Source)
4. [Description](#Description)
5. [Classes](#Classes)
6. [CNN Model Architecture](#CNN-Model-Architecture)
7. [Achievement](#Achievement)

## Goal

Build a binary classifier model that can predict whether an area is at risk of a wildfire or not.

## Dataset

After a few days of research, we found a curated dataset on [Kaggle](https://www.kaggle.com/). The dataset is called [Wildfire Prediction Dataset (Satellite Images)](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset). This dataset contains satellite images of areas that previously experienced wildfires in Canada.

## Source

Refer to Canada's website for the original wildfires data: [Forest Fires - Open Government Portal](https://open.canada.ca/data/en/dataset/9d8f219c-4df0-4481-926f-8a2a532ca003)

Original license for the data: [Creative Commons 4.0 Attribution (CC-BY) license – Quebec](https://www.donneesquebec.ca/fr/licence/)

## Description

This dataset contains satellite images (350x350px) in 2 classes:

- `wildfire`: 22710 images
- `nowildfire`: 20140 images

The data was divided into train, test and validation with these percentages:

- Train: ~70% (`wildfire`: 15750 images, `nowildfire`: 14500 images)
- Test: ~15% (`wildfire`: 3480 images, `nowildfire`: 2820 images)
- Validation: ~15% (`wildfire`: 3480 images, `nowildfire`: 2820 images)

### Collection Methodology

Coordinates found in the source file and extracted satellite images using MapBox API to 350x350px .jpg images

## Classes

### 1- wildfire

Sample images of wildfire class:

<p float="left">
  <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/img/kaggle_data/wildfire/-60.9878%2C50.4112.jpg" width="350" title="wildfire_img_1" />
  <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/img/kaggle_data/wildfire/-61.5607%2C50.52878.jpg" width="350" title="wildfire_img_2" />
</p>

### 2- nowildfire

Sample images of wildfire class:

<p float="left">
  <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/img/kaggle_data/nowildfire/-73.7181%2C45.486459.jpg" width="350" title="nowildfire_img_1" />
  <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/img/kaggle_data/nowildfire/-73.8275%2C45.552381.jpg" width="350" title="nowildfire_img_2" />
</p>

## CNN Model Architecture

### Code

Neural Network Architecture Code:

```python
WildfireBinClassifier(
  (conv1): ConvBlock(
    (conv): Conv2d(3, 8, kernel_size=(2, 2), stride=(1, 1), bias=False)
    (relu): ReLU(inplace=True)
    (maxpool_2d): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (conv2): ConvBlock(
    (conv): Conv2d(8, 16, kernel_size=(2, 2), stride=(1, 1), bias=False)
    (relu): ReLU(inplace=True)
    (maxpool_2d): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (conv3): ConvBlock(
    (conv): Conv2d(16, 32, kernel_size=(2, 2), stride=(1, 1), bias=False)
    (relu): ReLU(inplace=True)
    (maxpool_2d): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (mlp): Sequential(
    (0): Dropout(p=0.4, inplace=False)
    (1): Linear(in_features=56448, out_features=2048, bias=True)
    (2): ReLU()
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=2048, out_features=300, bias=True)
    (5): ReLU()
    (6): Dropout(p=0.5, inplace=False)
    (7): Linear(in_features=300, out_features=2, bias=True)
    (8): ReLU()
    (9): Softmax(dim=1)
  )
)
```

### Diagram

Neural Network Architecture Diagram:

<img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/wildfire_bin_classifier/cnn_arch/diagram/wildfire_bin_classifier_archDiagram.png" title="wildfire_bin_classifier_arch" />

Code is available [here](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/app/with_curated_dataset/wildfire_bin_classifier/src/model.py)

### Transforms

Image transformations:

```python
# image transformations
image_transforms = transforms.Compose([
    transforms.Resize((350, 350)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])
```

Code is available [here](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/app/with_curated_dataset/wildfire_bin_classifier/src/main.py#L54)

### Hyperparameters

Used for training the model:

| Hyperparameter   | Value     |
|------------------|-----------|
| Batch Size       | 256       |
| Num Epochs       | 10        |
| Test Batch Size  | 256       |
| Learning Rate    | 1e-3      |
| Weight Decay     | 1e-5      |
| Log Interval     | 10        |

Code is available [here](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/app/with_curated_dataset/wildfire_bin_classifier/src/checkpoints/v1/hparams.py)

### Training 

#### Resources

We have used 2 different resources to train the model. 

1. `CUDA` by enabling GPU on **Google Colab**.
2. `MPS` on **Apple MacBook Pro with M1 chip with 32 GB RAM**.

For this experiment, the 2nd option (`MPS`) was the one chosen for the final training to avoid any issues with the internet connection or runtime disconnection.

Code is available [here](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/app/with_curated_dataset/wildfire_bin_classifier/src/train_model.py)

#### Evidence

```python
Train Epoch: 9 [28160/30250 (93%)]	Loss: 0.358686
\Validation set: Average loss: 0.3591, Accuracy: 5853/6300 (93%)

Final Test set: Average loss: 0.3521, Accuracy: 93.68%
Saving model to /projects/aidl-upc-winter2024-satellite-imagery/app/wildfire_bin_classifier/src/checkpoints...
Model saved successfully!
```

#### Plot

<img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/app/with_curated_dataset/wildfire_bin_classifier/src/checkpoints/v1/learning_curves.png" title="CNN_learning_curves" />

#### Checkpoint

Model checkpoint is available [here](https://drive.google.com/file/d/1dPMRYltQbwkPT71I4jc_P6RraxFOnfhM/view?usp=sharing)

## Achievement

At this point, we successfully built a binary classifier model that can predict whether an area is at risk of a wildfire or not. This was accomplished from scratch using curated satellite imagery data.

From a given satellite image with similar characteristics to the ones within the [curated dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset), the model can predict the class (1 - 'wildfire' or 2 - 'nowildfire') with an accuracy of 93.68%.

### Inference

Classes

```python
{'nowildfire': 0, 'wildfire': 1}
```

#### Example with class nowildfire

Given image:

<img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/wildfire_bin_classifier/inference/-73.58813,45.482892.jpg" title="inference_class0" />

Prediction: `nowildfire` ✅

#### Example with class nowildfire

Given image:

<img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/wildfire_bin_classifier/inference/-65.2239,49.10492.jpg" title="inference_class1" />

Prediction: `wildfire` ✅

### Application Code

Available [here](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/tree/main/app/with_curated_dataset/wildfire_bin_classifier).
