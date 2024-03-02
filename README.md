  
# Wildfire Probability Prediction Using Satelite Images

Welcome to our GitHub repository for the Wildfire Segmentation Project, developed as the capstone project for the Postgraduate Program in Artificial Intelligence with Deep Learning. 
This project is aimed at leveraging the power of deep learning to address the critical and increasingly prevalent issue of wildfires. 
By employing semantic segmentation techniques on satellite images, we aim to accurately identify and predict wildfire-affected areas, contributing to better management, response, and mitigation strategies.

The original code is the Final Project delivery for the [UPC Artificial Intelligence with Deep Learning Postgraduate Course](https://www.talent.upc.edu/ing/estudis/formacio/curs/310401/postgraduate-course-artificial-intelligence-deep-learning/) 2023-2024 edition, authored by:

- [Julian Boronat](https://es.linkedin.com/in/boronatje)
- [Jose Cajide](https://es.linkedin.com/in/jose-cajide-385bb8199)
- [Paula Espina](https://es.linkedin.com/in/paulaec)
- [Ignacio Sánchez](https://es.linkedin.com/in/igsalvar/en)

Advised by professor [Amanda Duarte](https://es.linkedin.com/in/amanda-cardoso-duarte)

**Project Objective**

Study the **application of semantic segmentation, as a computer vision task, to satellite imagery for the precise identification of areas affected by wildfires**. 

By utilizing *state-of-the-art deep learning* techniques, we aim to develop a model capable of distinguishing between burned and unburned areas, thereby facilitating more effective wildfire monitoring and management. 

The project underscores **the importance of advanced AI in environmental protection and disaster** response efforts.


**TABLE OF CONTENTS**

1. [General Configuration:](#GeneralConfiguration)
2. [Models](#Models)
3. [Dealing with Satellite Imagery data is not easy at all](#DealingwithSatelliteImagerydataisnoteasyatall)
4. [Experiments](#Experiments)
5. [Final Conclusions](#FinalConclusions)
6. [Future Usage](#FutureUsage)
7. [Bibliography](#Bibliography)


## 1. General Configuration

To use the database already prepared go to (add link) and download the zip.
1. Download the zip and check the name
2. Check that the folder name macthes the one on the Colab.

## 2. Models

- VGG15
- ResNET18
- ResNet50
- DeepLabV3

## 3. Dealing with Satellite Imagery data is not easy at all

In the initial phase of the project, we were uncertain about the feasibility of building our own dataset. During this stage, we explored various satellite products, such as:

- [MODIS Thermal Anomalies/Fire 8-Day](https://planetarycomputer.microsoft.com/dataset/modis-14A2-061)
- [Landsat Collection 2 Level-1](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l1) and [Landsat Collection 2 Level-2](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2)

We realized that constructing our satellite imagery dataset would be quite challenging. So, we decided to split the work force into two groups. One group would continue investigating the feasibility of building our own dataset, while the other group would find a curated dataset using the following data sources:

- [National Agriculture Imagery Program (NAIP)](https://naip-usdaonline.hub.arcgis.com/)
- [The Terra Moderate Resolution Imaging Spectroradiometer (MODIS) Thermal Anomalies and Fire 8-Day (MOD14A2) Version 6.1](https://lpdaac.usgs.gov/products/mod14a2v061/) 
- [The Terra and Aqua combined MCD64A1 Version 6.1 Burned Area](https://lpdaac.usgs.gov/products/mcd64a1v061/)

## 4. Experiments

### Experiment A: Binary classifier model for wildfire risk prediction

**Contents**

1. [Goal](#Goal)
2. [Dataset](#Dataset)
3. [Source](#Source)
4. [Description](#Description)
5. [Classes](#Classes)
6. [CNN Model Architecture](#CNN-Model-Architecture)
7. [Achievement](#Achievement)


#### 1. Goal

Build a binary classifier model that can predict whether an area is at risk of a wildfire or not.

#### 2. Dataset

After a few days of research, we found a curated dataset on [Kaggle](https://www.kaggle.com/). The dataset is called [Wildfire Prediction Dataset (Satellite Images)](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset). This dataset contains satellite images of areas that previously experienced wildfires in Canada.

**Source**

Refer to Canada's website for the original wildfires data: [Forest Fires - Open Government Portal](https://open.canada.ca/data/en/dataset/9d8f219c-4df0-4481-926f-8a2a532ca003)

Original license for the data: [Creative Commons 4.0 Attribution (CC-BY) license – Quebec](https://www.donneesquebec.ca/fr/licence/)

**Description**

This dataset contains satellite images (350x350px) in 2 classes:

- `wildfire`: 22710 images
- `nowildfire`: 20140 images

The data was divided into train, test and validation with these percentages:

- Train: ~70% (`wildfire`: 15750 images, `nowildfire`: 14500 images)
- Test: ~15% (`wildfire`: 3480 images, `nowildfire`: 2820 images)
- Validation: ~15% (`wildfire`: 3480 images, `nowildfire`: 2820 images)

**Collection Methodology**

Coordinates found in the source file and extracted satellite images using MapBox API to 350x350px .jpg images

**Classes**

a) wildfire:

Sample images of wildfire class:

<p float="left">
  <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/img/kaggle_data/wildfire/-60.9878%2C50.4112.jpg" width="350" title="wildfire_img_1" />
  <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/img/kaggle_data/wildfire/-61.5607%2C50.52878.jpg" width="350" title="wildfire_img_2" />
</p>

b) no wildfire:

Sample images of wildfire class:

<p float="left">
  <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/img/kaggle_data/nowildfire/-73.7181%2C45.486459.jpg" width="350" title="nowildfire_img_1" />
  <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/img/kaggle_data/nowildfire/-73.8275%2C45.552381.jpg" width="350" title="nowildfire_img_2" />
</p>

#### 3. CNN Model Architecture

**Code**

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

**Diagram**

Neural Network Architecture Diagram:

<img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/wildfire_bin_classifier/cnn_arch/diagram/wildfire_bin_classifier_archDiagram.png" title="wildfire_bin_classifier_arch" />

Code is available [here](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/app/with_curated_dataset/wildfire_bin_classifier/src/model.py)

**Transforms**

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

**Hyperparameters**

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

#### 4.Training 

**Resources**

We have used 2 different resources to train the model. 

1. `CUDA` by enabling GPU on **Google Colab**.
2. `MPS` on **Apple MacBook Pro with M1 chip with 32 GB RAM**.

For this experiment, the 2nd option (`MPS`) was the one chosen for the final training to avoid any issues with the internet connection or runtime disconnection.

Code is available [here](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/app/with_curated_dataset/wildfire_bin_classifier/src/train_model.py)

**Results**

```python
Train Epoch: 9 [28160/30250 (93%)]	Loss: 0.358686
\Validation set: Average loss: 0.3591, Accuracy: 5853/6300 (93%)

Final Test set: Average loss: 0.3521, Accuracy: 93.68%
Saving model to /projects/aidl-upc-winter2024-satellite-imagery/app/wildfire_bin_classifier/src/checkpoints...
Model saved successfully!
```

**Plot**

<img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/app/with_curated_dataset/wildfire_bin_classifier/src/checkpoints/v1/learning_curves.png" title="CNN_learning_curves" />

**Checkpoint**

Model checkpoint is available [here](https://drive.google.com/file/d/1dPMRYltQbwkPT71I4jc_P6RraxFOnfhM/view?usp=sharing)

#### 5. Achievement

At this point, we successfully built a binary classifier model that can predict whether an area is at risk of a wildfire or not. This was accomplished from scratch using curated satellite imagery data.

From a given satellite image with similar characteristics to the ones within the [curated dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset), the model can predict the class (1 - 'wildfire' or 2 - 'nowildfire') with an accuracy of 93.68%.

**Inference**

Classes

```python
{'nowildfire': 0, 'wildfire': 1}
```

**Example with class nowildfire**

Given image:

<img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/wildfire_bin_classifier/inference/-73.58813,45.482892.jpg" title="inference_class0" />

Prediction: `nowildfire` ✅

**Example with class nowildfire**

Given image:

<img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/wildfire_bin_classifier/inference/-65.2239,49.10492.jpg" title="inference_class1" />

Prediction: `wildfire` ✅

**Application Code**

Available [here](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/tree/main/app/with_curated_dataset/wildfire_bin_classifier).

*Conclusions*

- This model works, as seen before the accuracy reaches **93.68%**. 
- Now that the model works, we want to try with another dataset made of masks obtained from NASA Satelites.
- The new model will use a more powerful CNN performed on AWS.


### Experiment B: Semantic segmentation on satellite images to identify and predict wildfire-affected areas

#### Dataset Overview
We curated a **custom dataset** by incorporating imagery from the National Agriculture Imagery Program (NAIP) and the Moderate Resolution Imaging Spectroradiometer (MODIS) Fire and Thermal Anomalies datasets, retrieved through Google Earth Engine (GEE). The dataset encompasses:

- **NAIP Imagery:** High-resolution aerial imagery from NAIP, crucial for detailed landscape features. The NAIP dataset can be accessed [here](https://developers.google.com/earth-engine/datasets/catalog/USDA_NAIP_DOQQ).
- **MODIS Thermal Anomalies and Fire Data:** Essential data for identifying wildfire occurrences, sourced from the MODIS dataset. This dataset can be found [here](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD14A1).

**Manual Annotation**

For the accurate identification of burned and not burned areas corresponding to the wildfires in Paradise, California, and Cameron Peak, Colorado, we performed [manual annotations](https://code.earthengine.google.com/0a897a0931637f0f86394ff412d6fdf3?hideCode=true). By visually inspecting the BurnDate mask provided by MODIS within GEE, we systematically marked areas affected by the fires. This meticulous process resulted in a precise dataset consisting of images and corresponding masks for training our model, with a total of 145 burned areas and 182 not burned areas annotated. This step was crucial for ensuring the high quality and reliability of our training dataset, directly impacting the model's ability to accurately segment satellite imagery based on the presence of wildfires.

![Paradise data points](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/notebooks/jose/assets/paradise_wildfire_data_points.jpg?raw=true)

**Data Processing and Preparation**

The creation of our dataset involved several key steps:
- **Data Acquisition:** Programmatic access and download of high-resolution NAIP imagery and MODIS thermal anomalies data from GEE.
- **Preprocessing:** Alignment and scaling of NAIP and MODIS data to ensure that each NAIP image was [perfectly matched](https://code.earthengine.google.com/9e8459b1f855f28f6dd91c3afa1d17fb?hideCode=true) with its corresponding MODIS mask based on geographic location and projection.
- **Augmentation and Splitting:** The dataset was augmented and randomly split into training, validation, and testing sets to ensure robust model training and evaluation.

**Satellite Imagery for Segmentation Algorithm Input**

<p align="center">
    <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/notebooks/jose/assets/landcover.png?raw=true" width=30% height=30% alt>
    <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/notebooks/jose/assets/landcover_ndvi.png?raw=true" width=30% height=30% alt>
    <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/notebooks/jose/assets/landcover_ndwi.png?raw=true" width=30% height=30% alt>
    <br>
    <em>From left to right: Landcover image showcasing diverse surface features, NDVI visualization revealing vegetation health, and NDWI mask highlighting water bodies.</em>
</p>

Crucially, these images, depicting landcover, NDVI, and NDWI, serve as the foundational dataset feeding into the segmentation algorithm. Each piece of information captured in these images contributes to training the model to distinguish and identify specific features, enabling precise semantic segmentation for wildfire prediction.

**NDVI (Normalized Difference Vegetation Index):** NDVI is a vegetation index commonly derived from satellite imagery. It quantifies the presence and health of vegetation by comparing the reflectance of near-infrared light to that of red light. Higher NDVI values typically indicate healthier and more abundant vegetation. [Watch it live in Google Earth Engine](https://code.earthengine.google.com/273812b048fb5ddba384f270912b1107?hideCode=true)

**NDWI (Normalized Difference Water Index):** NDWI is another spectral index, but it focuses on water content. It is derived by comparing the reflectance of green and near-infrared light. Higher NDWI values suggest the presence of water, aiding in the identification of water bodies.

**Role in Algorithm:** Including NDVI and NDWI data in the segmentation algorithm is crucial. NDVI helps in delineating vegetation, enabling the model to discern areas susceptible to wildfires. On the other hand, NDWI assists in identifying water bodies, aiding in a more comprehensive understanding of the landscape. Together, these indices enhance the algorithm's ability to accurately segment satellite imagery, contributing to effective wildfire prediction and management.

For detailed information on the data and the process of obtaining it from GEE, please refer to the notebook `segmentation_model_data.ipynb` in this repository. This notebook contains comprehensive details to facilitate the understanding and reproduction of the analysis conducted in this project.

**Notebook**: [segmentation_model_data.ipynb](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/notebooks/segmentation_model_data.ipynb)


#### Model Training with PyTorch
Our model is based on a transfer learning approach using DeepLabv3 with a ResNet50 backbone, pre-trained on a relevant dataset to kickstart the learning process. We further customized the model to accept inputs with additional channels (RGB, NDVI, and NDWI features) and output two layers representing the probabilities assigned to each pixel for being part of a burned or unburned area.

**Initial Model Training Details**

- **Environment:** Training was conducted on Google Colab Pro with GPU acceleration and high-memory VMs to handle the computational demands.
- **Monitoring and Experimentation:** We utilized TensorBoard for real-time monitoring of training metrics and experimented with various configurations using Weights & Biases (W&B).
- **Loss Function:** The model employs Focal Loss to address the class imbalance issue inherent in our dataset, focusing on harder-to-classify examples for improved performance.
- **Early Stopping:** To prevent overfitting, we implemented an EarlyStopping mechanism based on validation loss.

**Notebook**: [segmentation_model_training_demo.ipynb](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/notebooks/segmentation_model_training_demo.ipynb)

**Final Model Training and Results**

Our segmentation model was trained on AWS using the complete dataset, comprising a total of [insert number] images. The training process leveraged GPU resources in the cloud, utilizing high-capacity machines capable of efficiently processing the extensive 12 GB dataset.

Training Details:

- **Training Duration:** The model was trained over multiple epochs, with noteworthy results achieved by the end of the third epoch.
- **Hardware Configuration:** The training process made use of GPUs in the cloud, ensuring accelerated computation to handle the large dataset.
- **Hyperparameters:**
  - Model: DeepLabv3+
  - Backbone: ResNet50
  - Input Channels: 5
  - Number of Classes: 2
  - Number of Filters: 3
  - Loss Function: Focal Loss
  - Learning Rate: 0.0001
  - Patience for Early Stopping: 5
  - Freezing Backbone: True
  - Freezing Decoder: False
  - Batch Size: 6
  - Patch Size: 256
  - Training Length: 4000
  - Number of Workers: 64

Model Metrics:

- **Model Accuracy:** 0.75 (epoch 3)

These results showcase the model's capability to effectively learn from the dataset and make accurate predictions. The chosen hyperparameters, model architecture, and careful training contribute to a robust and reliable segmentation model for wildfire prediction.


#### Results

Below, in our developed application, we showcase the predictive capability of our segmentation model. Leveraging geographic coordinates, the model adeptly identifies areas at heightened risk of wildfire occurrence.

This experiment paves the way for further exploration of harnessing satellite imagery's metadata richness alongside deep learning models. The goal is to construct high-impact applications that contribute significantly to wildfire mitigation efforts.

![image](https://github.com/ColourDread/MyStuff/assets/149001130/96e5787e-8339-4953-971a-ba0f5b1d9d39)

Wildfire probability zones (marked on yellow)

#### Segmentation Experiment Conclusions 
- The model has achieved  **75% accuracy**.
- Through this project, we demonstrate the potential of deep learning in enhancing our ability to monitor and respond to wildfires. We invite you to explore our notebooks, replicate our findings, and contribute to this vital field of research.

## Final Conclusions
- As satellites are continually monitoring and photographing the planet, an application of this type can be very useful to detect particularly sensitive areas so that they can be subjected to stricter surveillance, thereby helping to prevent fires.
- We've experimented with the two datasets and two models and, in both cases, they suggest the feasibility of the project.
- The use of masks is particularly useful for:
  - Allowing the precise identification and delineation of objects or regions of interest in a satellite image.
  - Reducing noise or parts of the image that are not essential for analysis.
  - Improving model accuracy; masks enable increased model precision by helping it better understand the image structure, leading to more accurate predictions.
  - Optimizing computational resources by eliminating irrelevant information in images.


# Future usage:
- Finetune to get a better accuracy
- Develop a frontend to visualize everything

# Bibliography:
- Documentation about [TorchGeo](https://torchgeo.readthedocs.io/en/latest/api/trainers.html#torchgeo.trainers.SemanticSegmentationTask)
- Documentation about [DeepLabV3](https://smp.readthedocs.io/en/latest/models.html#deeplabv3)

Thank you for visiting our repository.
