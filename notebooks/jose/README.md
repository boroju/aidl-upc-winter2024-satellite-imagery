
# 🔥 Wildfire Segmentation Project - UPC Master in Artificial Intelligence with Deep Learning

## Introduction
Welcome to our GitHub repository for the Wildfire Segmentation Project, developed as the capstone project for the Postgraduate Program in Artificial Intelligence with Deep Learning. This project is aimed at leveraging the power of deep learning to address the critical and increasingly prevalent issue of wildfires. By employing semantic segmentation techniques on satellite images, we aim to accurately identify and predict wildfire-affected areas, contributing to better management, response, and mitigation strategies.

## Project Objective
The primary objective of this project is to apply semantic segmentation to satellite imagery for the precise identification of areas affected by wildfires. Utilizing state-of-the-art deep learning techniques, we aim to develop a model capable of distinguishing between burned and unburned areas, thereby facilitating more effective wildfire monitoring and management. The project underscores the importance of advanced AI in environmental protection and disaster response efforts.


## Dataset Overview
Our unique dataset is crafted using images from the National Agriculture Imagery Program (NAIP) and the Moderate Resolution Imaging Spectroradiometer (MODIS) Fire and Thermal Anomalies datasets, accessed through Google Earth Engine (GEE). Specifically, the dataset includes:

- **NAIP Imagery:** High-resolution aerial imagery from NAIP, crucial for detailed landscape features. The NAIP dataset can be accessed [here](https://developers.google.com/earth-engine/datasets/catalog/USDA_NAIP_DOQQ).
- **MODIS Thermal Anomalies and Fire Data:** Essential data for identifying wildfire occurrences, sourced from the MODIS dataset. This dataset can be found [here](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD14A1).

### Manual Annotation

For the accurate identification of burned and not burned areas corresponding to the wildfires in Paradise, California, and Cameron Peak, Colorado, we performed manual annotations. By visually inspecting the BurnDate mask provided by MODIS within GEE, we systematically marked areas affected by the fires. This meticulous process resulted in a precise dataset consisting of images and corresponding masks for training our model, with a total of 145 burned areas and 182 not burned areas annotated. This step was crucial for ensuring the high quality and reliability of our training dataset, directly impacting the model's ability to accurately segment satellite imagery based on the presence of wildfires.

![Paradise data points](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/notebooks/jose/assets/paradise_wildfire_data_points.jpg?raw=true)

### Data Processing and Preparation
The creation of our dataset involved several key steps:
- **Data Acquisition:** Programmatic access and download of high-resolution NAIP imagery and MODIS thermal anomalies data from GEE.
- **Preprocessing:** Alignment and scaling of NAIP and MODIS data to ensure that each NAIP image was perfectly matched with its corresponding MODIS mask based on geographic location and projection.
- **Augmentation and Splitting:** The dataset was augmented and randomly split into training, validation, and testing sets to ensure robust model training and evaluation.

<p align="center">
    <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/notebooks/jose/assets/landcover.png?raw=true" width=30% height=30% alt>
    <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/notebooks/jose/assets/landcover_ndvi.png?raw=true" width=30% height=30% alt>
    <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/notebooks/jose/assets/landcover_ndwi.png?raw=true" width=30% height=30% alt>
    <br>
    <em>Caption</em>
</p>

NDVI is **used to quantify vegetation greenness** and is useful in understanding vegetation density and assessing changes in plant health. NDVI is calculated as a ratio between the red (R) and near infrared (NIR) values in traditional fashion: (NIR - R) / (NIR + R)

The **Normalized Difference Water Index** (NDWI) is known to be strongly related to the plant water content. It is therefore a very good proxy for plant water stress. The Normalized Difference Water Index (NDWI) (Gao, 1996) is a satellite-derived index from the Near-Infrared (NIR) and Short Wave Infrared (SWIR) channels.

For detailed information on the data and the process of obtaining it from GEE, please refer to the notebook `segmentation_model_data.ipynb` in this repository. This notebook contains comprehensive details to facilitate the understanding and reproduction of the analysis conducted in this project.


## Model Training with PyTorch
Our model is based on a transfer learning approach using DeepLabv3 with a ResNet50 backbone, pre-trained on a relevant dataset to kickstart the learning process. We further customized the model to accept inputs with additional channels (RGB, NDVI, and NDWI features) and output two layers representing the probabilities assigned to each pixel for being part of a burned or unburned area.

### Training Details
- **Environment:** Training was conducted on Google Colab Pro with GPU acceleration and high-memory VMs to handle the computational demands.
- **Monitoring and Experimentation:** We utilized TensorBoard for real-time monitoring of training metrics and experimented with various configurations using Weights & Biases (W&B).
- **Loss Function:** The model employs Focal Loss to address the class imbalance issue inherent in our dataset, focusing on harder-to-classify examples for improved performance.
- **Early Stopping:** To prevent overfitting, we implemented an EarlyStopping mechanism based on validation loss.

## Getting Started
This repository contains all the code and notebooks required to replicate our project, from data preprocessing in GEE to model training and evaluation. Follow the instructions in each notebook for a step-by-step guide to our process.

### Prerequisites
- Access to Google Earth Engine for data acquisition.
- A Google Colab Pro account for model training.
- Basic familiarity with PyTorch and satellite image processing.

## Conclusion
Through this project, we demonstrate the potential of deep learning in enhancing our ability to monitor and respond to wildfires. We invite you to explore our notebooks, replicate our findings, and contribute to this vital field of research.

Thank you for visiting our repository.
