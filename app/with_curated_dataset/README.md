# Dealing with Satellite Imagery data is not easy at all

In the initial phase of the project, we were uncertain about the feasibility of building our own dataset. During this stage, we explored various satellite products, such as:

- [MODIS Thermal Anomalies/Fire 8-Day](https://planetarycomputer.microsoft.com/dataset/modis-14A2-061) for Fire Mask
- [Landsat Collection 2 Level-1](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l1) and [Landsat Collection 2 Level-2](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2) for Land Cover

We realized that constructing our satellite imagery dataset would be quite challenging. So, we decided to split the work force into two groups. One group would continue investigating the feasibility of building our own dataset, while the other group would find a curated dataset.

# Experiment A: Working with a curated dataset

After a few days of research, we found a curated dataset on [Kaggle](https://www.kaggle.com/). The dataset is called [Wildfire Prediction Dataset (Satellite Images)](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset). This dataset contains satellite images of areas that previously experienced wildfires in Canada.

## Source

Refer to Canada's website for the original wildfires data: [Forest Fires - Open Government Portal](https://open.canada.ca/data/en/dataset/9d8f219c-4df0-4481-926f-8a2a532ca003)

Original license for the data: [Creative Commons 4.0 Attribution (CC-BY) license – Quebec](https://www.donneesquebec.ca/fr/licence/)

## Description

This dataset contains satellite images (350x350px) in 2 classes:

- `wildfire`: 22710 images
- `no_wildfire`: 20140 images

The data was divided into train, test and validation with these percentages:

- Train: ~70% (`wildfire`: 15750 images, `no_wildfire`: 14500 images)
- Test: ~15% (`wildfire`: 3480 images, `no_wildfire`: 2820 images)
- Validation: ~15% (`wildfire`: 3480 images, `no_wildfire`: 2820 images)

## How it looks

Using Longitude and Latitude coordinates for each wildfire spot (> 0.01 acres burned) satellite images were extracted by using MapBox API to create a convenient format of the data for deep learning.

### Classes

### wildfire

Sample images of wildfire class:

![sample_img_wildfire1](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/img/kaggle_data/wildfire/-60.9878%2C50.4112.jpg)

![sample_img_wildfire2](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/img/kaggle_data/wildfire/-61.5607%2C50.52878.jpg)

### no_wildfire

![sample_img_no_wildfire1](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/img/kaggle_data/no_wildfire/-73.7181%2C45.486459.jpg)

![sample_img_no_wildfire2](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/img/kaggle_data/no_wildfire/-73.8275%2C45.552381.jpg)

## Goal

Build a binary classifier model that can predict whether an area is at risk of a wildfire or not.
