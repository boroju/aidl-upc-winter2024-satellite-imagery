This project leverages machine learning and deep learning techniques to predict wildfire occurrences using satellite imagery. Below are the notebooks available in this repository, each serving a specific purpose in the development and analysis of the predictive model.

## Notebooks for the segmentation task:

* [Data Preparation](segmentation_model_data.ipynb): This notebook details the extraction and preprocessing steps, including the creation of a custom dataset using images from the National Agriculture Imagery Program (NAIP) and Moderate Resolution Imaging Spectroradiometer (MODIS) datasets. Learn how to download the annotated data manually selected in Google Earth Engine. The notebook covers the extraction of data from Google Earth Engine and the conversion to various formats for model training.

* [Model Training](segmentation_model_training_demo.ipynb): Explore the model training process, where we use deep learning techniques for segmentation. The notebook covers data normalization, dataloaders setup, and the training of the predictive model.

* [Prediction Demo](Nacho/end_to_end_model_creation_cloud_env.ipynb): Witness the model's predictive capabilities in action. Using geographic coordinates, the notebook demonstrates the model's ability to detect areas at high risk of wildfires.
