# Dealing with Satellite Imagery data is not easy at all

In the initial phase of the project, we were uncertain about the feasibility of building our own dataset. During this stage, we explored various satellite products, such as:

- [MODIS Thermal Anomalies/Fire 8-Day](https://planetarycomputer.microsoft.com/dataset/modis-14A2-061) for Fire Mask
- [Landsat Collection 2 Level-1](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l1) and [Landsat Collection 2 Level-2](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2) for Land Cover

And we realized that constructing our satellite imagery dataset would be quite challenging. So, we decided to split the work force into two groups. One group would continue investigating the feasibility of building our own dataset, while the other group would find a curated dataset kind of ready for a deep learning use case.

# Experiment A: Binary classifier model for wildfire risk prediction

## Contents

1. [Goal](#Goal)
2. [Dataset](#Dataset)
3. [Classes](#Classes)
4. [CNN Model Architecture](#CNN-Model-Architecture)
5. [Training](#Training)
6. [Prediction](#Prediction)
7. [Achievement](#Achievement)
8. [Conclusions](#Conclusions)

## Goal

Build a binary classifier model that can predict whether an area is at risk of a wildfire or not.

## Dataset

After a few days of research, we found a curated dataset on [Kaggle](https://www.kaggle.com/). The dataset is called [Wildfire Prediction Dataset (Satellite Images)](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset). This dataset contains satellite images of areas that previously experienced wildfires in Canada.

### Source

Refer to Canada's website for the original wildfires data: [Forest Fires - Open Government Portal](https://open.canada.ca/data/en/dataset/9d8f219c-4df0-4481-926f-8a2a532ca003)

Original license for the data: [Creative Commons 4.0 Attribution (CC-BY) license – Quebec](https://www.donneesquebec.ca/fr/licence/)

### Description

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

### 0- nowildfire

Sample images of nowildfire class:

<p float="left">
  <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/img/kaggle_data/nowildfire/-73.7181%2C45.486459.jpg" width="350" title="nowildfire_img_1" />
  <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/img/kaggle_data/nowildfire/-73.8275%2C45.552381.jpg" width="350" title="nowildfire_img_2" />
</p>

### 1- wildfire

Sample images of wildfire class:

<p float="left">
  <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/img/kaggle_data/wildfire/-60.9878%2C50.4112.jpg" width="350" title="wildfire_img_1" />
  <img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/img/kaggle_data/wildfire/-61.5607%2C50.52878.jpg" width="350" title="wildfire_img_2" />
</p>

## CNN Model Architecture

### Diagram

Neural Network Architecture Diagram:

<img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/resources/wildfire_bin_classifier/cnn_arch/diagram/wildfire_bin_classifier_archDiagram.png" title="wildfire_bin_classifier_arch" />

Code is available [here](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/app/with_curated_dataset/wildfire_bin_classifier/src/model.py)

## Training

### Resources

*   `MPS` device on **Apple MacBook Pro with M1 chip with 32 GB RAM**.
*   `Poetry` for setting up the virtual environment and installing the dependencies.

### Explanation

In the context of **Apple's M1 chip**, `MPS` stands for `Metal Performance Shaders`. **Metal Performance Shaders** is a framework provided by Apple that allows developers to perform advanced computations on the `GPU (Graphics Processing Unit)`.

By specifying `device = torch.device("mps")`, we utilized `MPS` for computations on the GPU.

This led us to:

1. Train the model faster (in just **27 minutes**).
2. Work with around 40,000 images using local resources, thereby avoiding the usage of Google Drive storage, which is slow and problematic.
3. Might be possible that the person who runs this experiment does not have an `Apple M1` computer. Keeping this in mind, we have provided the checkpoint model prediction validation option through a Google Colab notebook.

### Evidence

```python
Train Epoch: 9 [23040/30250 (76%)]	Loss: 0.348124
Train Epoch: 9 [25600/30250 (85%)]	Loss: 0.360356
Train Epoch: 9 [28160/30250 (93%)]	Loss: 0.362320
\Validation set: Average loss: 0.3616, Accuracy: 5836/6300 (93%)

Final Test set: Average loss: 0.3527, Accuracy: 93.65%
Saving model to /Users/julianesteban.borona/Github/upc/projects/aidl-upc-winter2024-satellite-imagery/app/with_curated_dataset/wildfire_bin_classifier/src/checkpoints...
Model saved successfully!
```
*   Training run log is available [here](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/app/with_curated_dataset/wildfire_bin_classifier/src/checkpoints/training_run_log.txt)

### How to run the training

This section would only be possible if you have an `Apple M1` computer. If you don't have an `Apple M1` computer, you can use the Google Colab notebook provided in the [Prediction](#Prediction) section.

1. Clone the repository.
2. Navigate to the `wildfire_bin_classifier` directory.
3. Execute the following `Poetry` commands to set up the virtual environment and install the dependencies:

```bash
poetry config virtualenvs.in-project true
```

Above command configures Poetry to create virtual environments within the project directory itself rather than in a global location. It ensures that each project has its own isolated environment, making it easier to manage dependencies and avoid conflicts between different projects.

```bash
poetry shell
```

Above command activates a virtual environment managed by Poetry, allowing you to work within an isolated environment where project dependencies are installed.

```bash
poetry install
```

Above command installs the dependencies specified in the `pyproject.toml` file using **Poetry**, ensuring that the project has all the necessary packages to run successfully.

That would be all you need to do to set up the virtual environment and install the dependencies. 

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset) and place it in a reachable directory within your local machine.
5. Accommodate dataset paths in the `wildfire_bin_classifier/src/main.py` file.
6. Configure your IDE to use the virtual environment created by Poetry.
7. Run the `wildfire_bin_classifier/src/main.py` file.
8. The model will start training and will save the checkpoint model in the `wildfire_bin_classifier/src/checkpoints` directory.

### Plot of Learning Curves

<img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/app/with_curated_dataset/wildfire_bin_classifier/src/checkpoints/learning_curves.png" title="CNN_learning_curves" />

### Checkpoint

Model checkpoint is available [here](https://drive.google.com/file/d/1NTI68QrPzffmW5Kbgzijs6Roxk_tdhG1/view?usp=sharing)

### Accuracy

**MODEL CHECKPOINT ACCURACY IS: 93.65% 👌**

## Prediction

### Easy execution

For the sake of simplicity while validating this experiment, we have created a straightforward **Google Colab notebook** that can be used to predict the class `{'nowildfire': 0, 'wildfire': 1}` of a given satellite image.

*   Notebook name: `Wildfire_BinClassifier_Notebook_Checkpoint_Predictions.ipynb`
*   Available [here](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/app/with_curated_dataset/wildfire_bin_classifier/src/prediction/Wildfire_BinClassifier_Notebook_Checkpoint_Predictions.ipynb).

#### Further details

Classes

```python
{'nowildfire': 0, 'wildfire': 1}
```

#### Example with class nowildfire

Given image:

<img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/app/with_curated_dataset/wildfire_bin_classifier/src/prediction/test_nw_-73.47513%2C45.58354.jpg" title="inference_class0" />

*   Expected: `nowildfire`
*   Prediction: `nowildfire` ✅

#### Example with class wildfire

Given image:

<img src="https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/blob/main/app/with_curated_dataset/wildfire_bin_classifier/src/prediction/test_w_-62.56176%2C51.29047.jpg" title="inference_class1" />

*   Expected: `wildfire`
*   Prediction: `wildfire` ✅

### Application Code

Available [here](https://github.com/boroju/aidl-upc-winter2024-satellite-imagery/tree/main/app/with_curated_dataset/wildfire_bin_classifier).

## Achievement

At this point, we successfully built a binary classifier model that can predict whether an area is at risk of a wildfire or not. This was accomplished from scratch using a kaggle curated dataset which contains satellite imagery data.

From a given satellite image with similar characteristics to the ones within the [curated dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset), the model can predict the class (0 - 'nowildfire' or 1 - 'wildfire') with an **accuracy of 93.65%**.

## Conclusions

*   Working with satellite imagery data is not easy at all. It requires a lot of effort to build a dataset from scratch.
*   We managed how to use `Poetry` to set up the virtual environment and install the dependencies.
*   We learnt how to take advantage of using `MPS` on an `Apple M1` computer to train a deep learning model faster.
*   At this point we covered a 1st base of the project ensuring that we - at least - have something to show on the final presentation.
*   We can now continue with the next steps of the project more relaxed, knowing that we have a model that can predict whether an area is at risk of a wildfire or not.