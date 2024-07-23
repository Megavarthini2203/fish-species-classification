# Fish species Classification
## Description
This repository contains a Jupyter Notebook (fdsProject.ipynb) for classifying images into nine different species using a convolutional neural network (CNN) and VGG16 pre_trained model with feature extraction PCA layer.

## Table of Contents
- Introduction(#introduction)
- Dataset(#dataset)
- Steps involved(#steps-involved)
- Model Architecture(#model-architecture)
- Training(#training)
- Evaluation(#evaluation)
- Results(#results)
- Usage(#usage)
- Dependencies(#dependencies)
- Acknowledgements

## Introduction
This project aims to classify images into one of nine species using a CNN. The notebook includes data preprocessing, model training, evaluation, and visualization of results.

## Dataset
The dataset used in this project consists of images belonging to the following nine species:

- Black Sea Sprat 
- Gilt-Head Bream
- Hourse Mackerel
- Red Mullet 
- Red Sea Bream 
- Sea Bass
- Shrimp
- Striped Red Mullet 
- Trout

## Step Involved
- Importing necessary modules and libraries
- Exploratory analysis
- Classification using CNN model
- Generating images for training and testing
- VGG16 Model
- Fitting the model
- Model evaluation
- Predicting sample image
- VGG16 Model with feature extraction PCA layer
- History of the model in each epoch
- Visualizing the performance of the model
- Evaluating model with testing images
- Classification report on trained model
- Confusion Matrix
- Dataframe to store predicted images
- Function to display VGG16 model predicted images
- Displaying correctly predicted image of VGG16 model
- Gradient-weighted Class Activation Mapping
- Function to display images with Grad-CAM heatmap
- Visualizing images with Grad-CAM heatmap
  
## Model Architecture
The model architecture is based on a CNN, designed to handle image data efficiently. The architecture includes several convolutional layers, max-pooling layers, and fully connected layers. Below is a summary of the model architecture:
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 222, 222, 32)      896       
                                                                 
 max_pooling2d (MaxPooling2D) (None, 111, 111, 32)     0         
                                                                 
 flatten (Flatten)           (None, 394272)            0         
                                                                 
 dense (Dense)               (None, 128)               50466944  
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 64)                8256      
                                                                 
 dense_2 (Dense)             (None, 9)                 585       
                                                                 
=================================================================
Total params: 50476681 (192.55 MB)
Trainable params: 50476681 (192.55 MB)
Non-trainable params: 0 (0.00 Byte)
_______________________________________________________

## Training
The model is trained on the dataset using a specified number of epochs. The notebook includes code for splitting the dataset into training and validation sets, data augmentation, and the training loop.

## Evaluation
The model's performance is evaluated using metrics such as accuracy and loss on the validation set. Confusion matrices and classification reports are generated to provide detailed insights into the model's performance.

## Results
The results section includes visualizations of the training and validation accuracy and loss over epochs. It also includes sample predictions and their corresponding ground truth labels.

## Usage
To use the notebook:

1. Clone this repository :
git clone https://github.com/yourusername/fdsProject.git

2. Navigate to the project directory:
cd fdsProject

3. Install the required dependencies:
pip install -r requirements.txt

4.Run the Jupyter Notebook:
jupyter notebook fdsProject.ipynb

## Dependencies
The project requires the following dependencies:

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

 Install the dependencies using the provided requirements.txt file.

## Acknowledgements
This project was developed as part of a coursework for the FDS class. Special thanks to the instructors and peers for their support and feedback

