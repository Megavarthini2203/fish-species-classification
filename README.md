# Fish species Classification
## Description
This notebook contains data analysis and visualization steps for image classification using various models, including CNN and VGG16, and techniques like PCA and Grad-CAM for model evaluation and visualization.

## Libraries Used
from sklearn.metrics import classification_report, confusion_matrix
import glob as gb
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2, VGG16
import pandas as pd
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import cv2
import torch.nn as nn
import warnings
import pytorch_lightning as pl
import torch
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.image as mpimg
from pytorch_lightning.loggers import CSVLogger
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, BatchNormalization, Dense
from albumentations.pytorch import ToTensorV2
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing import image
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import seaborn as sns
import matplotlib.cm as cm

## Step
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
