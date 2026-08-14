# 06 - CNN Foundations and Evaluation with Fashion-MNIST

## Overview

This activity introduces Convolutional Neural Networks (CNNs) using image-classification data. In the previous activities, you worked mainly with text data, tokenisation, embeddings, RNNs and LSTMs. In this activity, you will move from text to images and examine why image data requires a different model structure.

You will load Fashion-MNIST, inspect images as arrays, visualise samples, train a basic dense neural-network baseline, train a CNN, evaluate performance, inspect confusion matrices, and reflect on overfitting, dropout and augmentation.

This activity is designed to fit into one evening. It does not cover transfer learning or large custom image datasets. Those topics are handled later in the module.

## Why are we doing this?

A text model usually works with sequences of words or tokens. An image model works with spatial structure. A shirt, sneaker or bag is not just a list of pixel values. The position of pixels and the relationship between nearby pixels matter.

CNNs are useful because they can learn local visual patterns such as edges, curves, textures and shapes. They do this using convolutional filters and pooling operations.

## Dataset

Use the Fashion-MNIST dataset available through TensorFlow/Keras.

Fashion-MNIST contains grayscale images of clothing items. Each image is 28 × 28 pixels and belongs to one of 10 classes.

The classes are:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

The dataset can be loaded using:

```python
from tensorflow.keras.datasets import fashion_mnist
```

## Resources to Consult

Use these resources to support your learning:
- TensorFlow Fashion-MNIST classification tutorial: https://www.tensorflow.org/tutorials/keras/classification
- Keras Conv2D layer documentation: https://keras.io/api/layers/convolution_layers/convolution2d/
- Keras MaxPooling2D layer documentation: https://keras.io/api/layers/pooling_layers/max_pooling2d/
- Keras image augmentation layers: https://keras.io/api/layers/preprocessing_layers/image_augmentation/
- TensorFlow model evaluation guide: https://www.tensorflow.org/tutorials/keras/classification

## Tasks

### 1. Prepare the Environment

- Import TensorFlow/Keras.
- Import NumPy, Pandas and Matplotlib.
- Import scikit-learn evaluation tools.
- Set a random seed where appropriate.

### 2. Load Fashion-MNIST

Load the dataset using TensorFlow/Keras.

You should:

- load the training and testing data
- display the shape of the arrays
- confirm the number of classes
- display the class names
- inspect the pixel-value range

### 3. Dataset Suitability Check

Create a Markdown section answering:

- What is the source of the dataset?
- What does each row/image represent?
- What is the input data?
- What is the target label?
- Why is this dataset suitable for CNN foundations?
- What are its limitations?

### 4. Image Data as Arrays

Inspect a single image.

You should:

- display the image shape
- print a small section of the pixel matrix
- explain what a grayscale pixel value represents
- explain why images need a channel dimension for CNNs

### 5. Visualise Sample Images

Create a grid of sample images with labels.

Answer:

- Which classes look visually distinct?
- Which classes may be confused?
- Why might shirts, coats and pullovers be difficult for a model?

### 6. Prepare the Data

Prepare the data for modelling.

You should:

- scale pixel values from 0–255 to 0–1
- create a flattened version for the dense baseline
- create a CNN version with shape `(28, 28, 1)`
- split part of the training data into validation data

### 7. Train a Dense Neural-Network Baseline

Train a simple dense model that does not use convolution.

The model should include:

- `Flatten`
- at least one `Dense` hidden layer
- `Dropout`
- a softmax output layer

Evaluate the model and record the test accuracy.

### 8. Train a Basic CNN

Train a CNN model using:

- `Conv2D`
- `MaxPooling2D`
- `Flatten`
- `Dense`
- `Dropout`
- softmax output

Evaluate the model and record the test accuracy.

### 9. Compare the Dense Baseline and CNN

Compare the two models fairly.

Discuss:

- Which model performed better?
- Why might the CNN be more suitable for images?
- Did the CNN overfit?
- What evidence do the loss and accuracy curves show?

### 10. Evaluate the CNN

Use the CNN predictions to create:

- classification report
- confusion matrix
- sample correct predictions
- sample incorrect predictions

Answer:

- Which classes were easiest to classify?
- Which classes were most often confused?
- What visual similarities may explain the errors?

### 11. Augmentation and Overfitting Discussion

Add a short discussion about image augmentation.

You do not need to build a full augmentation pipeline in this activity, but you should explain:

- what augmentation is
- why augmentation can reduce overfitting
- which augmentations make sense for Fashion-MNIST
- which augmentations may be inappropriate

Optional: train a small CNN with Keras augmentation layers and compare the result.

### 12. Save Results

Save a small CSV file containing the dense baseline and CNN results.

Suggested columns:

- `activity`
- `dataset`
- `data_type`
- `task`
- `model`
- `test_accuracy`
- `main_strength`
- `main_limitation`

Save as:

```text
activity06_cnn_results.csv
```

## Discussion Questions

- Why do CNNs preserve image structure better than dense networks?
- What does a convolutional filter learn?
- Why is pooling useful?
- Why might a CNN confuse shirts, coats and pullovers?
- What is overfitting?
- How can augmentation help with overfitting?
- What would make a dataset suitable for the final image-recognition PoE?
