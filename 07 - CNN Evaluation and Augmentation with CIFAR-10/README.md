# 07 - CNN Evaluation and Augmentation with CIFAR-10

## Overview

This activity continues the CNN work from Activity 06, but now uses a different kind of image data. Activity 06 used Fashion-MNIST, which contains small grayscale clothing images. This activity uses CIFAR-10, which contains small colour images from ten object classes.

The purpose is not only to get a working model. The purpose is to understand why colour image classification is more difficult, how image shape changes from grayscale to RGB, how overfitting appears in CNNs, and how basic augmentation can help a model generalise.

This gives the CNN topic a proper second evening without turning Activity 07 into a disconnected theory session.

## Why are we doing this?

Fashion-MNIST is useful for a first CNN because the images are small and simple. Real image-classification problems are usually messier. Colour images have three channels, objects vary more, backgrounds differ, and classes may be visually similar.

By the end of this activity, you should be able to explain:

- the difference between grayscale and RGB image tensors
- why CNNs are suitable for image classification
- why a dense model is a weak baseline for images
- how to evaluate a CNN using accuracy, loss curves and a confusion matrix
- how to identify overfitting
- how augmentation changes the training data
- why model performance must be interpreted in relation to the dataset

## Dataset

The solution notebook uses the CIFAR-10 dataset from Keras.

CIFAR-10 contains 60,000 colour images across 10 classes. Each image is 32 × 32 pixels with 3 colour channels.

The classes are:

```text
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

## Tasks

### 1. Prepare the Environment

Import the required libraries:

- NumPy
- Pandas
- Matplotlib
- TensorFlow / Keras
- scikit-learn metrics
- Spark for EDA where possible

Set a random seed.

### 2. Load CIFAR-10

Load the CIFAR-10 dataset from Keras.

Inspect:

- training shape
- test shape
- number of classes
- image height and width
- number of colour channels

### 3. Visualise the Images

Display a grid of sample images with their class names.

Answer:

- Are these images easier or harder than Fashion-MNIST?
- Which classes look visually similar?
- What makes this dataset more realistic?

### 4. Use Spark for Dataset Summary

Create a Spark DataFrame containing image metadata, not the raw images.

Include:

- image index
- class label
- class name
- split name
- mean pixel value
- standard deviation of pixel values

Use Spark to:

- count records
- check class balance
- summarise pixel statistics
- compare train and test class distributions

### 5. Prepare the Image Data

Normalise pixel values to the range 0 to 1.

Confirm the shape:

```text
number of images, height, width, channels
```

Explain why CNNs need the image shape to be preserved.

### 6. Build a Dense Baseline

Build a simple dense neural network by flattening the images.

This baseline is intentionally limited because flattening removes spatial structure.

Evaluate the baseline model.

### 7. Build a CNN Model

Build a CNN using:

- Conv2D
- MaxPooling2D
- Flatten or GlobalAveragePooling2D
- Dense layers
- Dropout
- softmax output

Train the model and evaluate it.

### 8. Compare Baseline and CNN

Compare the dense baseline and CNN on the same CIFAR-10 test set.

Discuss:

- Which model performed better?
- Why is the comparison fair here?
- What does the CNN preserve that the dense model loses?

### 9. Analyse Overfitting

Plot training and validation accuracy/loss.

Answer:

- Did training accuracy improve faster than validation accuracy?
- Did validation loss increase while training loss decreased?
- Is the model overfitting, underfitting or learning reasonably?

### 10. Confusion Matrix and Classification Report

Create a confusion matrix and classification report.

Identify:

- strongest classes
- weakest classes
- commonly confused pairs

### 11. Add Basic Augmentation

Train a CNN using basic augmentation such as:

- random horizontal flip
- random rotation
- random zoom

Compare the augmented model with the previous CNN.

### 12. Save Results

Save a results file named:

```text
activity07_cifar10_cnn_results.csv
```

Include:

- model name
- test accuracy
- test loss
- notes about overfitting or generalisation

## Discussion Questions

- Why is CIFAR-10 harder than Fashion-MNIST?
- What does the channel dimension represent?
- Why is flattening a weak approach for image classification?
- Why are CNNs suitable for image data?
- What does augmentation try to achieve?
- Why should models be compared only when they use the same dataset and task?
