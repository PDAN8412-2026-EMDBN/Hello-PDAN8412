# 02 - Basic Neural Networks for Text Classification

## Overview

This activity introduces a basic neural network for text classification. You will load the cleaned 20 Newsgroups dataset prepared in Activity 01, convert text into numeric sequences, train a simple baseline neural network, evaluate its performance, and interpret what the results mean.

This activity is the bridge between Spark-based data preparation and recurrent sequence models. It does not use an RNN or LSTM yet. The goal is to first build a simple neural-network baseline so that later RNN and LSTM models can be compared against something meaningful.

## Why are we doing this?

Before building an RNN or LSTM, you need to understand the normal modelling workflow:

```text
clean text -> numeric representation -> baseline model -> prediction -> evaluation -> interpretation
```

The cleaned dataset from Activity 01 is important because neural networks need consistent input. Very short, empty and extreme-length documents should already have been handled before this activity starts.

## Dataset

Use the cleaned dataset created in Activity 01:

```text
cleaned_20_newsgroups.parquet
```

or:

```text
cleaned_20_newsgroups.csv
```

The dataset should contain:

- `clean_text`
- `target`
- `target_name`
- `word_count`
- `char_length`

If the cleaned dataset is missing, you may reload 20 Newsgroups and recreate a simplified cleaned version. However, the preferred route is to use the file saved from Activity 01.

## Resources to Consult

Use these resources to support your learning. Do not copy code blindly. Any copied or adapted code must be attributed in your notebook/report.

- TensorFlow text classification guide: https://www.tensorflow.org/tutorials/keras/text_classification
- TensorFlow TextVectorization documentation: https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization
- Keras Embedding layer documentation: https://keras.io/api/layers/core_layers/embedding/
- Keras GlobalAveragePooling1D documentation: https://keras.io/api/layers/pooling_layers/global_average_pooling1d/
- TensorFlow text classification with an RNN tutorial: https://www.tensorflow.org/text/tutorials/text_classification_rnn

## Tasks

### 1. Load the Cleaned Dataset

Load the cleaned dataset from Activity 01.

Use one of:

- `cleaned_20_newsgroups.parquet`
- `cleaned_20_newsgroups.csv`

Display:

- the first few rows
- the number of records
- the number of classes
- the available columns
- the distribution of topics

### 2. Confirm the Dataset is Ready for Modelling

Check that the dataset contains usable text for modelling.

Confirm:

- `clean_text` exists
- there are no missing labels
- there are no missing or empty text values
- the shortest documents are no longer too short
- extreme-length documents have been reduced or handled

Answer briefly:

- Why was this cleaning needed before neural-network training?
- What might happen if we trained using empty, very short or extremely long documents?

### 3. Prepare the Features and Labels

Create:

- `X`: the cleaned text values
- `y`: the numeric topic labels

Make sure the labels are suitable for a multi-class neural network.

Then split the data into:

- 80% training data
- 20% testing data

Use a stratified split where possible so that all topics are represented fairly in both sets.

### 4. Create a Baseline for Comparison

Before training a neural network, calculate a simple majority-class baseline.

Answer:

- What accuracy would we get if the model always predicted the largest class?
- Why is this useful before training a neural network?

### 5. Create a Text Vectorisation Layer

Use a `TextVectorization` layer to convert text into integer sequences.

Choose values for:

- vocabulary size
- sequence length
- output mode

The vectoriser must be adapted on the **training text only**.

Explain:

- why text must be converted into numbers
- why the vectoriser should not be adapted on the test set
- what sequence length means

### 6. Build a Basic Neural Network

Build a simple model using:

- `TextVectorization`
- `Embedding`
- `GlobalAveragePooling1D`
- `Dense`
- output layer for multi-class classification

This is not an RNN. It is a baseline neural network.

### 7. Train the Model

Train the model for a small number of epochs.

Track:

- training loss
- validation loss
- training accuracy
- validation accuracy

### 8. Evaluate the Model

Evaluate the model using:

- test accuracy
- test loss
- loss curve
- accuracy curve
- classification report
- confusion matrix

### 9. Make Predictions

Select a few test documents and display:

- the text preview
- the true topic
- the predicted topic
- whether the prediction was correct

### 10. Interpret the Results

Answer:

- Did the model perform better than the majority-class baseline?
- Which classes were easier to predict?
- Which classes were confused with one another?
- Is there evidence of overfitting?
- What are the limitations of this basic model?
- Why will the next activity use an RNN?

## Discussion Questions

- What is the purpose of an embedding layer?
- Why can a neural network not use raw text directly?
- What is the difference between fitting the vectoriser and fitting the model?
- Why should the vectoriser be adapted only on training data?
- Why is a baseline neural network useful before trying an RNN or LSTM?
- What type of patterns might an RNN capture better than this model?
