# 03 - Recurrent Neural Networks for Text Classification

## Overview

This activity introduces Recurrent Neural Networks (RNNs) for text classification. You will load the cleaned 20 Newsgroups dataset prepared in Activity 01, reuse the text-preparation workflow from Activity 02, train a SimpleRNN model, evaluate the results, and explain why RNNs are useful for sequence data.

This activity comes after the basic neural network baseline. The goal is not simply to get a higher accuracy score. The goal is to understand how text can be treated as a sequence and why recurrent models were designed for this type of data.

## Why are we doing this?

As a data scientist, you need to know how to work with text data and build an RNN/LSTM-style model that can classify text. In this activity, the label is still the newsgroup topic.

The transferable workflow is:

```text
clean text -> label -> tokenisation -> sequence preparation -> RNN model -> evaluation -> interpretation
```

## Resources to Consult

Use these resources to support your learning. Do not copy code blindly. Any copied or adapted code must be attributed in your notebook/report.

- TensorFlow text classification with an RNN: https://www.tensorflow.org/text/tutorials/text_classification_rnn
- Keras SimpleRNN documentation: https://keras.io/api/layers/recurrent_layers/simple_rnn/
- Keras Embedding layer documentation: https://keras.io/api/layers/core_layers/embedding/
- Keras TextVectorization layer documentation: https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/

## Tasks

### 1. Load the Cleaned Dataset

Load the cleaned dataset saved from Activity 01.

Use either:

```text
cleaned_20_newsgroups.parquet
```

or:

```text
cleaned_20_newsgroups.csv
```

Your dataset should contain:

- `clean_text`
- `target`
- `target_name`
- `word_count`
- `char_length`

Display:

- the first few rows
- the number of records
- the number of classes
- the class names

### 2. Confirm Dataset Readiness

Before training the RNN, check that:

- there are no missing values in `clean_text`
- there are no empty text values
- the target labels are available
- each class has enough records
- text lengths are not extreme after cleaning

Write a short Markdown explanation of whether the dataset is ready for modelling.

### 3. Prepare Train and Test Data

Create:

- `X_train`
- `X_test`
- `y_train`
- `y_test`

Use an 80/20 train/test split.

Use stratification where possible so that all classes are represented properly.

### 4. Create the Text Vectorisation Layer

Use a `TextVectorization` layer to convert text into integer sequences.

Suggested values:

```text
max_tokens = 20000
sequence_length = 300
```

Important:

- Fit the vectorisation layer on the training text only.
- Do not fit it on the testing data.
- Explain why this matters.

### 5. Build a SimpleRNN Model

Build a model with:

- string input layer
- text vectorisation layer
- embedding layer
- SimpleRNN layer
- dropout layer
- dense output layer with softmax activation

The model should predict the topic/category of a text document.

### 6. Train the RNN Model

Train the model using:

- training text
- training labels
- validation split
- a small number of epochs to start

Record:

- training accuracy
- validation accuracy
- training loss
- validation loss

### 7. Evaluate the RNN Model

Evaluate the model on the test set.

You should produce:

- test accuracy
- classification report
- confusion matrix
- prediction examples

### 8. Plot the Training History

Create plots showing:

- training vs validation accuracy
- training vs validation loss

Use these plots to comment on:

- underfitting
- overfitting
- whether more epochs would help
- whether the model is learning useful patterns

### 9. Compare with the Basic Neural Network

Compare the RNN model to the Activity 02 baseline.

Answer:

- Did the RNN perform better?
- Did it train slower?
- Was it more complex?
- Is better accuracy guaranteed when using an RNN?
- Why might a simple baseline sometimes perform very well on text-classification tasks?

### 10. Reflect on RNN Limitations

Write a short explanation of the main limitations of a basic RNN.

Include:

- vanishing gradients
- difficulty remembering long-range patterns
- training time
- sensitivity to sequence length

### 11. Save the Model or Results

Save at least one of the following:

- trained RNN model
- training history
- classification report
- predictions sample

This will help you compare the RNN with the LSTM in Activity 04.


### 12. Improve the SimpleRNN

After evaluating the first SimpleRNN model, create an improved SimpleRNN experiment.

You may adjust:

- sequence length
- vocabulary size
- embedding dimension
- dropout
- number of epochs
- learning rate
- early stopping
- class weights
- bidirectional SimpleRNN

Important:

- Do not use LSTM or GRU in this activity.
- Keep this as a SimpleRNN-based model.
- Compare the improved SimpleRNN against the first SimpleRNN.
- Then compare it against the Activity 02 baseline neural network.

Write a short interpretation explaining whether the changes improved the model and whether they solved the main weaknesses of the original SimpleRNN.

## Discussion Questions

- Why does an RNN process text differently from a basic neural network?
- Why is word order important in text classification?
- What does the sequence length control?
- What happens if the sequence length is too short?
- What happens if the sequence length is too long?
- Why can basic RNNs struggle with long documents?
- Why do we need an LSTM after learning about RNNs?
