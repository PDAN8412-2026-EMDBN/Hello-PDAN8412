# 04 - LSTM for Text Classification

## Overview

This activity introduces Long Short-Term Memory (LSTM) networks for text classification. You will load the cleaned text dataset, prepare text sequences, train an LSTM model, evaluate it, compare it to the SimpleRNN model from Activity 03, and write an interpretation.

This activity still uses the 20 Newsgroups dataset rather than an author-labelled assessment dataset.

## Why are we doing this?

For us to classify text, we need text preparation, sequence modelling, model evaluation, and interpretation. LSTMs are useful because they are designed to handle longer-term sequence patterns better than a basic RNN.

The transferable workflow is:

```text
labelled text -> cleaning -> sequence preparation -> LSTM -> evaluation -> retraining/improvement -> report
```

## Resources to Consult

Use these resources to support your learning. Do not copy code blindly. Any copied or adapted code must be attributed in your notebook/report.

- TensorFlow text classification with an RNN: https://www.tensorflow.org/text/tutorials/text_classification_rnn
- Keras LSTM documentation: https://keras.io/api/layers/recurrent_layers/lstm/
- Keras Bidirectional wrapper: https://keras.io/api/layers/recurrent_layers/bidirectional/
- Keras TextVectorization layer: https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/
- TensorFlow model training guide: https://www.tensorflow.org/guide/keras/training_with_built_in_methods
- scikit-learn classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- scikit-learn confusion matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
- Google Machine Learning Crash Course: https://developers.google.com/machine-learning/crash-course/

## Tasks

### 1. Load the Cleaned Dataset

Load the cleaned dataset from Activity 01.

Confirm that the dataset contains:

- `clean_text`
- `target`
- `target_name`
- `word_count`
- `char_length`

Display:

- number of records
- number of classes
- class distribution
- sample text records

### 2. Short explanations

Write a short Markdown explanation answering:

- What is the classification label in this activity?
- What parts of this workflow are transferable to author-style classification?

### 3. Prepare Train and Test Data

Create:

- `X_train`
- `X_test`
- `y_train`
- `y_test`

Use an 80/20 split and stratify the labels where possible.

### 4. Create the Text Vectorisation Layer

Create a `TextVectorization` layer.

Suggested values:

```text
max_tokens = 20000
sequence_length = 300
```

Fit the vectorisation layer on the training text only.

### 5. Build an LSTM Model

Build an LSTM model with:

- string input layer
- text vectorisation layer
- embedding layer
- LSTM layer
- dropout layer
- dense output layer with softmax activation

The model should predict the document topic.

### 6. Train the LSTM Model

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

### 7. Evaluate the LSTM Model

Evaluate the model on the test set.

Produce:

- test accuracy
- classification report
- confusion matrix
- prediction examples

### 8. Improve or Retrain the Model

Treat the first LSTM as a first attempt, not the final answer.

Make a justified improvement and retrain the model.

Recommended improvement:

- reduce the sequence length using evidence from Activity 01
- increase the embedding size
- use a Bidirectional LSTM
- add dropout
- use class weights
- add early stopping
- allow more epochs, but stop automatically if validation loss stops improving

Explain what changed and whether it helped.

A massive improvement is **not guaranteed**. If the Activity 02 baseline still performs better, explain why a simpler model may work well for topic classification.

### 9. Optional Extension

If the improved LSTM is still weak and there is enough class time, run one additional controlled experiment.

Change only one or two parameters, such as:

- sequence length
- embedding size
- LSTM units
- number of epochs
- dropout rate

Do not change many things at once. Each change must be justified.

### 10. Compare the Basic Neural Network, RNN and LSTM

Create a short comparison table.

Compare:

- model type
- training time
- complexity
- accuracy
- strengths
- weaknesses

### 11. Save the Model or Results

Save at least one of the following:

- trained LSTM model
- improved model
- classification report
- comparison table
- predictions sample

## Discussion Questions

- Why can LSTMs work better than basic RNNs on longer sequences?
- Did the LSTM perform better than the SimpleRNN?
- Did the LSTM perform better than the Activity 02 baseline?
- If it did not perform better, what might explain that?
- Why might topic classification not show the full benefit of an LSTM?
- Why should we include more than just accuracy when evaluating?
- What metrics would be useful for an author-classification task?
- What would count as evidence that the model is learning writing style rather than accidental clues?
- What kinds of data leakage could affect an author-classification dataset?
