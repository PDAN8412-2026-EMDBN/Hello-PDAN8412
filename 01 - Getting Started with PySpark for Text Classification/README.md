# 01 - Getting Started with PySpark for Text Classification

## Overview

This activity introduces Apache Spark using Python. You will install and configure Spark, load a labelled text dataset, perform exploratory data analysis, use Spark SQL, apply DataFrame transformations, clean text data, investigate text-length outliers, and prepare a dataset that could later be used for an RNN or LSTM classification model.

This activity does not train a neural network yet. The goal is to understand and prepare the data first.

## Why are we doing this?

As a data scientist, you will often work with text data and build an RNN/LSTM-style model that can classify writing. This activity uses topic classification so that you can apply the same workflow for other similar tasks.

The cleaned dataset produced in this activity will be used in later activities, where you will build a basic neural network, an RNN, and then an LSTM.

## Dataset

Use the 20 Newsgroups dataset.

This dataset contains text documents labelled by topic/category. The dataset can be loaded using scikit-learn:

```python
from sklearn.datasets import fetch_20newsgroups
```

Use the dataset with headers, footers and quoted replies removed where possible.

## Resources to Consult

Use these resources to support your learning. Do not copy code blindly. Any copied or adapted code must be attributed in your notebook/report.

- Apache Spark PySpark DataFrame documentation: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html
- Spark SQL documentation: https://spark.apache.org/docs/latest/sql-programming-guide.html
- PySpark functions documentation: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html
- scikit-learn 20 Newsgroups dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
- TensorFlow text classification with an RNN tutorial: https://www.tensorflow.org/text/tutorials/text_classification_rnn
- Keras SimpleRNN layer documentation: https://keras.io/api/layers/recurrent_layers/simple_rnn/
- Keras LSTM layer documentation: https://keras.io/api/layers/recurrent_layers/lstm/
- Google Machine Learning Crash Course: https://developers.google.com/machine-learning/crash-course/
- Towards Data Science multi-class text classification with LSTM: https://medium.com/towards-data-science/multi-class-text-classification-with-lstm-1590bee1bd17
- Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

## Tasks

### 1. Install Spark and Set Up the Environment

- Install PySpark if it is not already available.
- Start a Spark session.
- Confirm that Spark is running.
- Use Jupyter Notebook or Google Colab where appropriate.

### 2. Load the 20 Newsgroups Dataset

Load the dataset using `fetch_20newsgroups`.

You should:

- remove headers, footers and quoted replies where possible
- create a Pandas DataFrame containing the text and label
- convert the Pandas DataFrame into a Spark DataFrame
- display the schema
- display the first few rows
- count the number of records

Your DataFrame should contain:

- `text`
- `target`
- `target_name`

### 3. Dataset Suitability Check

Create a short Markdown section answering:

- What is the source of the dataset?
- What does each row represent?
- What is the input column?
- What is the target label?
- How many records are available?
- Why is this dataset suitable for text classification?
- Why is this dataset useful preparation for RNN/LSTM work?
- What are the limitations of using this dataset?

### 4. Basic EDA

Use Spark to explore the dataset.

Complete the following:

- Count the number of records.
- Count the number of columns.
- Display column names and data types.
- Check for missing values.
- Check for duplicate rows.
- Count how many documents exist per topic.

### 5. Add Text-Length Features

Create the following columns:

- `char_length`: number of characters in the text
- `word_count`: number of words in the text

Use these columns to answer:

- What is the average document length?
- Which topics have the longest average documents?
- Which topics have the shortest average documents?
- Are there documents that are too short to be useful?

### 6. Spark SQL Analysis

Convert the DataFrame into a temporary SQL table called `newsgroups`.

Use Spark SQL to answer:

- How many documents are in each topic?
- What is the average word count per topic?
- What are the top 5 topics by document count?
- What are the bottom 5 topics by document count?
- Which topics may be harder for a model to learn because they have fewer examples?

### 7. Method Chaining / Dot Notation

Repeat selected SQL tasks using DataFrame transformations and method chaining.

Use operations such as:

- `select`
- `filter`
- `groupBy`
- `agg`
- `orderBy`
- `withColumn`

### 8. Clean the Text Data

Create a new column called `clean_text`.

Apply the following cleaning steps:

- Convert text to lowercase.
- Remove URLs.
- Remove email addresses.
- Remove punctuation and special characters.
- Remove extra spaces.
- Remove rows with missing or empty text.
- Remove rows where the text has fewer than 20 words.

After cleaning, create updated length columns:

- `clean_char_length`: number of characters in the cleaned text
- `clean_word_count`: number of words in the cleaned text

Display examples comparing:

- original `text`
- cleaned `clean_text`
- original `word_count`
- cleaned `clean_word_count`

### 9. Investigate Very Short and Very Long Documents

The first visualisations may show that the word-count distribution is skewed. Some documents may be extremely short, while others may be very long.

Use Spark to:

- find the minimum, maximum, mean and standard deviation of `clean_word_count`
- calculate the 95th percentile of `clean_word_count`
- calculate the 99th percentile of `clean_word_count`
- display the longest documents after cleaning

Then decide on a sensible modelling range.

For this activity, create a final modelling dataset by keeping documents that meet these rules:

```text
clean_word_count >= 20
clean_word_count <= 99th percentile word count
```

This removes very short documents and extreme outliers while keeping most of the dataset.

Write a short Markdown explanation answering:

- Why are very short documents a problem for text classification?
- Why are extremely long documents a problem for RNN/LSTM training?
- Why might we cap documents at the 99th percentile instead of using the maximum length?
- What information could be lost when long documents are truncated later?

### 10. Class Balance Analysis

Analyse the topic distribution after cleaning and outlier handling.

Calculate:

- total records per topic
- percentage of the full cleaned dataset per topic
- largest topic
- smallest topic

Then answer:

- Is the dataset balanced?
- Could imbalance affect a future classification model?
- What could be done if some classes are too small?

### 11. Train/Test Preparation

Create a final DataFrame containing:

- `clean_text`
- `target`
- `target_name`
- `clean_word_count`
- `clean_char_length`

For compatibility with later activities, also rename:

- `clean_word_count` to `word_count`
- `clean_char_length` to `char_length`

Split the data into:

- 80% training data
- 20% testing data

Display the record count for each split.

### 12. Visualisation

Convert only aggregated Spark results to Pandas.

Create:

- bar chart of document count per topic after cleaning
- bar chart of average cleaned word count per topic
- histogram of cleaned document word counts after outlier handling
- optional: top 20 most frequent words after cleaning

Do not convert the full Spark DataFrame to Pandas unless you have sampled or aggregated it first.

### 13. Save the Cleaned Dataset

Save the cleaned dataset for use in later activities.

Save as:

```text
cleaned_20_newsgroups.parquet
```

or:

```text
cleaned_20_newsgroups.csv
```

The saved dataset should include:

- `clean_text`
- `target`
- `target_name`
- `word_count`
- `char_length`

These column names are important because later activities will load this file directly.

## Discussion Questions

- Why is Spark useful when working with large text datasets?
- Why should we avoid converting a full Spark DataFrame to Pandas?
- How is topic classification similar to author-style classification?
- How is topic classification different from author-style classification?
- What makes a text dataset suitable for an RNN or LSTM?
- Why is class balance important?
- What is data leakage?
- Why should headers, footers and quoted replies be removed from this dataset?
- Why do very long documents matter when training neural networks?
- What is the difference between removing an outlier and truncating a sequence later?
- What extra work would be needed before training an RNN/LSTM model?
