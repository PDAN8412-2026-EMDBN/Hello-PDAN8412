# 05 - Authorship Classification with Blog Data

## Overview

This activity extends the text-classification workflow from Activities 01 to 04. In the earlier activities, you classified documents by topic using the 20 Newsgroups dataset. In this activity, you will classify documents by author using a blog authorship dataset. A model may need to learn patterns in writing style, phrasing, structure, vocabulary preference and repeated language habits.

You will load a blog dataset, create a manageable author-classification subset, perform Spark-based exploratory analysis, clean the text, prepare sequences, train multiple models, and compare whether recurrent models are more useful when the target is the writer rather than the topic.

## Why are we doing this?

The 20 Newsgroups dataset is useful for learning the pipeline, but it is mostly a topic classification problem. Many classes can be predicted from obvious keywords.
Authorship classification is different. The target label is the writer. This means the model should not only learn what the text is about, but also how the writer writes.

## Dataset

Use the Kaggle Blog Authorship Corpus:

```text
https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus
```

The solution notebook attempts to download this dataset directly using `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("rtatman/blog-authorship-corpus")
```

If the Kaggle download does not work in your environment, manually download the dataset and place `blogtext.csv` in the same folder as the notebook.

Typical columns include:

- `id` or `author_id`
- `gender`
- `age`
- `topic`
- `sign`
- `date`
- `text`

The exact column names may differ depending on the dataset version.

For this activity, the main target label is:

```text
author_id
```

Do **not** use gender, age, zodiac sign or topic as the main target for this activity. Those fields may be inspected during EDA, but the modelling task should be author classification.

## Resources to Consult

Use these resources to support your learning. Do not copy code blindly. Any copied or adapted code must be attributed in your notebook/report.

- Blog Authorship Corpus dataset page: https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus
- Sample Blog Corpus dataset page: https://www.kaggle.com/datasets/saurabhbagchi/sample-blog-corpus
- Apache Spark PySpark DataFrame documentation: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html
- TensorFlow TextVectorization documentation: https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/
- Keras SimpleRNN layer documentation: https://keras.io/api/layers/recurrent_layers/simple_rnn/
- Keras LSTM layer documentation: https://keras.io/api/layers/recurrent_layers/lstm/
- TensorFlow text classification with RNN tutorial: https://www.tensorflow.org/text/tutorials/text_classification_rnn
- Google Machine Learning Crash Course: https://developers.google.com/machine-learning/crash-course/


### Large dataset note

The real `blogtext.csv` file is large. The solution notebook reads it directly with Spark using `spark.read.csv(...)`. Do not load the full file into Pandas and then convert it using `spark.createDataFrame(...)`, because this can cause a Java heap-space error in Colab or on machines with limited memory.

The notebook also reduces the dataset **before** running expensive text-cleaning operations. The full dataset is used for basic Spark EDA, but the heavier cleaning is applied only after a manageable author-balanced subset has been selected.

## Tasks

### 1. Prepare the Environment

- Import the required libraries.
- Start a Spark session.
- Confirm that Spark is running.
- Import TensorFlow/Keras, scikit-learn, Pandas and Matplotlib.

### 2. Load the Blog Dataset

Load the Blog Authorship Corpus from Kaggle.

Use the dataset page:

```text
https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus
```

Your notebook should first try to pull the dataset using KaggleHub:

```python
import kagglehub
path = kagglehub.dataset_download("rtatman/blog-authorship-corpus")
```

Then locate the CSV file, usually called:

```text
blogtext.csv
```

If KaggleHub is unavailable, download the dataset manually and place `blogtext.csv` in the same folder as the notebook.

You should:

- load or download the dataset
- confirm that the notebook loaded the real Kaggle dataset
- display the first few rows
- display the column names
- identify the text column
- identify the author column
- create a Spark DataFrame
- display the schema
- count the number of records

Your working DataFrame should contain at least:

- `author_id`
- `text`

### 3. Dataset Suitability Check

Create a short Markdown section answering:

- What is the source of the dataset?
- Did the notebook load the real Kaggle dataset successfully?
- What does each row represent?
- What is the input column?
- What is the target label?
- How many records are available before filtering?
- Why is this dataset suitable for authorship classification?
- Why is this dataset useful preparation for RNN/LSTM work?
- What are the limitations of using blog data?

### 4. Ethical and Practical Considerations

Write a short note covering:

- why personal blog data should be handled carefully
- why demographic labels should not be used carelessly
- why old datasets may not represent current writing habits
- why authorship models can raise privacy concerns
- how you will avoid exposing personal details in your analysis

### 5. Basic Spark EDA

Use Spark to explore the dataset.

Complete the following:

- Count the number of records.
- Count the number of columns.
- Display column names and data types.
- Check for missing values in `author_id` and `text`.
- Check for duplicate rows.
- Count how many posts each author has.
- Identify the authors with the most posts.
- Identify authors with too few posts for modelling.

### 6. Add Text-Length Features

Create the following columns:

- `char_length`: number of characters in the original text
- `word_count`: number of words in the original text

Use these columns to answer:

- What is the average post length?
- Which authors have the longest average posts?
- Which authors have the shortest average posts?
- Are there posts that are too short to be useful?
- Are there extreme long-text outliers?

### 7. Create a Manageable Authorship Subset Before Cleaning

The full blog dataset may contain too many authors and too many rows to clean unnecessarily. Do not train on all authors, and do not run heavy regex cleaning across the full dataset first.

Use Spark to create a manageable subset before cleaning:

- keep only the columns needed for this task
- apply light filtering only, such as removing null text and very short raw posts
- identify authors with enough posts
- select 5 to 10 authors
- sample the same number of raw posts per author
- keep this raw subset balanced before moving to full text cleaning

Suggested starting point:

```text
n_authors = 8
raw_posts_per_author_cap = 250
minimum_raw_words = 30
```

If your machine is slow, reduce the number of authors or the number of posts per author.

### 8. Clean the Smaller Balanced Subset

Create a new column called `clean_text` using the manageable subset from Task 7.

Apply the following cleaning steps:

- Convert text to lowercase.
- Remove HTML tags.
- Remove URLs.
- Remove email addresses.
- Remove punctuation and special characters.
- Remove extra spaces.
- Remove rows with missing or empty text.
- Remove rows where the cleaned text has fewer than 30 words.
- Inspect the 95th and 99th percentile of cleaned word counts.
- Remove extreme long-text outliers using a reasonable threshold.
- Rebalance the selected authors after cleaning, because cleaning may remove different numbers of posts per author.

### 9. Class Balance Analysis

Analyse the author distribution after subsetting.

Calculate:

- total records per author
- percentage of the subset per author
- largest class
- smallest class

Then answer:

- Is the subset balanced?
- Why is balance important for authorship classification?
- What could happen if one author has many more posts than the others?

### 10. Train/Test Preparation

Create a final modelling DataFrame containing:

- `clean_text`
- `author_id`
- `author_label`
- `word_count`
- `char_length`

Split the data into:

- 80% training data
- 20% testing data

Use a stratified split where possible.

Display the record count for each split and check class distribution in both sets.

### 11. Text Vectorisation

Use Keras `TextVectorization`.

You should:

- adapt the vectoriser only on the training text
- choose a vocabulary size
- choose a sequence length
- explain why the sequence length matters
- avoid fitting text-processing steps on the test set

Suggested starting values:

```text
max_tokens = 20000
sequence_length = 150
```

### 12. Train a Majority-Class Baseline

Before training neural networks, calculate a simple majority-class baseline.

Answer:

- What would the accuracy be if the model always predicted the most common author?
- Why is this useful as a baseline?

### 13. Train a Baseline Neural Network

Train a simple neural network using:

- `TextVectorization`
- `Embedding`
- `GlobalAveragePooling1D`
- `Dense`
- `Dropout`
- `Softmax` output

This model does not preserve word order. It is a useful comparison point.

### 14. Train a SimpleRNN Model

Train a SimpleRNN model using:

- `TextVectorization`
- `Embedding`
- `Bidirectional(SimpleRNN)`
- `Dense`
- `Dropout`
- `Softmax` output

Record the accuracy, loss curves and classification report.

### 15. Train an LSTM Model

Train an LSTM model using:

- `TextVectorization`
- `Embedding`
- `Bidirectional(LSTM)`
- `Dense`
- `Dropout`
- `Softmax` output

Record the accuracy, loss curves and classification report.

### 16. Compare Model Performance

Create a comparison table showing:

- majority-class baseline accuracy
- baseline neural network test accuracy
- SimpleRNN test accuracy
- LSTM test accuracy

Then answer:

- Which model performed best?
- Did the recurrent models improve over the baseline neural network?
- Was the improvement larger than in 20 Newsgroups?
- What does this suggest about dataset choice and model choice?

### 17. Visualisation

Create:

- bar chart of posts per selected author
- histogram of cleaned word counts
- training and validation accuracy plots for each model
- training and validation loss plots for each model
- confusion matrix for the best model

Do not convert the full Spark DataFrame to Pandas unless you have filtered or sampled it first.

### 18. Save the Prepared Dataset

Save the balanced cleaned subset for possible future use.

Save as:

```text
cleaned_blog_authorship_subset.csv
```

or:

```text
cleaned_blog_authorship_subset.parquet
```

## Discussion Questions

- Why should the number of authors be limited in this activity?
- Why is class balance important?
- Why might LSTM perform better on authorship classification than on topic classification?
- Why might LSTM still fail to outperform a simpler baseline?
- What is the difference between learning topic vocabulary and learning writing style?
- What are the ethical risks of authorship prediction?
