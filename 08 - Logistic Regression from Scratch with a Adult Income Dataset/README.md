# 08 - Logistic Regression from Scratch with Adult Income Dataset

## Overview

This activity introduces Logistic Regression from scratch using a real open dataset. In the previous activities, you used high-level libraries to train neural-network, recurrent and convolutional models. In this activity, you will slow down and implement the key parts of a Logistic Regression classifier yourself so that you understand the algorithm behind the model.

You will use the Adult Census Income dataset to predict whether a person earns more than 50,000 per year. The dataset is suitable for learning the Logistic Regression workflow because it is a real binary-classification dataset with more than 10,000 records, numeric features, categorical features, missing/unknown values, class imbalance and a clear target variable.

You will use Spark for exploratory data analysis, prepare features in Python, implement the sigmoid function, implement binary cross-entropy loss, train weights using gradient descent, evaluate the from-scratch model, and compare it against built-in Logistic Regression models.

Note: This activity focuses on the first working implementation and a fair built-in comparison. Deeper model improvement, convergence analysis, regularisation, solver choices and threshold tuning will be extended later when Gareth covers Logistic Regression improvement.

## Why are we doing this?

The course requires you to create a Logistic Regression algorithm from scratch. This means you need to understand the training process, not only how to call a library model.

You need to understand:

- target variables
- feature variables
- numeric and categorical features
- feature encoding
- feature scaling
- sigmoid function
- weights and bias
- predicted probabilities
- binary cross-entropy loss
- gradient descent
- thresholding predictions
- accuracy, precision, recall and F1-score
- how to compare your implementation with a built-in implementation

## Dataset

This activity uses the Adult Census Income dataset.

Prediction task:

```text
Predict whether income is >50K or <=50K.
```

Target column:

```text
income
```

Binary version of the target:

```text
income_over_50k
```

The original dataset contains demographic, education and employment-related variables. Some examples are:

- age
- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- hours-per-week
- native-country
- income

The notebook first looks for a local file called `adult.csv`. If it is not found, it downloads the dataset from the public UCI Machine Learning Repository files. No synthetic fallback is used.

## Resources to Consult

Use these resources to support your learning. Do not copy code blindly. Any copied or adapted code must be attributed in your notebook/report.
- UCI Machine Learning Repository: Adult dataset: https://archive.ics.uci.edu/dataset/2/adult
- scikit-learn Logistic Regression documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- scikit-learn model evaluation documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Apache Spark DataFrame documentation: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.html
- Apache Spark ML Logistic Regression documentation: https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression
- Google Machine Learning Crash Course: Logistic Regression: https://developers.google.com/machine-learning/crash-course/logistic-regression
- Google Machine Learning Crash Course: Classification metrics: https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall

## Tasks

### 1. Prepare the Environment

- Import NumPy, Pandas and Matplotlib.
- Import Spark.
- Import scikit-learn tools for splitting, encoding, scaling and evaluation.
- Set a random seed.

### 2. Load a Real Binary-Classification Dataset

Load the Adult Census Income dataset.

You should:

- load the data from `adult.csv` if available
- otherwise download the public Adult data files
- combine the train and test files if needed
- standardise the target values
- create a binary target column called `income_over_50k`

### 3. Use Spark for EDA

Convert the working data to a Spark DataFrame.

Use Spark to:

- display the schema
- count the records
- check missing values
- check class balance
- summarise numeric features
- inspect categorical values
- identify possible data-quality issues

### 4. Dataset Suitability Check

Write a Markdown section answering:

- What is the prediction target?
- Is the target binary?
- How many records are available?
- What are the feature variables?
- Which features are numeric?
- Which features are categorical?
- Which features may be useful predictors?
- Is the class distribution balanced?
- What data-quality issues must be checked?
- Why is this dataset suitable for learning Logistic Regression?

### 5. Prepare Features and Target

Prepare the modelling arrays.

You should:

- select feature columns
- separate `X` and `y`
- split into training and testing data
- impute missing values
- one-hot encode categorical features
- scale numeric features
- keep the same split for all models

### 6. Implement the Sigmoid Function

Implement:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```
Test the function on a small array of values.

Then, conduct research and explain why Logistic Regression outputs probabilities between 0 and 1.

### 7. Implement Prediction

Create a function that calculates:

- linear score
- predicted probability
- predicted class using a threshold

Use a default threshold of:

```text
0.5
```

### 8. Implement Binary Cross-Entropy Loss

Implement binary cross-entropy loss.

Explain:

- why loss measures model error
- why a lower loss is better

### 9. Implement Gradient Descent

Train the model from scratch.

Your training loop should:

- initialise weights and bias
- calculate predictions
- calculate loss
- calculate gradients
- update weights and bias
- store loss history

### 10. Evaluate the From-Scratch Model

Evaluate the model using:

- accuracy
- precision
- recall
- F1-score
- confusion matrix
- classification report

Explain which metric is most important for the scenario.

### 11. Compare with Built-in Logistic Regression Models

Train built-in Logistic Regression models on the same data split.

Use them as sanity checks, not as replacements for the from-scratch implementation.

You should compare against:

- scikit-learn `LogisticRegression`
- PySpark ML `LogisticRegression`, where Spark ML is available

Answer:

- Is your from-scratch model close to the built-in model?
- Which model performed best?
- Why might the built-in model perform differently?

### 12. Basic Improvement Experiment

Run one small improvement experiment on the from-scratch model, such as changing:

- learning rate
- number of iterations
- decision threshold

## Discussion Questions
- Why is Logistic Regression used for binary classification?
- What is the purpose of the sigmoid function?
- What do weights represent?
- Why do we scale features before gradient descent?
- Why do we one-hot encode categorical features?
- What does the loss curve tell us?
- Why can accuracy be misleading?
- Why should your from-scratch model be compared with built-in Logistic Regression implementations?

