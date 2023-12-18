
---

# Car Safety Rating Prediction using K-Nearest Neighbors

## Overview
This repository contains a machine learning project that applies the K-Nearest Neighbors (KNN) classification algorithm to predict car safety ratings. The project uses a dataset of cars, with features such as buying price, maintenance cost, number of doors, persons, lug boot size, and safety. The goal is to accurately predict the safety rating of a car based on these features.

## Dataset
The dataset used in this project is named `car.csv`. It consists of various car attributes, including:
- Buying price
- Maintenance cost
- Number of doors
- Number of persons the car can accommodate
- Size of the luggage boot
- Safety rating

The target variable is the safety rating of the car.

## Methodology

### Data Preprocessing
- **OneHotEncoding**: Categorical variables in the dataset are transformed into a format that can be provided to the machine learning algorithm.
- **Label Encoding**: The target variable (safety rating) is encoded from string labels to integers.

### Model Training
- **K-Nearest Neighbors Classifier**: The KNN algorithm is used for classification, with `n_neighbors=5` and `metric='minkowski'`.
- **Training and Testing**: The dataset is split into training and testing sets, with 80% of the data used for training and 20% for testing.

### Model Evaluation
- **Confusion Matrix**: Used to evaluate the accuracy of the classification.
- **Accuracy Score**: Calculates the proportion of correctly predicted instances.

## Requirements
- Python 3.x
- Pandas
- NumPy
- scikit-learn

## Usage
1. Clone the repository.
2. Run the script using a Python interpreter.
3. The script will train the KNN model on the training set and evaluate its performance on the test set.

## Results
- The output of the script includes the predictions made by the model, a confusion matrix, and the accuracy score.
- These results help in understanding the effectiveness of the KNN algorithm in classifying car safety ratings based on given features.

## Contributing
Contributions to this project are welcome. Feel free to fork the repository, add improvements, and submit a pull request.

---
