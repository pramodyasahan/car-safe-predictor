# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

# Loading the dataset from a CSV file
dataset = pd.read_csv('car.csv')

# Extracting features (X) and target variable (y) from the dataset
X = dataset.iloc[:, :-1].values  # Selecting all rows and all but the last column as features
y = dataset.iloc[:, -1].values  # Selecting all rows and only the last column as the target variable

# Applying OneHotEncoder to categorical columns and converting the output to a dense array
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1, 2, 3, 4])], remainder='passthrough',
                       sparse_threshold=0)
X = np.array(ct.fit_transform(X))

# Printing the transformed features to check the result of one-hot encoding
print(X)

# Applying LabelEncoder to the target variable to encode string labels to integers
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset into Training and Test sets (20% data for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating and training the K-Nearest Neighbors classifier
classifier = KNeighborsClassifier(n_neighbors=5,
                                  metric='minkowski')  # Using 'minkowski' metric which is generalization of Euclidean and Manhattan distance
classifier.fit(X_train, y_train)  # Fitting the classifier to the training data

# Predicting the target variable for the test set
y_pred = classifier.predict(X_test)

# Printing the predicted and actual values side by side for comparison
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Generating a confusion matrix to evaluate the performance of the classification
cm = confusion_matrix(y_test, y_pred)
print(cm)  # Printing the confusion matrix

# Calculating and printing the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)  # Printing the accuracy score
