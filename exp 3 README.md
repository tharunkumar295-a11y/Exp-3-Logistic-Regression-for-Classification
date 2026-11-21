# Experiment 3: Logistic Regression for Classification

## Aim

To compute descriptive statistics and probabilities using NumPy and
Pandas.

## Algorithm

1.  **Start.**

2.  Load the dataset and extract features `X` and target `y`.

3.  Split the data into training and test sets (e.g., 80:20).

4.  Standardize features using `StandardScaler`.

5.  Train `LogisticRegression` on the training data.

    The logistic (sigmoid) function is:

    \[ `\sigma`{=tex}(z) = `\frac{1}{1 + e^{-z}}`{=tex} \]

    where

    \[ z = `\beta`{=tex}\_0 + `\beta`{=tex}\_1 x_1 + `\cdots `{=tex}+
    `\beta`{=tex}\_p x_p \]

6.  Predict labels on the test data.

7.  Evaluate performance using accuracy, confusion matrix, and
    classification report.

8.  Print the evaluation metrics.

9.  **Stop.**



## Program
# Experiment 3: Logistic Regression for Classification

# Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print("Accuracy Score:")
print(accuracy)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)


## output
<img width="525" height="407" alt="image" src="https://github.com/user-attachments/assets/2978e4e6-f360-4c76-9d86-42774c9f374c" />



## Result
Logistic Regression was successfully implemented, and the classification performance was evaluated using accuracy score, confusion matrix, and classification report.
