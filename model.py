from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split the dataset into training and testing sets Train: 40%, Validation:30%, Test:30%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Gaussian Naive Bayes classifier
naiveBayes = GaussianNB()

# Train the classifier on the training data
naiveBayes.fit(X_train, y_train)

# Use the classifier to make predictions on the validation data
y_validation_pred = naiveBayes.predict(X_validation)

# Calculate accuracy on the validation data
accuracy_validation = accuracy_score(y_validation, y_validation_pred)
print("Validation Accuracy:", accuracy_validation)

# Use the classifier to make predictions on the testing data
y_test_pred = naiveBayes.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy:", accuracy)