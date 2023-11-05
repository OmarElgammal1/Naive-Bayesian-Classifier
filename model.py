from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from utils import train_validate_test_split
from utils import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load the iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split the dataset into training and testing sets Train: 40%, Validation:30%, Test:30%
X_train, X_val, X_test, y_train, y_val, y_test = train_validate_test_split(X, y, testRatio=0.3, valRatio=0.3, random_state=43)

# Gaussian Naive Bayes classifier
naiveBayes = GaussianNB()

# Train the classifier on the training data
naiveBayes.fit(X_train, y_train)

# Use the classifier to make predictions on the validation data
y_validation_pred = naiveBayes.predict(X_val)

# Calculate accuracy on the validation data
accuracy_validation = accuracy_score(y_val, y_validation_pred)
print("Validation Accuracy:", accuracy_validation)

# Use the classifier to make predictions on the testing data
y_test_pred = naiveBayes.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy:", accuracy)

# Create a list of feature pairs to visualize
feature_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
# Create subplots for each feature pair
plt.figure(figsize=(12, 10))
for i, (feature1, feature2) in enumerate(feature_pairs, 1):
    plt.subplot(3, 2, i)

    # Select the two features for this pair
    X_pair = X[:, [feature1, feature2]]
    
    # Train the classifier on the selected features
    naiveBayes.fit(X_pair, y)

    # Define a meshgrid for decision boundary visualization
    x_min, x_max = X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1
    y_min, y_max = X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))

    # Use the classifier to make predictions on the meshgrid
    Z = naiveBayes.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary along with the data points
    plt.contourf(xx, yy, Z, alpha=0.6)
    plt.scatter(X_pair[:, 0], X_pair[:, 1], c=y, edgecolor='k')
    plt.xlabel(f'Feature {features[feature1]}')
    plt.ylabel(f'Feature {features[feature2]}')
    plt.title(f'Decision Boundary for Features {features[feature1]} and {features[feature2]}')

plt.tight_layout()
plt.show()