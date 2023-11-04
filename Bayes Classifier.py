from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the iris dataset
iris_model = load_iris()
X = iris_model.data
y = iris_model.target

# Split the dataset into training and testing sets Train: 40%, Validation:30%, Test:30%
def train_validate_test_split (data, labels, testRatio =0.3, valRatio =0.3, random_state = 40):
       
    if testRatio < 0 or valRatio < 0 or testRatio + valRatio >=1:
        raise ValueError("Invalid test and validation ratio values")
    
    # Calculate the sizes of each split
    total_samples = len(data)
    test_size = int(testRatio * total_samples)
    val_size = int(valRatio * total_samples)
    train_size = total_samples - test_size - val_size
    
    # Shuffle the data and the labels
    combined = list(zip(data, labels))
    np.random.shuffle(combined)
    data, labels = zip(*combined)
    
    # Split the data and labels
    train_data = data[:train_size] 
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    train_labels = labels[:train_size] 
    val_labels = labels[train_size:train_size + val_size] 
    test_labels = labels[train_size + val_size:]
    return train_data, val_data, test_data, train_labels, val_labels, test_labels

# Split data and labels into test, validation and trainig sets 
X_train, X_val, X_test, y_train, y_val, y_test = train_validate_test_split(X, y, testRatio=0.3, valRatio=0.3)

# Gaussian Naive Bayes classifier
naiveBayes = GaussianNB()

# Train the classifier on the training data
# the model learns to make predictions by matching features in X_train with the target labels in y_train.
naiveBayes.fit(X_train, y_train)

# Use the classifier to make predictions on the validation data
y_validation_pred = naiveBayes.predict(X_val)

# calculate_accuracy function
def calculate_accuracy(predicted_y, y):
    predections = np.sum(predicted_y == y)
    totalSamples = len(y)
    accuracy = predections/totalSamples
    return accuracy
# calculate accuracy for validation set
validation_accuracy = calculate_accuracy(y_validation_pred, y_val)
print("Validation accuracy = ", validation_accuracy)