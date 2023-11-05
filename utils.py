import numpy as np

# Split the dataset into training and testing sets Train: 40%, Validation:30%, Test:30%
def train_validate_test_split (data, labels, testRatio =0.3, valRatio =0.3, random_state=0):
    if testRatio < 0 or valRatio < 0 or testRatio + valRatio >=1:
        raise ValueError("Invalid test and validation ratio values")
    
    # Calculate the sizes of each split
    total_samples = len(data)
    test_size = int(testRatio * total_samples)
    val_size = int(valRatio * total_samples)
    train_size = total_samples - test_size - val_size
    
    # Randomize based on random_state
    if(random_state):
        np.random.seed(random_state)

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


# calculate_accuracy function
def accuracy_score(predicted_y, y):
    predections = np.sum(predicted_y == y)
    totalSamples = len(y)
    accuracy = predections/totalSamples
    return accuracy
