import numpy as np

# Split the dataset into training and testing sets Train: 40%, Validation:30%, Test:30%
def train_validate_test_split (data, labels, test_ratio =0.3, val_ratio =0, random_state=0):
    if test_ratio < 0 or val_ratio < 0 or test_ratio + val_ratio >=1:
        raise ValueError("Invalid test and validation ratio values")
    
    # Calculate the sizes of each split
    total_samples = len(data)
    test_size = int(test_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    train_size = total_samples - test_size - val_size
    
    # Randomize based on random_state
    if(random_state):
        np.random.seed(random_state)


    # Shuffle the data and the labels
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    # Split the data and labels
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    train_data, val_data, test_data = data[train_indices], data[val_indices], data[test_indices]
    train_labels, val_labels, test_labels = labels[train_indices], labels[val_indices], labels[test_indices]
    
    return train_data, val_data, test_data, train_labels, val_labels, test_labels


# calculate_accuracy function
def accuracy_score(predicted_y, y):
    predections = np.sum(predicted_y == y)
    totalSamples = len(y)
    accuracy = predections/totalSamples
    return accuracy
