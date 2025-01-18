import pickle
import numpy as np
from sklearn.metrics import accuracy_score


try:
    with open('./annotated_data/random_forest_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Random Forest model loaded successfully.")
except FileNotFoundError:
    print("Error: 'random_forest_model.pkl' not found.")
    exit()


try:
    data_dict = pickle.load(open('./hand_gesture_data.pkl', 'rb'))
    print("Test data loaded successfully.")
except FileNotFoundError:
    print("Error: 'hand_gesture_data.pkl' not found.")
    exit()


test_data = data_dict['data']
test_labels = data_dict['labels']


data_lengths = [len(sample) for sample in test_data]
if len(set(data_lengths)) > 1:
    print(f"Inconsistent test data lengths detected: {set(data_lengths)}")
    max_length = max(data_lengths)
    print(f"Padding all test samples to the maximum length: {max_length}")

    
    padded_data = []
    for sample in test_data:
        if len(sample) < max_length:
           
            sample += [0] * (max_length - len(sample))
        elif len(sample) > max_length:
            
            sample = sample[:max_length]
        padded_data.append(sample)
    
    test_data = np.array(padded_data)
else:
    test_data = np.array(test_data)

test_labels = np.array(test_labels)
print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")


predictions = model.predict(test_data)


test_accuracy = accuracy_score(test_labels, predictions)
print(f"Random Forest Model Test Accuracy: {test_accuracy * 100:.2f}%")


print("First 10 predictions:", predictions[:10])
print("First 10 actual labels:", test_labels[:10])
