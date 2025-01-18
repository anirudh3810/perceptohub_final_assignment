import pickle
import numpy as np
from sklearn.metrics import accuracy_score


try:
    with open('./svm_model.pkl', 'rb') as model_file:
        svm_model = pickle.load(model_file)
    print("SVM model loaded successfully.")
except FileNotFoundError:
    print("Error: 'svm_model.pkl' not found.")
    exit()


try:
    data_dict = pickle.load(open('./hand_gesture_data.pkl', 'rb'))
    print("Test data loaded successfully.")
except FileNotFoundError:
    print("Error: 'hand_gesture_data.pkl' not found.")
    exit()


test_data = data_dict['data']
test_labels = data_dict['labels']


expected_length = 42  


def pad_or_truncate(sample, expected_length):
    if len(sample) < expected_length:
        return sample + [0] * (expected_length - len(sample))  
    return sample[:expected_length] 


processed_test_data = [pad_or_truncate(sample, expected_length) for sample in test_data]


test_data = np.array(processed_test_data)
test_labels = np.array(test_labels)

print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")


predictions = svm_model.predict(test_data)


test_accuracy = accuracy_score(test_labels, predictions)
print(f"SVM Model Test Accuracy: {test_accuracy * 100:.2f}%")


print("First 10 predictions:", predictions[:10])
print("First 10 actual labels:", test_labels[:10])
