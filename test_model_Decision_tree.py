import pickle
import numpy as np
from sklearn.metrics import accuracy_score


with open('decision_tree_model.pkl', 'rb') as model_file:
    dt_model = pickle.load(model_file)


data_dict = pickle.load(open('./hand_gesture_data.pkl', 'rb'))


test_data = data_dict['data']
test_labels = data_dict['labels']


expected_length = 42 


def pad_data(data_aux, expected_length):
    if len(data_aux) < expected_length:
      
        data_aux.extend([0] * (expected_length - len(data_aux)))
    return data_aux[:expected_length] 


processed_test_data = [pad_data(d, expected_length) for d in test_data]


processed_test_data = np.array(processed_test_data)
test_labels = np.array(test_labels)


predictions = dt_model.predict(processed_test_data)


test_accuracy = accuracy_score(test_labels, predictions)
print(f"Decision Tree Model Test Accuracy: {test_accuracy * 100:.2f}%")


print("First 10 predictions:", predictions[:10])
print("First 10 actual labels:", test_labels[:10])
