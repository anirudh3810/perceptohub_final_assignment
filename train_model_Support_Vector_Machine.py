import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


try:
    data_dict = pickle.load(open('./hand_gesture_data.pkl', 'rb'))
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'hand_gesture_data.pkl' not found.")
    exit()


data = data_dict['data']
labels = data_dict['labels']


expected_length = 42  


def pad_or_truncate(sample, expected_length):
    if len(sample) < expected_length:
        return sample + [0] * (expected_length - len(sample))  
    return sample[:expected_length]  


processed_data = [pad_or_truncate(sample, expected_length) for sample in data]


data = np.array(processed_data)
labels = np.array(labels)

print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
print("Data split into training and testing sets.")


svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)


print("Training SVM model...")
svm_model.fit(x_train, y_train)
print("SVM model trained successfully.")


y_predict_svm = svm_model.predict(x_test)


accuracy_svm = accuracy_score(y_test, y_predict_svm)
print(f"SVM Model Accuracy: {accuracy_svm * 100:.2f}%")


with open('./svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm_model, model_file)
print("SVM model saved as 'svm_model.pkl'.")
