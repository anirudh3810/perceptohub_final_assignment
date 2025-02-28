import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


try:
    data_dict = pickle.load(open('./hand_gesture_data.pkl', 'rb'))
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'hand_gesture_data.pkl' not found.")
    exit()


data = data_dict['data']
labels = data_dict['labels']

print(f"Loaded {len(data)} samples with labels.")


data_lengths = [len(sample) for sample in data]
if len(set(data_lengths)) > 1:
    print(f"Inconsistent data lengths detected: {set(data_lengths)}")
    max_length = max(data_lengths)
    print(f"Padding all samples to the maximum length: {max_length}")
    
   
    padded_data = []
    for sample in data:
        if len(sample) < max_length:
          
            sample += [0] * (max_length - len(sample))
        elif len(sample) > max_length:
          
            sample = sample[:max_length]
        padded_data.append(sample)
    
    data = np.array(padded_data)
else:
    data = np.array(data)

labels = np.array(labels)

print(f"Data shape after processing: {data.shape}, Labels shape: {labels.shape}")


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
print("Data split into training and testing sets.")


model = RandomForestClassifier()


model.fit(x_train, y_train)
print("Model training successful.")


y_predict = model.predict(x_test)


accuracy = accuracy_score(y_test, y_predict)
print(f"Model accuracy: {accuracy * 100:.2f}%")


try:
    with open('./annotated_data/random_forest_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    print("Model saved successfully in 'annotated_data' folder as 'random_forest_model.pkl'.")
except Exception as e:
    print(f"Error saving the model: {e}")
