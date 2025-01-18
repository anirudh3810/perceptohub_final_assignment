import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


data_dict = pickle.load(open('./hand_gesture_data.pkl', 'rb'))


data = data_dict['data']
labels = data_dict['labels']


expected_length = 42 


def pad_data(data_aux, expected_length):
    if len(data_aux) < expected_length:
      
        data_aux.extend([0] * (expected_length - len(data_aux)))
    return data_aux[:expected_length] 


processed_data = [pad_data(d, expected_length) for d in data]


data = np.array(processed_data)
labels = np.array(labels)


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)


dt_model = DecisionTreeClassifier(random_state=42)


dt_model.fit(x_train, y_train)


y_predict_dt = dt_model.predict(x_test)


accuracy_dt = accuracy_score(y_test, y_predict_dt)
print(f"Decision Tree Model Accuracy: {accuracy_dt * 100:.2f}%")


with open('decision_tree_model.pkl', 'wb') as model_file:
    pickle.dump(dt_model, model_file)

print("Decision Tree model saved as 'decision_tree_model.pkl'.")
