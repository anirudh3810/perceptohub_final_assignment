# perceptohub_final_assignment
Overview

This project focuses on developing a Hand Sign Detection model that utilizes hand landmarks for recognizing specific gestures. The workflow includes data collection, feature extraction, and the implementation of three machine learning models: Support Vector Machine (SVM), Random Forest, and Decision Tree.

Features

Hand Landmarks: Key points on the hand were identified and used as features for classification.

Machine Learning Models:

Support Vector Machine (SVM)

Random Forest Classifier

Decision Tree Classifier

Custom Dataset: A dataset was collected and annotated with various hand gestures.

Project Workflow

1. Data Collection

Hand gesture data was collected using a camera and labeled with corresponding gesture classes. Each sample includes landmarks of key points on the hand (e.g., fingertips, joints).

2. Preprocessing

Extracted hand landmarks using a hand tracking library (e.g., MediaPipe or similar).

Normalized landmark coordinates to improve model performance.

3. Model Training

Three machine learning algorithms were implemented and compared:

Support Vector Machine (SVM): Used for its robustness in high-dimensional spaces.

Random Forest: Leveraged for its ability to handle non-linear data effectively.

Decision Tree: Selected for its simplicity and interpretability.

4. Model Evaluation

Each model was evaluated using standard metrics such as accuracy, precision, recall, and F1-score. Cross-validation was employed to ensure the robustness of results.
