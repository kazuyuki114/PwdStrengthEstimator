import pandas as pd
import numpy as np
import string
import math
from collections import Counter
from Levenshtein import distance as levenshtein_distance
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load Preprocessed Data
data_path = "../../processed_data/processed_data.csv"
df = pd.read_csv(data_path)

# Separate Features & Labels
X = df.drop(columns=['Strength_Level'])  # Features
y = df['Strength_Level']                 # Labels (Target)

# Split into Training & Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize Features for MLP (Recommended for Neural Networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Define MLP Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input Layer
    keras.layers.Dense(32, activation='relu'),  # Hidden Layer 1
    keras.layers.Dense(16, activation='relu'),  # Hidden Layer 2
    keras.layers.Dense(len(y.unique()), activation='softmax')  # Output Layer (Multi-class classification)
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Evaluate Model on Test Data
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Show Detailed Classification Report
print(classification_report(y_test, y_pred))

# Save the model with pickle
with open('../../models/mlp_model.pkl', 'wb') as file:
    pickle.dump(model, file)