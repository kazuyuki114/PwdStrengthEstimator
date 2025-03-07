import pickle
import pandas as pd
import numpy as np
import sys
import math
from collections import Counter
from Levenshtein import distance as levenshtein_distance
from sklearn.preprocessing import StandardScaler

# Load common passwords (keep this at top)
def load_common_passwords(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

common_passwords = load_common_passwords("../../data/top1k-vn-passwords.txt")

# Feature calculation functions (keep these as-is)
def shannon_entropy(password):
    if not password:
        return 0
    freq = Counter(password)
    return -sum((count / len(password)) * math.log2(count / len(password)) for count in freq.values())

def char_diversity_score(password):
    return len(set(password)) / len(password) if password else 0

def longest_repeating_sequence(password):
    max_count = count = 1
    for i in range(1, len(password)):
        count = count + 1 if password[i] == password[i - 1] else 1
        max_count = max(max_count, count)
    return max_count

def longest_numeric_sequence(password):
    max_count = count = 0
    for char in password:
        count = count + 1 if char.isdigit() else 0
        max_count = max(max_count, count)
    return max_count

def longest_letter_sequence(password):
    max_count = count = 0
    for char in password:
        count = count + 1 if char.isalpha() else 0
        max_count = max(max_count, count)
    return max_count

def contains_common_password(password):
    return int(password.lower() in common_passwords)

def contains_reversed_password(password):
    return int(password.lower()[::-1] in common_passwords)

# Load the saved model (using pickle)
with open('../../models/mlp_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the fitted scaler from file
with open('../../models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Main execution
if len(sys.argv) > 1:
    password = sys.argv[1]

    # Calculate all features
    features = {
        'password_length': len(password),
        'num_uppercase': sum(1 for c in password if c.isupper()),
        'num_lowercase': sum(1 for c in password if c.islower()),
        'num_digits': sum(1 for c in password if c.isdigit()),
        'num_special_chars': len([c for c in password if not c.isalnum()]),
        'shannon_entropy': shannon_entropy(password),
        'char_diversity_score': char_diversity_score(password),
        'longest_repeating_seq': longest_repeating_sequence(password),
        'longest_numeric_seq': longest_numeric_sequence(password),
        'longest_letter_seq': longest_letter_sequence(password),
        'contains_common_password': contains_common_password(password),
        'contains_reversed_password': contains_reversed_password(password),
    }

    # Define feature order (must match training)
    feature_order = [
        'password_length',
        'num_uppercase',
        'num_lowercase',
        'num_digits',
        'num_special_chars',
        'shannon_entropy',
        'char_diversity_score',
        'longest_repeating_seq',
        'longest_numeric_seq',
        'longest_letter_seq',
        'contains_common_password',
        'contains_reversed_password',
    ]

    # Create DataFrame with correct column order
    input_df = pd.DataFrame([features])[feature_order]
    print("Raw Features:\n", input_df)

    # Apply the loaded scaler to standardize the input features
    input_df_scaled = scaler.transform(input_df)
    print("Scaled Features:\n", input_df_scaled)

    # Make prediction using the loaded model
    prediction = model.predict(input_df_scaled)
    predicted_class = int(np.argmax(prediction, axis=1)[0])

    # Map predicted class index to a human-readable strength label
    strength_labels = {0: 'very_weak', 1: 'weak', 2: 'average', 3: 'strong', 4: 'very_strong'}
    print("Prediction:", strength_labels[predicted_class])
else:
    print("Error: Please provide a password as command line argument")
