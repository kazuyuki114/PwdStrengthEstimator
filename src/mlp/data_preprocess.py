import math
import string
from collections import Counter

import pandas as pd
from Levenshtein import distance as levenshtein_distance

data_path = "../../data/pwlds_full.csv"

df = pd.read_csv(data_path).dropna()
print(df[:10])

# Shannon Entropy Calculation
def shannon_entropy(password):
    if not password:
        return 0
    freq = Counter(password)
    length = len(password)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())

# Character Diversity Score
def char_diversity_score(password):
    if not password:
        return 0
    charset = set()
    for c in password:
        charset.add(c)
    return len(charset) / len(password)

# Longest Consecutive Repeating Character Sequence
def longest_repeating_sequence(password):
    max_count, count = 1, 1
    for i in range(1, len(password)):
        if password[i] == password[i - 1]:
            count += 1
        else:
            count = 1
        max_count = max(max_count, count)
    return max_count

# Longest Consecutive Numeric Sequence
def longest_numeric_sequence(password):
    max_count, count = 0, 0
    for char in password:
        if char.isdigit():
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count

# Longest Consecutive Letter Sequence
def longest_letter_sequence(password):
    max_count, count = 0, 0
    for char in password:
        if char.isalpha():
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count



# Define common password
def load_common_passwords(path):
    with open(path, "r", encoding="utf-8") as f:
        common_passwords = set(line.strip() for line in f if line.strip())  # Remove empty lines
    return common_passwords

file_path = "../../data/top1k-vn-passwords.txt"
common_passwords = load_common_passwords(file_path)

# Print Sample Data
print(f"Loaded {len(common_passwords)} common passwords.")
print(list(common_passwords)[:10])  # Show first 10 passwords


# Check if the password contains a common password
def contains_common_password(password):
    return int(password.lower() in common_passwords)


# Check if the reversed password contains a common password
def contains_reversed_password(password):
    return int(password.lower()[::-1] in common_passwords)


# Compute Levenshtein distance to common passwords
def min_levenshtein_distance(password):
    return min([levenshtein_distance(password, common) for common in common_passwords])

# Feature extraction
def extract_features(df):
    print("Starting Feature Extraction...")

    df['password_length'] = df['Password'].apply(len)
    print("Extracted: Password Length")

    df['num_uppercase'] = df['Password'].apply(lambda x: sum(1 for c in x if c.isupper()))
    print("Extracted: Number of Uppercase Letters")

    df['num_lowercase'] = df['Password'].apply(lambda x: sum(1 for c in x if c.islower()))
    print("Extracted: Number of Lowercase Letters")

    df['num_digits'] = df['Password'].apply(lambda x: sum(1 for c in x if c.isdigit()))
    print("Extracted: Number of Digits")

    df['num_special_chars'] = df['Password'].apply(lambda x: sum(1 for c in x if c in string.punctuation))
    print("Extracted: Number of Special Characters")

    df['shannon_entropy'] = df['Password'].apply(shannon_entropy)
    print("Extracted: Shannon Entropy")

    df['char_diversity_score'] = df['Password'].apply(char_diversity_score)
    print("Extracted: Character Diversity Score")

    df['longest_repeating_seq'] = df['Password'].apply(longest_repeating_sequence)
    print("Extracted: Longest Repeating Sequence")

    df['longest_numeric_seq'] = df['Password'].apply(longest_numeric_sequence)
    print("Extracted: Longest Numeric Sequence")

    df['longest_letter_seq'] = df['Password'].apply(longest_letter_sequence)
    print("Extracted: Longest Letter Sequence")

    df['contains_common_password'] = df['Password'].apply(contains_common_password)
    print("Extracted: Contains Common Password")

    df['contains_reversed_password'] = df['Password'].apply(contains_reversed_password)
    print("Extracted: Contains Reversed Password")

    df['levenshtein_distance'] = df['Password'].apply(min_levenshtein_distance)
    print("Extracted: Levenshtein Distance")

    print("Feature Extraction Completed!")
    return df

# Apply feature extraction
df = extract_features(df)

# Drop original password column
df.drop(columns=['Password'], inplace=True)

# Save processed data
df.to_csv("processed_password_data.csv", index=False)

print(df.head())