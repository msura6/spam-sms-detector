import pandas as pd

# Load dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])

# Show first 5 rows
print(df.head())

# Convert 'spam' to 1 and 'ham' to 0
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Check label distribution
print("\nLabel distribution:")
print(df['label'].value_counts())

from sklearn.model_selection import train_test_split

# Split into features (X) and labels (y)
X = df['message']
y = df['label']

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit on training data and transform both train and test
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Vectorized training data shape: {X_train_vec.shape}")

from sklearn.naive_bayes import MultinomialNB

# Create and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

print("\nModel training complete.")

from sklearn.metrics import classification_report, accuracy_score

# Predict on test data
y_pred = model.predict(X_test_vec)

# Print accuracy
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# --- Predict Custom Messages ---
print("\nType your own messages to test (type 'exit' to quit):")

while True:
    user_input = input("Enter a message: ")
    if user_input.lower() == 'exit':
        break

    # Vectorize and predict
    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)[0]

    if prediction == 1:
        print("Prediction: 🚨 Spam")
    else:
        print("Prediction: ✅ Ham")

import pickle

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
