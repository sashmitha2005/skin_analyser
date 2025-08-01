print("Running model training...")

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('dermatology_database_updated.csv')

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert all columns except 'class' to numeric
for col in df.columns:
    if col != 'class':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

print("Dataset sample:")
print(df.head())
print("\nColumns in dataset:")
print(df.columns)

# Split into features and target
X = df.drop('class', axis=1)
y = df['class']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.2f}")

# Create Voting Classifier
print("\nTraining VotingClassifier...")
voting_clf = VotingClassifier(estimators=[
    ('lr', models['Logistic Regression']),
    ('rf', models['Random Forest']),
    ('nb', models['Naive Bayes']),
    ('svm', models['SVM']),
    ('knn', models['KNN'])
], voting='hard')

voting_clf.fit(X_train, y_train)

# Evaluate ensemble
voting_preds = voting_clf.predict(X_test)
voting_acc = accuracy_score(y_test, voting_preds)
print(f"Voting Classifier Accuracy: {voting_acc:.2f}")

# Ensure model folder exists
os.makedirs('models', exist_ok=True)

# Save model and label encoder
with open('models/voting_model.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("✅ Voting model and label encoder saved successfully!")
