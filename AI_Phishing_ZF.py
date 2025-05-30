import pandas as pd
import re
import string
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data_path = r"C:\Users\zachf\Downloads\archive\phishing_email.csv"
df = pd.read_csv(data_path)

# Show the first 5 rows
print(df.head())

# Show info about the dataset
print(df.info())

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = ' '.join(text.split())  # Remove extra spaces
    return text

# Apply cleaning to the text_combined column
df['cleaned_text'] = df['text_combined'].apply(clean_text)

# Vectorize the cleaned text using TF-IDF
vectorizer_path = "vectorizer.joblib"
if os.path.exists(vectorizer_path):
    print("Loading saved vectorizer...")
    vectorizer = joblib.load(vectorizer_path)
    X = vectorizer.transform(df['cleaned_text'])  # transform only
else:
    print("Creating and fitting new vectorizer...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text'])  # fit and transform
    joblib.dump(vectorizer, vectorizer_path)          # save fitted vectorizer

# Labels
y = df['label']

print(f"Feature matrix shape: {X.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train or load the model
model_path = "phishing_model.joblib"
if os.path.exists(model_path):
    print("Loading saved model...")
    model = joblib.load(model_path)
else:
    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print("Model saved to disk.")

# Make predictions
y_pred = model.predict(X_test)

# Print metrics with 3 decimal places
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

