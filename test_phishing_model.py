import joblib
import re
import string

# Text cleaning function (same as in training)
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

# Load the saved vectorizer and model
vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("phishing_model.joblib")

def predict_email(email_text):
    cleaned = clean_text(email_text)
    vector = vectorizer.transform([cleaned])  # transform the single email
    prediction = model.predict(vector)[0]
    return prediction

# Example emails to test
test_emails = [
    "Hey your project is due at 11:59 tonight.",
    "Please find attached the meeting notes from yesterday's call.",
    "Urgent: Your account has been compromised. Reset your password immediately.",
]

for i, email in enumerate(test_emails):
    pred = predict_email(email)
    label = "Phishing" if pred == 1 else "Legitimate"
    print(f"Email {i+1} prediction: {label}")

