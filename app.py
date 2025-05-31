import streamlit as st
import joblib
import re
import string
from sklearn.exceptions import NotFittedError

# Load saved model and vectorizer
model = joblib.load("phishing_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Text cleaning function (same as training)
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

# Streamlit UI
st.title("Phishing Email Classifier")
st.write("There was an error running the application. Try again soon.")

# Input box
user_input = st.text_area("Email Text", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some email text!")
    else:
        cleaned = clean_text(user_input)
        try:
            vector = vectorizer.transform([cleaned])
        except NotFittedError:
            st.error("There was an error running the application.")
            st.stop()

        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("⚠️ This email is likely PHISHING.")
        else:
            st.success("✅ This email is likely NOT phishing.")


