import streamlit as st
import joblib
import os
import re
import string
from sklearn.exceptions import NotFittedError

# Debug info to verify deployment directory and file presence
st.write("Current working directory:", os.getcwd())
st.write("Files in working directory:", os.listdir())

# Define absolute paths to model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "phishing_model.joblib")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.joblib")

# Load the model and vectorizer with error handling
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError as e:
    st.error(f"File not found: {e.filename}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error while loading model or vectorizer: {e}")
    st.stop()

# Text cleaning function (same as used in training)
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

# Streamlit UI
st.title("Phishing Email Classifier")
st.write("Enter an email message below to check if it's phishing or not.")

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
            st.error("Vectorizer is not fitted. Please retrain the model.")
            st.stop()
        except Exception as e:
            st.error(f"Error during vectorization: {e}")
            st.stop()

        try:
            prediction = model.predict(vector)[0]
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        if prediction == 1:
            st.error("⚠️ This email is likely PHISHING.")
        else:
            st.success("✅ This email is likely NOT phishing.")
