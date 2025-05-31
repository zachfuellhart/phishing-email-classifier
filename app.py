import streamlit as st
import joblib
import re
import string
import os
from sklearn.exceptions import NotFittedError

# Set up base directory and paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "phishing_model.joblib")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.joblib")

# Optional debug output
st.write("Current working directory:", BASE_DIR)
st.write("Files in directory:", os.listdir(BASE_DIR))

# Load saved model
if not os.path.exists(model_path):
    st.error(f"Model not found at {model_path}")
    st.stop()

model = joblib.load(model_path)

# Load vectorizer
if not os.path.exists(vectorizer_path):
    st.error(f"Vectorizer not found at {vectorizer_path}")
    st.stop()

vectorizer = joblib.load(vectorizer_path)

# Text cleaning function (same as training)
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

# Streamlit UI
st.title("Phishing Email Classifier")
st.write("Enter an email text below to check if it's phishing or not.")

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
            st.error(f"Error transforming input: {e}")
            st.stop()

        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("⚠️ This email is likely PHISHING.")
        else:
            st.success("✅ This email is likely NOT phishing.")

