# 1. Phishing Email Classifier

## 1.1 Overview
This project is a machine learning application that detects phishing emails based on their content. It uses a Logistic Regression model trained on a large dataset of labeled emails to classify emails as phishing or legitimate.

The project includes:
- Data preprocessing and cleaning
- TF-IDF vectorization for text feature extraction
- Model training and evaluation
- A user-friendly web interface built with Streamlit to test emails interactively

## 1.2 Features
- High accuracy (~98%) in detecting phishing emails
- Text cleaning and feature extraction pipeline
- Save/load model and vectorizer to avoid retraining
- Streamlit app for easy input and real-time predictions

# 2. Getting Started

## 2.1 Prerequisites
- Python 3.8+
- pip package manager

## 2.2 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/zachfuellhart/phishing-email-classifier.git
   cd phishing-email-classifier
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
3. Install required packages:
   pip install -r requirements.txt
4. Train the model or load saved model/vectorizer by running:
   ```bash
   python AI_Phishing_ZF.py
5. Run the Streamlit app to test emails interactively:
   streamlit run app.py

# 3. Usage   

   - Open the Streamlit UI in your browser.
   - Paste or type an email content into the input box.
   - Click "Predict" to see if the email is classified as phishing or legitimate.

# 4. Model Performance   

   - Model achieves approximately 98% accuracy on the test dataset.
   - Includes classification report with precision, recall, and F1-score.
   
# 5. Contact

   Created by Zachary Fuellhart.
   email: znf5026@psu.edu
   GitHub: zachfuellhart
   Feel free to reach out for questions or collaboration.
