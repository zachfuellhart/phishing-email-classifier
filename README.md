# 1. Phishing Email Classifier

## 1.1 Overview
This project is a machine learning application that detects phishing emails based on their content. It uses a Logistic Regression model trained on a large dataset of labeled emails to classify emails as phishing or legitimate.

Preview of application UI:
![image](https://github.com/user-attachments/assets/cfe8cc4a-3834-4e2e-a18d-8e9e0b6956c0)

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
   ```bash
   pip install -r requirements.txt
5. Train the model or load saved model/vectorizer by running:
   ```bash
   python AI_Phishing_ZF.py
6. Run the Streamlit app to test emails interactively:
   ```bash
   streamlit run app.py

## 2.3 Dataset
The phishing email dataset used to train this model was sourced from Keggle. 
It contains over 82,000 labeled emails (phishing and legitimate) and is publicly available at: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset Citation: *Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman, S. A. U. (2024, May 19). Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection. ArXiv.org. https://arxiv.org/abs/2405.11619*

## 2.4 Project Structure
phishing-email-classifier/
- ├── .gitignore               # Specifies intentionally untracked files to ignore
- ├── AI_Phishing_ZF.py        # Core script to train and save the model
- ├── LICENSE                  # MIT License file
- ├── README.md                # Project overview and usage guide
- ├── app.py                   # Streamlit app to test emails
- ├── phishing_model.joblib    # Saved trained model
- ├── requirements.txt         # Required Python packages
- ├── runtime.txt              # Python version specification for deployment (e.g., python-3.10.11)
- ├── test_phishing_model.py   # Script to test the model's performance
- ├── vectorizer.joblib        # Saved TF-IDF vectorizer

# 3. Usage   

   - Open the Streamlit UI in your browser.
   - Paste or type an email content into the input box.
   - Click "Predict" to see if the email is classified as phishing or legitimate.

# 4. Model Performance   

   - Model achieves approximately 98% accuracy on the test dataset.
   - Includes classification report with precision, recall, and F1-score.

# 5. License  
   This project is licensed under the MIT License - see the LICENSE file for details.

# 6. Contact

   - Created by Zachary Fuellhart.
   - email: znf5026@psu.edu
   - GitHub: zachfuellhart
   - Feel free to reach out for questions or collaboration.
