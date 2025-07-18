import streamlit as st
import joblib
import nltk
import re
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.data import find

# Ensure nltk_data folder exists and use it
NLTK_DATA_PATH = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Function to check and download NLTK resources
def download_nltk_resource(resource):
    try:
        find(resource)
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_PATH)

# Download only if missing
download_nltk_resource("punkt")
download_nltk_resource("stopwords")
download_nltk_resource("wordnet")

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# UI
st.title("📰 Fake News Detection App")
st.markdown("Enter a news article below to check whether it's **Real** or **Fake**.")

user_input = st.text_area("Enter News Article", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        clean_text = preprocess(user_input)
        vector = vectorizer.transform([clean_text])
        prediction = model.predict(vector)[0]
        if prediction == 1:
            st.success("✅ This looks like **Real News**.")
        else:
            st.error("🚨 This appears to be **Fake News**.")
