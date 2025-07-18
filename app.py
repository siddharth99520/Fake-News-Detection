import streamlit as st
import joblib
import nltk
import re
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Add local NLTK data path (for deployment)
nltk.data.path.append('./')

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocessing setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
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
