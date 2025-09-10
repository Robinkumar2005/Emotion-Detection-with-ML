import streamlit as st
import joblib
import re
from scipy.stats import mode
import numpy as np

# ===== 1️⃣ Load Pickled Models and Vectorizer =====
lr_model = joblib.load('lr_model.pkl')
svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# ===== 2️⃣ Numeric-to-Label Mapping with Emojis =====
emotion_labels = {
    0: ("Sadness", "😢"),
    1: ("Anger", "😠"),
    2: ("Love", "❤️"),
    3: ("Surprise", "😲"),
    4: ("Fear", "😨"),
    5: ("Joy", "😄")
}

# ===== 3️⃣ Text Preprocessing Function =====
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===== 4️⃣ Prediction Function =====
def predict_emotion(text):
    text_clean = preprocess_text(text)
    X_vect = tfidf_vectorizer.transform([text_clean])
    lr_pred = lr_model.predict(X_vect)
    svm_pred = svm_model.predict(X_vect)
    ensemble_pred = mode(np.vstack([lr_pred, svm_pred]), axis=0).mode.flatten()[0]
    label, emoji = emotion_labels[ensemble_pred]
    return label, emoji

# ===== 5️⃣ Streamlit UI =====
st.set_page_config(page_title="Interactive Emotion Detection", layout="wide")
st.title("🧠 Emotion Detection from Text")
st.write("Detect the emotion behind any text instantly!Just type or paste a sentence, and the model will predict if it expresses joy, surprise, sadness, anger, love or fear.")


# ===== Sidebar Enhancements =====
st.sidebar.header("Tips for Better Predictions")
st.sidebar.write("- Use longer sentences for better context")
st.sidebar.write("- Include trigger words: wow, amazing, love")
st.sidebar.write("- Try describing emotions clearly")
# Sidebar with example prompts
st.sidebar.header("💡 Example Prompts")
st.sidebar.write("""
- Sadness: 'I feel so lonely today.'  
- Anger: 'I am so frustrated with this situation!'  
- Love: 'I adore spending time with you.'  
- Surprise: 'I’m stunned by what just occurred!'  
- Fear: 'I’m scared of what might happen.'  
- Joy: 'I’m thrilled with my new achievement!'
""")

user_input = st.text_area("Type your text here:")

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        label, emoji = predict_emotion(user_input)
        st.markdown(f"### Predicted Emotion: {emoji} **{label}**")
        # Optional: Add colored background
        color_map = {
            "Sadness": "#a0c4ff",
            "Anger": "#ffadad",
            "Love": "#ffafcc",
            "Surprise": "#fdffb6",
            "Fear": "#9bf6ff",
            "Joy": "#caffbf"
        }
        st.markdown(f"<div style='background-color:{color_map[label]};padding:15px;border-radius:5px'>Try expressing another emotion!</div>", unsafe_allow_html=True)
