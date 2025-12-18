import streamlit as st
import pickle
import re
import string

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Sarcasm Detection App",
    page_icon="😏",
    layout="centered"
)

st.title("😏 Sarcasm Detection App")
st.write("Detect whether a sentence or headline is **Sarcastic** or **Not Sarcastic**.")

# -------------------- LOAD MODEL & VECTORIZER --------------------
try:
    with open("sarcasm_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

except FileNotFoundError:
    st.error("❌ Model or vectorizer file not found.")
    st.stop()

# -------------------- TEXT CLEANING --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)                 # remove text in []
    text = re.sub(r'\S*\d\S*', '', text)                # remove words with digits
    text = re.sub(r'(https|http)?:\/\/\S+', '', text)   # remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()            # remove extra spaces
    return text

# -------------------- USER INPUT --------------------
user_input = st.text_area(
    "✍️ Enter text",
    height=150,
    placeholder="Example: Oh great, another exam tomorrow..."
)

# -------------------- PREDICTION --------------------
if st.button("🔍 Analyze Sarcasm"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_text = tfidf.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]

        st.subheader("🧠 Prediction Result")

        if prediction == 1:
            st.error("😏 **Sarcastic**")
        else:
            st.success("🙂 **Not Sarcastic**")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Sarcasm Analysis using Machine Learning & Streamlit")
