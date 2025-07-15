import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("📩 SMS Spam Classifier")

# User input
message = st.text_area("Enter your SMS message here:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        data = vectorizer.transform([message])
        prediction = model.predict(data)[0]

        if prediction == 1:
            st.error("🚫 This message is likely **SPAM**.")
        else:
            st.success("✅ This message is **Not Spam**.")
