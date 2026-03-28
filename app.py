import streamlit as st
import pickle
import nltk


nltk.download('stopwords')
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📧 Fake Email Detection System")

email = st.text_area("Enter Email Content")

if st.button("Predict"):
    if email.strip() == "":
        st.warning("Please enter email text")
    else:
        data = vectorizer.transform([email])
        prediction = model.predict(data)[0]

        if prediction == 1:
            st.error("🚨 Spam Email Detected")
        else:
            st.success("✅ Safe Email")
