import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit UI
st.title("📱 SMS Spam Detection")
st.write("Enter an SMS message below and detect if it's Spam or Ham.")

user_input = st.text_area("Type your message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        vect_msg = vectorizer.transform([user_input])
        prediction = model.predict(vect_msg)[0]

        if prediction == 1:
            st.error("🚨 It's SPAM!")
        else:
            st.success("✅ It's HAM (not spam).")
