

import streamlit as st
import pickle
import sklearn

# Load the model
with open('language_detection_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit UI
st.title("🌍 Language Detection App")

user_input = st.text_area("Enter text here:", "")

if st.button("Detect Language"):
    if user_input.strip() == "":
        st.warning("Please enter some text to detect.")
    else:
        # Transform input
        input_transformed = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(input_transformed)
        
        # Show result
        st.success(f"Detected Language: {prediction[0]}")
