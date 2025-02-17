import streamlit as st
import joblib

# Load the trained models
category_model = joblib.load("category_model.pkl")
urgency_model = joblib.load("urgency_model.pkl")

st.title("Text Classification App")
st.write("Enter a text phrase (from the Sample Petition) to predict its Category and Urgency Level.")

# Text input field
text_input = st.text_area("Enter text:")

if st.button("Classify"):
    if text_input.strip():
        # Make predictions
        predicted_category = category_model.predict([text_input])[0]
        predicted_urgency = urgency_model.predict([text_input])[0]

        # Display the predictions
        st.subheader("Predictions:")
        st.write(f"**Category:** {predicted_category}")
        st.write(f"**Urgency Level:** {predicted_urgency}")
    else:
        st.warning("Please enter some text.")
