import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import os
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
from datetime import datetime
import csv

# Set page configuration
st.set_page_config(
    page_title="Petition Classification System",
    page_icon="📝",
    layout="wide"
)

# Define the CSV file path
SUBMISSIONS_FILE = "petition_submissions.csv"

def ensure_csv_exists():
    """
    Create the CSV file if it doesn't exist
    """
    if not os.path.exists(SUBMISSIONS_FILE):
        with open(SUBMISSIONS_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Petition Text', 'Category', 'Confidence', 'Priority', 'Status'])

def save_submission(petition_text, category, confidence, priority):
    """
    Save the petition submission to CSV file
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Determine priority based on keywords
    if "safety" in petition_text.lower() or "emergency" in petition_text.lower():
        priority = "High"
    elif "improve" in petition_text.lower() or "upgrade" in petition_text.lower():
        priority = "Medium"
    else:
        priority = "Low"
    
    with open(SUBMISSIONS_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, petition_text, category, confidence, priority, "Pending"])

def load_and_sort_submissions():
    """
    Load submissions from CSV and sort by priority
    """
    if not os.path.exists(SUBMISSIONS_FILE):
        return pd.DataFrame()
    
    df = pd.read_csv(SUBMISSIONS_FILE)
    
    # Define priority order
    priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
    
    # Sort by priority and then by timestamp
    df['Priority_Order'] = df['Priority'].map(priority_order)
    df = df.sort_values(['Priority_Order', 'Timestamp'], ascending=[True, False])
    df = df.drop('Priority_Order', axis=1)
    
    return df

def convert_audio_to_text(audio_file):
    """
    Convert uploaded audio file to text using speech recognition
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_path = tmp_file.name

    recognizer = sr.Recognizer()
    
    if audio_file.name.lower().endswith('.mp3'):
        audio = AudioSegment.from_mp3(tmp_path)
        wav_path = tmp_path.replace('.wav', '_converted.wav')
        audio.export(wav_path, format='wav')
    else:
        wav_path = tmp_path

    try:
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        raise Exception(f"Error in speech recognition: {str(e)}")
    finally:
        os.unlink(tmp_path)
        if 'wav_path' in locals() and wav_path != tmp_path:
            os.unlink(wav_path)

def predict_category(text, model, tokenizer, label_encoder):
    """
    Predict category for a new petition
    """
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded)
    category = label_encoder.inverse_transform([prediction.argmax()])[0]
    confidence = float(prediction.max())
    return category, confidence

def load_trained_model():
    """
    Load the trained model and preprocessing objects
    """
    if os.path.exists("models/petition_classifier.h5"):
        model = load_model("models/petition_classifier.h5")
        tokenizer = joblib.load("models/tokenizer.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        return model, tokenizer, label_encoder
    return None, None, None

def main():
    st.title("🏛️ Petition Classification System")
    
    # Ensure CSV file exists
    ensure_csv_exists()
    
    model, tokenizer, label_encoder = load_trained_model()
    
    if model is None:
        st.error("No trained model found! Please ensure the model files are present in the 'models' directory.")
        return
    
    # Add tabs for text input, audio upload, and submissions
    tabs = st.tabs(["Text Input", "Audio Upload", "View Submissions"])
    
    with tabs[0]:
        petition_text = st.text_area("Enter petition text:", height=100)
        
        if st.button("Submit Petition", key="text_predict"):
            if petition_text:
                with st.spinner("Analyzing petition..."):
                    category, confidence = predict_category(petition_text, model, tokenizer, label_encoder)
                    
                    # Determine priority
                    if "safety" in petition_text.lower() or "emergency" in petition_text.lower():
                        priority = "High"
                    elif "improve" in petition_text.lower() or "upgrade" in petition_text.lower():
                        priority = "Medium"
                    else:
                        priority = "Low"
                    
                    # Save submission
                    save_submission(petition_text, category, confidence, priority)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Category", category)
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    with col3:
                        st.metric("Priority", priority)
                    
                    st.success("Petition submitted successfully!")
            else:
                st.warning("Please enter petition text!")
    
    with tabs[1]:
        st.write("Upload an audio file with your petition (WAV or MP3 format)")
        audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
        
        if audio_file is not None:
            if st.button("Transcribe and Submit", key="audio_predict"):
                try:
                    with st.spinner("Transcribing audio..."):
                        transcribed_text = convert_audio_to_text(audio_file)
                        st.success("Audio transcribed successfully!")
                        st.subheader("Transcribed Text:")
                        st.write(transcribed_text)
                        
                        with st.spinner("Analyzing petition..."):
                            category, confidence = predict_category(transcribed_text, model, tokenizer, label_encoder)
                            
                            # Determine priority
                            if "safety" in transcribed_text.lower() or "emergency" in transcribed_text.lower():
                                priority = "High"
                            elif "improve" in transcribed_text.lower() or "upgrade" in transcribed_text.lower():
                                priority = "Medium"
                            else:
                                priority = "Low"
                            
                            # Save submission
                            save_submission(transcribed_text, category, confidence, priority)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Predicted Category", category)
                            with col2:
                                st.metric("Confidence", f"{confidence:.2%}")
                            with col3:
                                st.metric("Priority", priority)
                            
                            st.success("Petition submitted successfully!")
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
    
    with tabs[2]:
        st.subheader("Submitted Petitions")
        
        # Load and display submissions
        df = load_and_sort_submissions()
        if not df.empty:
            # Add filters
            st.sidebar.header("Filters")
            priority_filter = st.sidebar.multiselect(
                "Filter by Priority",
                options=['High', 'Medium', 'Low'],
                default=['High', 'Medium', 'Low']
            )
            
            # Apply filters
            filtered_df = df[df['Priority'].isin(priority_filter)]
            
            # Display the filtered dataframe
            st.dataframe(filtered_df, height=400)
            
            # Display statistics
            st.subheader("Submission Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Submissions", len(df))
            with col2:
                st.metric("High Priority", len(df[df['Priority'] == 'High']))
            with col3:
                st.metric("Pending Review", len(df[df['Status'] == 'Pending']))
        else:
            st.info("No petitions submitted yet.")

if __name__ == "__main__":
    main()