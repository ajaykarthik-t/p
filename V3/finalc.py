import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import os

# Load the trained models
category_model = joblib.load("category_model.pkl")
urgency_model = joblib.load("urgency_model.pkl")

# Configure wide layout
st.set_page_config(layout="wide")

# Initialize session state for submissions
if 'submissions' not in st.session_state:
    st.session_state.submissions = []

# Function to transcribe audio
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

# Main app
st.title("Petition Classification System")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Text Input", "Audio Upload", "View Submissions"])

with tab1:
    st.header("Text Submission")
    text_input = st.text_area("Enter petition text:", height=200)
    
    if st.button("Submit Text"):
        if text_input.strip():
            # Make predictions
            predicted_category = category_model.predict([text_input])[0]
            confidence = max(category_model.predict_proba([text_input])[0])  # Get max probability
            predicted_urgency = urgency_model.predict([text_input])[0]
            
            # Store submission
            st.session_state.submissions.append({
                "text": text_input,
                "category": predicted_category,
                "confidence": confidence,
                "priority": predicted_urgency,
                "date": datetime.now(),
                "type": "text"
            })
            
            # Display results
            st.success("Petition submitted successfully!")
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Category", predicted_category)
            col2.metric("Confidence", f"{confidence:.2%}")
            col3.metric("Priority Level", predicted_urgency)
        else:
            st.warning("Please enter some text before submitting")

with tab2:
    st.header("Audio Submission")
    audio_file = st.file_uploader("Upload WAV/MP3 file", type=["wav", "mp3"])
    
    if st.button("Transcribe and Classify"):
        if audio_file is not None:
            try:
                # Transcribe audio
                transcribed_text = convert_audio_to_text(audio_file)
                
                # Make predictions
                predicted_category = category_model.predict([transcribed_text])[0]
                confidence = max(category_model.predict_proba([transcribed_text])[0])
                predicted_urgency = urgency_model.predict([transcribed_text])[0]
                
                # Store submission
                st.session_state.submissions.append({
                    "text": transcribed_text,  # Include transcribed text
                    "category": predicted_category,
                    "confidence": confidence,
                    "priority": predicted_urgency,
                    "date": datetime.now(),
                    "type": "audio"
                })
                
                # Display results
                st.success("Audio submitted successfully!")
                st.subheader("Transcribed Text:")
                st.write(transcribed_text)  # Display transcribed text in the UI
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted Category", predicted_category)
                col2.metric("Confidence", f"{confidence:.2%}")
                col3.metric("Priority Level", predicted_urgency)
            except Exception as e:
                st.error(f"Transcription failed: {str(e)}")
        else:
            st.warning("Please upload an audio file first")

with tab3:
    st.header("View Submissions")
    
    # Sidebar filters
    with st.sidebar:
        st.subheader("Filter Submissions")
        selected_priorities = st.multiselect(
            "Select priorities to show:",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
    
    # Filter submissions
    filtered_submissions = [sub for sub in st.session_state.submissions 
                          if sub['priority'] in selected_priorities]
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Submissions", len(st.session_state.submissions))
    col2.metric("High Priority Petitions", 
               len([s for s in st.session_state.submissions if s['priority'] == "High"]))
    col3.metric("Pending Review Cases", 
               len([s for s in st.session_state.submissions if s['priority'] == "High"]))
    
    # Display table
    if filtered_submissions:
        # Convert to DataFrame for better display
        df = pd.DataFrame(filtered_submissions)[['date', 'category', 'priority', 'confidence', 'type', 'text']]
        st.dataframe(df.sort_values('priority', ascending=False), use_container_width=True)
    else:
        st.info("No submissions matching selected filters")

# Error handling message placeholder
if 'error' in st.session_state:
    st.error(st.session_state.error)
    del st.session_state.error