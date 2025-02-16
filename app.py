import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import speech_recognition as sr
from pydub import AudioSegment
import tempfile

# Set page configuration
st.set_page_config(
    page_title="Petition Classification System",
    page_icon="📝",
    layout="wide"
)

# Function to generate dataset
@st.cache_data
def generate_petition_dataset(num_samples=1000):
    """
    Generate a synthetic dataset for petitions with realistic content
    """
    # Define base components for generating petitions
    issues = [
        "road maintenance", "public transportation", "waste management",
        "park facilities", "street lighting", "traffic signals",
        "sidewalk repairs", "bike lanes", "public safety", "noise pollution",
        "air quality", "water supply", "school facilities", "healthcare access",
        "senior services", "youth programs", "affordable housing",
        "small business support", "parking facilities", "community centers"
    ]
    
    locations = [
        "downtown", "north district", "south district", "west side",
        "east side", "central area", "suburban area", "business district",
        "residential area", "industrial zone"
    ]
    
    actions = [
        "improve", "implement", "upgrade", "maintain", "develop",
        "establish", "enhance", "renovate", "expand", "create"
    ]
    
    reasons = [
        "for better quality of life", "to ensure public safety",
        "for community development", "to meet growing demands",
        "to address resident concerns", "for environmental protection",
        "to support local economy", "for sustainable development",
        "to improve accessibility", "for future generations"
    ]
    
    categories = {
        "Infrastructure": 0.25,
        "Transportation": 0.15,
        "Environment": 0.12,
        "Public Safety": 0.13,
        "Education": 0.10,
        "Healthcare": 0.10,
        "Community Services": 0.08,
        "Economic Development": 0.07
    }
    
    departments = {
        "Infrastructure": ["Public Works", "City Planning"],
        "Transportation": ["Transportation", "Traffic Management"],
        "Environment": ["Environmental Protection", "Parks and Recreation"],
        "Public Safety": ["Public Safety", "Emergency Services"],
        "Education": ["Education", "Youth Services"],
        "Healthcare": ["Health Services", "Public Health"],
        "Community Services": ["Community Development", "Social Services"],
        "Economic Development": ["Economic Development", "Business Relations"]
    }
    
    data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for _ in range(num_samples):
        issue = np.random.choice(issues)
        location = np.random.choice(locations)
        action = np.random.choice(actions)
        reason = np.random.choice(reasons)
        
        petition_text = f"Request to {action} {issue} in the {location} {reason}"
        category = np.random.choice(list(categories.keys()), p=list(categories.values()))
        department = np.random.choice(departments[category])
        
        submission_date = start_date + timedelta(days=np.random.randint(0, 365))
        days_since_submission = (datetime.now() - submission_date).days
        
        if days_since_submission > 30:
            status = np.random.choice(["Resolved", "Rejected", "In Progress"], p=[0.6, 0.1, 0.3])
        else:
            status = np.random.choice(["Under Review", "In Progress"], p=[0.7, 0.3])
        
        if "safety" in petition_text.lower() or "emergency" in petition_text.lower():
            priority = "High"
        elif "improve" in petition_text.lower() or "upgrade" in petition_text.lower():
            priority = "Medium"
        else:
            priority = "Low"
        
        first_names = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Emily"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
        
        name = f"{np.random.choice(first_names)} {np.random.choice(last_names)}"
        email = f"{name.lower().replace(' ', '.')}@example.com"
        
        data.append({
            "name": name,
            "email": email,
            "petition": petition_text,
            "category": category,
            "department": department,
            "status": status,
            "priority": priority,
            "created_at": submission_date,
            "votes": np.random.randint(0, 1000)
        })
    
    return pd.DataFrame(data)

def convert_audio_to_text(audio_file):
    """
    Convert uploaded audio file to text using speech recognition
    """
    # Create a temporary file to save the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_path = tmp_file.name

    # Initialize the recognizer
    recognizer = sr.Recognizer()
    
    # Convert audio file to wav format if needed
    if audio_file.name.lower().endswith('.mp3'):
        audio = AudioSegment.from_mp3(tmp_path)
        wav_path = tmp_path.replace('.wav', '_converted.wav')
        audio.export(wav_path, format='wav')
    else:
        wav_path = tmp_path

    # Perform speech recognition
    try:
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        raise Exception(f"Error in speech recognition: {str(e)}")
    finally:
        # Clean up temporary files
        os.unlink(tmp_path)
        if 'wav_path' in locals() and wav_path != tmp_path:
            os.unlink(wav_path)

def train_model(df):
    """
    Train the LSTM model and return training history
    """
    # Prepare text data
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['petition'])
    sequences = tokenizer.texts_to_sequences(df['petition'])
    
    max_length = 100
    X = pad_sequences(sequences, maxlen=max_length)
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['category'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Sequential([
        Embedding(5000, 128, input_length=max_length),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Save model and preprocessing objects
    os.makedirs('models', exist_ok=True)
    model.save("models/petition_classifier.h5")
    joblib.dump(tokenizer, "models/tokenizer.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")
    
    # Evaluate on test set
    loss, accuracy = model.evaluate(X_test, y_test)
    
    return history, accuracy

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
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dataset Generation", "Model Training", "Make Predictions", "Analytics"])
    
    if page == "Dataset Generation":
        st.header("Generate Synthetic Petition Dataset")
        num_samples = st.slider("Number of samples to generate", 100, 5000, 1000)
        
        if st.button("Generate Dataset"):
            with st.spinner("Generating dataset..."):
                df = generate_petition_dataset(num_samples)
                st.session_state['df'] = df
                st.success(f"Generated {num_samples} petition records!")
                
                st.subheader("Sample of Generated Data")
                st.dataframe(df.head())
                
                # Display basic statistics
                st.subheader("Dataset Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Petitions", len(df))
                    
                with col2:
                    st.metric("Active Petitions", len(df[df['status'].isin(['Under Review', 'In Progress'])]))
                    
                with col3:
                    st.metric("Resolved Petitions", len(df[df['status'] == 'Resolved']))
    
    elif page == "Model Training":
        st.header("Train Classification Model")
        
        if 'df' not in st.session_state:
            st.warning("Please generate a dataset first!")
            return
            
        if st.button("Train Model"):
            with st.spinner("Training model... This may take a few minutes."):
                history, accuracy = train_model(st.session_state['df'])
                st.success(f"Model trained successfully! Test accuracy: {accuracy:.2%}")
                
                # Plot training history
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=history.history['accuracy'], name='Training Accuracy'))
                fig.add_trace(go.Scatter(y=history.history['val_accuracy'], name='Validation Accuracy'))
                fig.update_layout(title='Training History', xaxis_title='Epoch', yaxis_title='Accuracy')
                st.plotly_chart(fig)
    
    elif page == "Make Predictions":
        st.header("Predict Petition Category")
        
        model, tokenizer, label_encoder = load_trained_model()
        
        if model is None:
            st.warning("Please train the model first!")
            return
            
        # Add tabs for text input and audio upload
        input_type = st.tabs(["Text Input", "Audio Upload"])
        
        with input_type[0]:
            petition_text = st.text_area("Enter petition text:", height=100)
            
            if st.button("Predict Category", key="text_predict"):
                if petition_text:
                    with st.spinner("Analyzing petition..."):
                        category, confidence = predict_category(petition_text, model, tokenizer, label_encoder)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Predicted Category", category)
                        with col2:
                            st.metric("Confidence", f"{confidence:.2%}")
                else:
                    st.warning("Please enter petition text!")
        
        with input_type[1]:
            st.write("Upload an audio file with your petition (WAV or MP3 format)")
            audio_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
            
            if audio_file is not None:
                if st.button("Transcribe and Predict", key="audio_predict"):
                    try:
                        with st.spinner("Transcribing audio..."):
                            transcribed_text = convert_audio_to_text(audio_file)
                            st.success("Audio transcribed successfully!")
                            st.subheader("Transcribed Text:")
                            st.write(transcribed_text)
                            
                            with st.spinner("Analyzing petition..."):
                                category, confidence = predict_category(transcribed_text, model, tokenizer, label_encoder)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Predicted Category", category)
                                with col2:
                                    st.metric("Confidence", f"{confidence:.2%}")
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
    
    elif page == "Analytics":
        st.header("Petition Analytics")
        
        if 'df' not in st.session_state:
            st.warning("Please generate a dataset first!")
            return
            
        df = st.session_state['df']
        
        # Category Distribution
        st.subheader("Category Distribution")
        fig_category = px.pie(df, names='category', title='Petition Categories')
        st.plotly_chart(fig_category)
        
        # Status Distribution
        st.subheader("Status Distribution")
        fig_status = px.bar(df['status'].value_counts(), title='Petition Status')
        st.plotly_chart(fig_status)
        
        # Priority Distribution by Category
        st.subheader("Priority Distribution by Category")
        fig_priority = px.bar(df, x='category', color='priority', title='Priority Distribution by Category')
        st.plotly_chart(fig_priority)
        
        # Votes Analysis
        st.subheader("Votes Analysis")
        fig_votes = px.box(df, x='category', y='votes', title='Vote Distribution by Category')
        st.plotly_chart(fig_votes)

if __name__ == "__main__":
    main()