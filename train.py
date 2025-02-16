import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import joblib
import os

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
    
    # Define categories and their weights
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
    
    # Define departments
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
    
    # Generate synthetic data
    data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for _ in range(num_samples):
        # Generate petition text
        issue = np.random.choice(issues)
        location = np.random.choice(locations)
        action = np.random.choice(actions)
        reason = np.random.choice(reasons)
        
        petition_text = f"Request to {action} {issue} in the {location} {reason}"
        
        # Select category based on weights
        category = np.random.choice(
            list(categories.keys()),
            p=list(categories.values())
        )
        
        # Select department based on category
        department = np.random.choice(departments[category])
        
        # Generate status based on date
        submission_date = start_date + timedelta(
            days=np.random.randint(0, 365)
        )
        days_since_submission = (datetime.now() - submission_date).days
        
        if days_since_submission > 30:
            status = np.random.choice(
                ["Resolved", "Rejected", "In Progress"],
                p=[0.6, 0.1, 0.3]
            )
        else:
            status = np.random.choice(
                ["Under Review", "In Progress"],
                p=[0.7, 0.3]
            )
        
        # Determine priority
        if "safety" in petition_text.lower() or "emergency" in petition_text.lower():
            priority = "High"
        elif "improve" in petition_text.lower() or "upgrade" in petition_text.lower():
            priority = "Medium"
        else:
            priority = "Low"
        
        # Generate petitioner details
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
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def train_petition_model(df, save_path="models"):
    """
    Train an LSTM model for petition classification
    """
    # Create models directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare text data
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['petition'])
    sequences = tokenizer.texts_to_sequences(df['petition'])
    
    # Pad sequences
    max_length = 100  # Limit to 100 words per petition
    X = pad_sequences(sequences, maxlen=max_length)
    
    # Prepare labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['category'])
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build model
    model = Sequential([
        Embedding(5000, 128, input_length=max_length),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    # Save model and preprocessing objects
    model.save(f"{save_path}/petition_classifier.h5")
    joblib.dump(tokenizer, f"{save_path}/tokenizer.pkl")
    joblib.dump(label_encoder, f"{save_path}/label_encoder.pkl")
    
    return model, tokenizer, label_encoder, history

def predict_category(text, model, tokenizer, label_encoder):
    """
    Predict category for a new petition text
    """
    # Prepare text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)
    
    # Make prediction
    prediction = model.predict(padded)
    category = label_encoder.inverse_transform([prediction.argmax()])[0]
    confidence = float(prediction.max())
    
    return category, confidence

if __name__ == "__main__":
    # Generate dataset
    print("Generating synthetic petition dataset...")
    df = generate_petition_dataset(1000)
    
    # Train model
    print("\nTraining model...")
    model, tokenizer, label_encoder, history = train_petition_model(df)
    
    # Example prediction
    print("\nTesting model with sample prediction...")
    sample_text = "Request to improve road maintenance in the downtown for better quality of life"
    category, confidence = predict_category(sample_text, model, tokenizer, label_encoder)
    print(f"\nSample petition: {sample_text}")
    print(f"Predicted category: {category}")
    print(f"Confidence: {confidence:.2%}")
    
    print("\nSaved files:")
    print("- models/petition_classifier.h5")
    print("- models/tokenizer.pkl")
    print("- models/label_encoder.pkl")