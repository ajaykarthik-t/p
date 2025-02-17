import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam

# Load real dataset
def load_real_dataset(csv_file):
    df = pd.read_csv(csv_file)

    # Ensure necessary columns exist
    required_columns = ["petition", "category"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Drop missing values
    df = df.dropna(subset=["petition", "category"])
    print(f"✅ Loaded dataset: {df.shape[0]} records")
    return df

# Train a text classification model
def train_petition_model(df):
    # Tokenization
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["petition"])
    sequences = tokenizer.texts_to_sequences(df["petition"])
    max_len = max(len(seq) for seq in sequences)
    X = pad_sequences(sequences, maxlen=max_len, padding="post")

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["category"])
    y = tf.keras.utils.to_categorical(y, num_classes=len(label_encoder.classes_))

    # Define model
    model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=max_len),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(len(label_encoder.classes_), activation="softmax")
    ])

    # Fix optimizer issue
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True
    )
    optimizer = Adam(learning_rate=lr_schedule)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Train the model
    history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    return model, tokenizer, label_encoder, history

# Main execution
if __name__ == "__main__":
    csv_file = "petition_data_updated.csv"  # Change this to your actual dataset file
    print("\nLoading real dataset...\n")
    df = load_real_dataset(csv_file)

    print("\nTraining model...\n")
    model, tokenizer, label_encoder, history = train_petition_model(df)

    print("\nTraining completed! Model is ready.\n")

    # Save the trained model
    model.save("petition_model.h5")
    print("\nModel saved as 'petition_model.h5'\n")
