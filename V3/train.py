import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("category_urgency_dataset.csv")

# Features and targets
X = df["Sample Petition"]
y_category = df["Category"]
y_urgency = df["Urgency Level"]

# Create a pipeline for Category classification using keyword features
category_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=200, random_state=42))
])

# Create a pipeline for Urgency classification using keyword features
urgency_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=200, random_state=42))
])

# Split the data (using the same splits for simplicity)
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X, y_category, test_size=0.2, random_state=42)
X_train_urg, X_test_urg, y_train_urg, y_test_urg = train_test_split(X, y_urgency, test_size=0.2, random_state=42)

# Train the models
category_pipeline.fit(X_train_cat, y_train_cat)
urgency_pipeline.fit(X_train_urg, y_train_urg)

# Save the models
joblib.dump(category_pipeline, "category_model.pkl")
joblib.dump(urgency_pipeline, "urgency_model.pkl")

print("Training complete! Models saved as 'category_model.pkl' and 'urgency_model.pkl'.")
