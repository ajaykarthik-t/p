# train.py

import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Define all classes and functions at the module level

# Define the wrapper class outside the function so it can be pickled
class SeparateModelsWrapper:
    def __init__(self, dept_model, urgency_model):
        self.dept_model = dept_model
        self.urgency_model = urgency_model
    
    def predict(self, X):
        dept_pred = self.dept_model.predict(X)
        urgency_pred = self.urgency_model.predict(X)
        # Return predictions as a list of [department, urgency] for each input
        return [(dept, urg) for dept, urg in zip(dept_pred, urgency_pred)]

# Define rule-based classifier at module level so it can be pickled
class RuleBasedClassifier:
    def __init__(self, department_keywords):
        self.department_keywords = department_keywords
    
    def classify(self, text):
        text = text.lower()
        scores = {}
        
        for dept, keywords in self.department_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[dept] = score
        
        # Get department with highest score
        if all(score == 0 for score in scores.values()):
            # If no keywords matched, return most common department (Education in this case)
            return "Education"
        else:
            return max(scores, key=scores.get)

# Define petition classifier at module level
class PetitionClassifier:
    def __init__(self, dept_model, urgency_model, combined_model, rule_based_model, departments):
        self.dept_model = dept_model
        self.urgency_model = urgency_model
        self.combined_model = combined_model
        self.rule_based_model = rule_based_model
        self.departments = departments
    
    def classify(self, text):
        try:
            # Try using the combined model first
            prediction = self.combined_model.predict([text])[0]
            department = prediction[0]
            urgency_code = prediction[1]
            urgency = "High" if urgency_code == 1 else "Normal"
        except:
            try:
                # Fall back to individual models
                department = self.dept_model.predict([text])[0]
                urgency_code = self.urgency_model.predict([text])[0]
                urgency = "High" if urgency_code == 1 else "Normal"
            except:
                # Fall back to rule-based classifier
                department = self.rule_based_model.classify(text)
                
                # Determine urgency based on keywords
                urgency_keywords = ['urgent', 'emergency', 'critical', 'severe', 'immediately', 
                                  'dangerous', 'safety', 'hazard', 'life-threatening']
                has_urgency = any(keyword in text.lower() for keyword in urgency_keywords)
                urgency = "High" if has_urgency else "Normal"
        
        return department, urgency

# Function to load and preprocess data
def load_data():
    # Check if training data exists, if not generate it
    if not os.path.exists('data/training_petitions.csv'):
        print("Training data not found. Generating synthetic data...")
        import dataset
        df = dataset.generate_dataset(500)
    else:
        print("Loading existing training data...")
        df = pd.read_csv('data/training_petitions.csv')
    
    # Preprocess data
    df['department_code'] = df['department'].astype('category').cat.codes
    df['urgency_code'] = (df['urgency'] == 'High').astype(int)
    
    return df

# Function to train department classification model
def train_department_classifier(X_train, y_train, X_test, y_test):
    print("Training department classification model...")
    
    # Create pipeline with TF-IDF and Random Forest
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Department Classification Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    with open('models/department_classifier.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    return pipeline

# Function to train urgency detection model
def train_urgency_detector(X_train, y_train, X_test, y_test):
    print("Training urgency detection model...")
    
    # Create pipeline with TF-IDF and Random Forest
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Urgency Detection Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'High']))
    
    # Save model
    with open('models/urgency_detector.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    return pipeline

# Function to train two separate models (avoiding the combined model due to data type issues)
def train_separate_models(X_train, y_train_dept, y_train_urgency, X_test, y_test_dept, y_test_urgency):
    print("Training separate models instead of combined model due to data type compatibility issues...")
    
    # We'll use the individual models that were already trained and create a wrapper
    # No need to train a combined model which is causing issues with mixed data types
    
    # Create the wrapper using the models we already trained
    dept_model = pickle.load(open('models/department_classifier.pkl', 'rb'))
    urgency_model = pickle.load(open('models/urgency_detector.pkl', 'rb'))
    wrapper = SeparateModelsWrapper(dept_model, urgency_model)
    
    # Evaluate the wrapper
    y_pred = wrapper.predict(X_test)
    y_pred_dept = [pred[0] for pred in y_pred]
    y_pred_urgency = [pred[1] for pred in y_pred]
    
    dept_accuracy = accuracy_score(y_test_dept, y_pred_dept)
    urgency_accuracy = accuracy_score(y_test_urgency, y_pred_urgency)
    
    print(f"Separate Models Department Accuracy: {dept_accuracy:.4f}")
    print(f"Separate Models Urgency Accuracy: {urgency_accuracy:.4f}")
    
    # Save the wrapper
    with open('models/combined_classifier.pkl', 'wb') as f:
        pickle.dump(wrapper, f)
    
    return wrapper

# Function to create a rule-based classifier from department keywords
def create_rule_based_classifier():
    print("Creating rule-based classifier from keywords...")
    
    # Load department keywords
    with open('data/department_keywords.json', 'r') as f:
        department_keywords = json.load(f)
    
    # Create an instance of the rule-based classifier
    rule_based_classifier = RuleBasedClassifier(department_keywords)
    
    # Save classifier using pickle
    with open('models/rule_based_classifier.pkl', 'wb') as f:
        pickle.dump(rule_based_classifier, f)
    
    print("Rule-based classifier created and saved.")
    
    return rule_based_classifier

# Main function to train all models
def train_all_models():
    # Load data
    df = load_data()
    
    # Split features and targets
    X = df['description']
    y_dept = df['department']
    y_urgency = df['urgency_code']
    
    # Split into train and test sets
    X_train, X_test, y_train_dept, y_test_dept, y_train_urgency, y_test_urgency = train_test_split(
        X, y_dept, y_urgency, test_size=0.2, random_state=42
    )
    
    # Train department classification model
    dept_model = train_department_classifier(X_train, y_train_dept, X_test, y_test_dept)
    
    # Train urgency detection model
    urgency_model = train_urgency_detector(X_train, y_train_urgency, X_test, y_test_urgency)
    
    # Train separate models instead of the combined model
    combined_model = train_separate_models(X_train, y_train_dept, y_train_urgency, X_test, y_test_dept, y_test_urgency)
    
    # Create rule-based classifier as fallback
    rule_based_model = create_rule_based_classifier()
    
    # Create a PetitionClassifier instance
    departments = df['department'].unique().tolist()
    classifier = PetitionClassifier(
        dept_model=dept_model,
        urgency_model=urgency_model,
        combined_model=combined_model,
        rule_based_model=rule_based_model,
        departments=departments
    )
    
    # Save the wrapper model
    with open('models/petition_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    
    print("\nAll models trained and saved successfully!")
    print("Models saved to: models/")
    
    return classifier

if __name__ == "__main__":
    print("Starting model training for Petition Classification System...")
    classifier = train_all_models()
    
    # Test the classifier on a few examples
    print("\nTesting classifier on example petitions:")
    
    test_petitions = [
        "Our school doesn't have enough computers for students to learn programming effectively.",
        "There's a dangerous pothole on Main Street that has caused multiple accidents.",
        "We have been without clean drinking water for three days now. This is an emergency!",
        "My property tax assessment is incorrect and much higher than my neighbors with similar houses."
    ]
    
    for petition in test_petitions:
        department, urgency = classifier.classify(petition)
        print(f"\nPetition: {petition}")
        print(f"Classified as: {department} department")
        print(f"Urgency: {urgency}")