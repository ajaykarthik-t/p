# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import uuid
import datetime
import time
import json
from pathlib import Path
import base64

# Define classes needed for model loading
class SeparateModelsWrapper:
    def __init__(self, dept_model, urgency_model):
        self.dept_model = dept_model
        self.urgency_model = urgency_model
    
    def predict(self, X):
        dept_pred = self.dept_model.predict(X)
        urgency_pred = self.urgency_model.predict(X)
        # Return predictions as a list of [department, urgency] for each input
        return [(dept, urg) for dept, urg in zip(dept_pred, urgency_pred)]

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

# Set page configuration
st.set_page_config(
    page_title="Petition Classification System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create necessary directories if they don't exist
for dir_name in ['uploads', 'audio', 'data', 'models']:
    os.makedirs(dir_name, exist_ok=True)

# Initialize session state variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
if 'department' not in st.session_state:
    st.session_state.department = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'petitions' not in st.session_state:
    # Load petitions data if it exists, otherwise create empty dataframe
    if os.path.exists('data/petitions.csv'):
        st.session_state.petitions = pd.read_csv('data/petitions.csv')
    else:
        st.session_state.petitions = pd.DataFrame(columns=[
            'id', 'name', 'phone', 'address', 'description', 'files', 
            'department', 'urgency', 'status', 'date_submitted', 'submitted_by'
        ])

# User credentials (hardcoded for demo)
user_credentials = {
    "test_user": "user123"
}

# Admin credentials (hardcoded for demo) - FIXED THE ADMIN CREDENTIALS FORMAT
admin_credentials = {
    "admin_edu": {"password": "edu123", "department": "Education"},
    "admin_health": {"password": "health123", "department": "Health"},
    "admin_trans": {"password": "trans123", "department": "Transport"},
    "admin_tax": {"password": "tax123", "department": "Taxation"},
    "admin_housing": {"password": "housing123", "department": "Housing"},
    "admin_energy": {"password": "energy123", "department": "Energy"},
    "admin_water": {"password": "water123", "department": "Water"},
    "admin_agri": {"password": "agri123", "department": "Agriculture"},
    "admin_infra": {"password": "infra123", "department": "Infrastructure"}
}

# Department list
departments = [
    "Education", "Health", "Transport", "Taxation", 
    "Housing", "Energy", "Water", "Agriculture", "Infrastructure"
]

# Custom CSS with improved readability
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div.stButton > button {
        width: 100%;
        height: 3em;
        font-weight: bold;
    }
    .petition-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: white;
        color: #333333; /* Darker text color for better readability */
        font-size: 16px; /* Larger font size */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1); /* Subtle shadow for depth */
    }
    .high-priority {
        border-left: 5px solid #ff4b4b;
    }
    .normal-priority {
        border-left: 5px solid #4b8bff;
    }
    .resolved {
        background-color: #f0f5ff;
    }
    .status-pill {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: bold;
    }
    .status-inprogress {
        background-color: #ffeecc;
        color: #995500;
    }
    .status-resolved {
        background-color: #ccffcc;
        color: #005500;
    }
    .department-pill {
        background-color: #e6e6ff;
        color: #0000aa;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: bold;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    p {
        font-size: 16px; /* Larger font size for better readability */
        line-height: 1.5; /* Improved line spacing */
    }
    h1, h2, h3 {
        color: #1E3A8A; /* Navy blue headings */
    }
    .petition-description {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        max-height: 200px;
        overflow-y: auto;
    }
    .admin-table {
        margin-top: 10px;
        margin-bottom: 20px;
        border-collapse: separate;
        border-spacing: 0;
        width: 100%;
        border: 1px solid #e6e6e6;
        border-radius: 5px;
        overflow: hidden;
    }
    .admin-table th {
        background-color: #4a6fa5;
        color: white;
        padding: 12px 15px;
        text-align: left;
        font-size: 16px;
    }
    .admin-table td {
        padding: 10px 15px;
        border-bottom: 1px solid #e6e6e6;
        font-size: 15px;
    }
    .admin-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .admin-table tr:hover {
        background-color: #f1f1f1;
    }
    .admin-table .code-cell {
        font-family: monospace;
        background-color: #f0f0f0;
        padding: 2px 6px;
        border-radius: 3px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Function to save petitions to CSV
def save_petitions():
    st.session_state.petitions.to_csv('data/petitions.csv', index=False)

# Display images using Streamlit's native image component
def display_images(file_paths_str, st_container):
    """Display images using Streamlit's native image component instead of HTML"""
    # Check if file_paths_str is NaN or not a string
    if not isinstance(file_paths_str, str) or pd.isna(file_paths_str):
        return  # Return if no files
    
    if not file_paths_str:  # Check for empty string
        return
    
    file_paths = file_paths_str.split(',')
    
    # Check if there are image files to display
    image_files = [path for path in file_paths if path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')) and os.path.exists(path)]
    
    if image_files:
        st_container.write("**Uploaded Images:**")
        
        # Create a columns layout for multiple images
        num_images = len(image_files)
        if num_images > 0:
            # Limit columns to 3 to avoid too small images
            num_cols = min(3, num_images)
            cols = st_container.columns(num_cols)
            
            # Display each image
            for i, img_path in enumerate(image_files):
                try:
                    # Use Streamlit's native image display
                    cols[i % num_cols].image(img_path, width=200)
                except Exception as e:
                    cols[i % num_cols].error(f"Error loading image")

# AI Classification function using trained models if available
def classify_petition(text):
    model_path = 'models/petition_classifier.pkl'
    
    # Try to use the trained model if it exists
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                classifier = pickle.load(f)
            return classifier.classify(text)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            # Fall back to rule-based classification
            return rule_based_classification(text)
    else:
        # Fall back to rule-based classification
        return rule_based_classification(text)

# Rule-based classification as fallback
def rule_based_classification(text):
    # Load department keywords if available
    keywords_path = 'data/department_keywords.json'
    if os.path.exists(keywords_path):
        try:
            with open(keywords_path, 'r') as f:
                department_keywords = json.load(f)
            
            # Simple rule-based classification
            text = text.lower()
            scores = {}
            
            for dept, keywords in department_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text)
                scores[dept] = score
            
            # Get department with highest score
            if all(score == 0 for score in scores.values()):
                department = np.random.choice(departments)  # Fallback
            else:
                department = max(scores, key=scores.get)
                
            # Determine urgency based on keywords
            urgency_keywords = ['urgent', 'emergency', 'critical', 'severe', 'immediately', 
                            'dangerous', 'safety', 'hazard', 'life-threatening']
            has_urgency = any(keyword in text.lower() for keyword in urgency_keywords)
            urgency = "High" if has_urgency else "Normal"
            
            return department, urgency
        except Exception as e:
            st.warning(f"Error in rule-based classification: {e}")
    
    # Last resort: random classification
    department = np.random.choice(departments)
    urgency = np.random.choice(["High", "Normal"], p=[0.3, 0.7])
    return department, urgency

# Function to simulate audio to text conversion
def audio_to_text(audio_file):
    """
    Convert audio file to text.
    In a production environment, this would use a speech-to-text API like Whisper.
    For this demo, we'll return simulated transcriptions based on file type.
    """
    # Get file extension
    file_extension = audio_file.split('.')[-1].lower()
    
    # Simulate different transcription qualities based on file format
    if file_extension == 'mp3':
        return "This is a simulated transcription from an MP3 file. In a real system, this would be processed using a speech recognition API like Whisper or DeepSpeech. The petition appears to be about infrastructure improvements in the local community."
    elif file_extension == 'wav':
        return "This is a simulated transcription from a WAV file. The audio quality would typically be better than compressed formats. The petition mentions concerns about public transportation services and road maintenance."
    else:
        return "This is a simulated transcription from an audio file. The petition requests attention to a community issue that requires prompt administrative action."

# Login page
def show_login_page():
    st.markdown("<h1 style='text-align: center;'>Petition Classification and Tracking System</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h2 style='text-align: center;'>Login</h2>", unsafe_allow_html=True)
        
        login_type = st.radio("Select Login Type:", ["User", "Admin"])
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        # Only show department selection for admin, but don't use it for validation
        if login_type == "Admin":
            st.info("Note: For admin login, use the correct username format as shown in the table below.")
        
        login_button = st.button("Login")
        
        if login_button:
            if login_type == "User" and username in user_credentials and user_credentials[username] == password:
                st.session_state.logged_in = True
                st.session_state.user_type = "user"
                st.session_state.username = username
                st.success("Login Successful!")
                time.sleep(1)
                st.rerun()
                
            elif login_type == "Admin" and username in admin_credentials:
                if admin_credentials[username]["password"] == password:
                    department = admin_credentials[username]["department"]
                    st.session_state.logged_in = True
                    st.session_state.user_type = "admin"
                    st.session_state.department = department
                    st.session_state.username = username
                    st.success(f"Login Successful! You are logged in to the {department} department.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Invalid password for admin user '{username}'!")
            else:
                st.error(f"Invalid username or user type! Please check your credentials.")
        
        # Display login instructions
        st.markdown("### Testing Credentials")
        if login_type == "User":
            st.code("Username: test_user\nPassword: user123")
        else:
            st.markdown("### Admin Credentials by Department")
            
            # Create a data frame for admin credentials - much cleaner!
            admin_data = []
            for username, details in admin_credentials.items():
                admin_data.append({
                    "Department": details["department"],
                    "Username": username,
                    "Password": details["password"]
                })
            
            # Convert to DataFrame and display as a styled table
            creds_df = pd.DataFrame(admin_data)
            st.dataframe(creds_df, use_container_width=True, 
                        column_config={
                            "Department": st.column_config.TextColumn("Department"),
                            "Username": st.column_config.TextColumn("Username", help="Admin username"),
                            "Password": st.column_config.TextColumn("Password", help="Admin password")
                        })
            
            # Add helpful note
            st.info("Use the credentials from the table above to log in as an admin.")

# User Dashboard
def show_user_dashboard():
    # Sidebar
    with st.sidebar:
        st.markdown(f"<div class='sidebar-header'>User Portal</div>", unsafe_allow_html=True)
        st.markdown(f"Logged in as: **{st.session_state.username}**")
        
        menu = st.radio("Navigation", ["Submit New Petition", "Track Petitions"])
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_type = None
            st.session_state.username = None
            st.rerun()
    
    # Main content
    st.markdown("<h1>User Portal</h1>", unsafe_allow_html=True)
    
    if menu == "Submit New Petition":
        show_petition_submission_form()
    else:
        show_petition_tracking()

# Petition submission form
def show_petition_submission_form():
    st.markdown("## Submit New Petition")
    
    with st.form("petition_form"):
        name = st.text_input("Full Name")
        phone = st.text_input("Phone Number")
        address = st.text_area("Address")
        
        description = st.text_area("Petition Description", height=150)
        
        files = st.file_uploader("Upload Supporting Documents (PDF, Images)", 
                              type=["pdf", "jpg", "jpeg", "png"], 
                              accept_multiple_files=True)
        
        audio_file = st.file_uploader("Upload Voice Petition (Optional)", type=["mp3", "wav"])
        
        submitted = st.form_submit_button("Submit Petition")
        
        if submitted:
            if not name or not phone or not description:
                st.error("Please fill all required fields.")
            else:
                petition_id = str(uuid.uuid4())[:8]
                
                # Process files
                file_paths = []
                if files:
                    for file in files:
                        file_path = f"uploads/{petition_id}_{file.name}"
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        file_paths.append(file_path)
                
                # Process audio if provided
                audio_text = None
                if audio_file:
                    audio_path = f"audio/{petition_id}_{audio_file.name}"
                    with open(audio_path, "wb") as f:
                        f.write(audio_file.getbuffer())
                    
                    # Convert audio to text (in a real system)
                    audio_text = audio_to_text(audio_path)
                    
                # Combine text and audio transcript for classification
                full_text = description
                if audio_text:
                    full_text += " " + audio_text
                
                # Classify the petition
                department, urgency = classify_petition(full_text)
                
                # Create new petition entry
                new_petition = {
                    'id': petition_id,
                    'name': name,
                    'phone': phone,
                    'address': address,
                    'description': description,
                    'files': ','.join(file_paths) if file_paths else '',
                    'department': department,
                    'urgency': urgency,
                    'status': 'In Progress',
                    'date_submitted': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'submitted_by': st.session_state.username
                }
                
                # Add to dataframe
                st.session_state.petitions = pd.concat([
                    st.session_state.petitions, 
                    pd.DataFrame([new_petition])
                ], ignore_index=True)
                
                # Save to CSV
                save_petitions()
                
                st.success(f"Petition submitted successfully! Your petition ID is: {petition_id}")
                st.info(f"Your petition has been classified to the {department} department.")

# Petition tracking page with improved readability
def show_petition_tracking():
    st.markdown("## Track Your Petitions")
    
    # Filter petitions for current user
    user_petitions = st.session_state.petitions[
        st.session_state.petitions['submitted_by'] == st.session_state.username
    ].sort_values('date_submitted', ascending=False)
    
    if len(user_petitions) == 0:
        st.info("You haven't submitted any petitions yet.")
    else:
        st.markdown(f"### You have {len(user_petitions)} petitions")
        
        for _, petition in user_petitions.iterrows():
            # Create card for each petition with improved readability
            with st.container() as petition_container:
                status_class = "status-resolved" if petition['status'] == "Resolved" else "status-inprogress"
                priority_class = "high-priority" if petition['urgency'] == "High" else "normal-priority"
                resolved_class = "resolved" if petition['status'] == "Resolved" else ""
                
                # Create a more readable petition card
                st.markdown(f"""
                <div class='petition-card {priority_class} {resolved_class}'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <h3 style='margin: 0; font-size: 18px;'>Petition #{petition['id']}</h3>
                        <div>
                            <span class='department-pill'>{petition['department']}</span>
                            <span class='status-pill {status_class}'>{petition['status']}</span>
                        </div>
                    </div>
                    <p><strong>Submitted:</strong> {petition['date_submitted']}</p>
                    <p><strong>Description:</strong></p>
                    <div class='petition-description'>{petition['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display images using Streamlit's native components
                display_images(petition['files'], st)

# Admin Dashboard
def show_admin_dashboard():
    # Sidebar
    with st.sidebar:
        st.markdown(f"<div class='sidebar-header'>Admin Portal</div>", unsafe_allow_html=True)
        st.markdown(f"Logged in as: **{st.session_state.username}**")
        st.markdown(f"Department: **{st.session_state.department}**")
        
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_type = None
            st.session_state.department = None
            st.session_state.username = None
            st.rerun()
    
    # Main content
    st.markdown(f"<h1>{st.session_state.department} Department - Admin Portal</h1>", unsafe_allow_html=True)
    
    # Filter petitions for current department
    dept_petitions = st.session_state.petitions[
        st.session_state.petitions['department'] == st.session_state.department
    ]
    
    # Sort by urgency (High first) and then by date (newest first)
    dept_petitions = dept_petitions.sort_values(
        ['status', 'urgency', 'date_submitted'], 
        ascending=[True, False, False]
    )
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.multiselect(
            "Filter by Status:", 
            options=["In Progress", "Resolved"],
            default=["In Progress", "Resolved"]
        )
    
    with col2:
        urgency_filter = st.multiselect(
            "Filter by Urgency:", 
            options=["High", "Normal"],
            default=["High", "Normal"]
        )
    
    # Apply filters
    filtered_petitions = dept_petitions[
        (dept_petitions['status'].isin(status_filter)) & 
        (dept_petitions['urgency'].isin(urgency_filter))
    ]
    
    if len(filtered_petitions) == 0:
        st.info("No petitions found for your department with the selected filters.")
    else:
        # Display count
        st.markdown(f"### Showing {len(filtered_petitions)} petitions")
        
        for idx, petition in filtered_petitions.iterrows():
            # Create card for each petition with improved readability
            with st.container() as petition_container:
                status_class = "status-resolved" if petition['status'] == "Resolved" else "status-inprogress"
                priority_class = "high-priority" if petition['urgency'] == "High" else "normal-priority"
                resolved_class = "resolved" if petition['status'] == "Resolved" else ""
                
                st.markdown(f"""
                <div class='petition-card {priority_class} {resolved_class}'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <h3 style='margin: 0; font-size: 18px;'>Petition #{petition['id']}</h3>
                        <div>
                            <span class='status-pill {status_class}'>{petition['status']}</span>
                            <span style='margin-left: 10px; font-weight: bold; color: {"red" if petition['urgency'] == "High" else "blue"};'>
                                {petition['urgency']} Priority
                            </span>
                        </div>
                    </div>
                    <p><strong>Submitted:</strong> {petition['date_submitted']}</p>
                    <p><strong>Name:</strong> {petition['name']}</p>
                    <p><strong>Phone:</strong> {petition['phone']}</p>
                    <p><strong>Address:</strong> {petition['address']}</p>
                    <p><strong>Description:</strong></p>
                    <div class='petition-description'>{petition['description']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display images using Streamlit's native components
                display_images(petition['files'], st)
                
                # Action buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if petition['status'] == "In Progress":
                        if st.button(f"Mark as Resolved #{petition['id']}", key=f"resolve_{petition['id']}"):
                            # Update status
                            st.session_state.petitions.loc[idx, 'status'] = "Resolved"
                            save_petitions()
                            st.success("Petition marked as Resolved!")
                            time.sleep(1)
                            st.rerun()
                
                with col2:
                    if petition['status'] == "Resolved":
                        if st.button(f"Mark as In Progress #{petition['id']}", key=f"progress_{petition['id']}"):
                            # Update status
                            st.session_state.petitions.loc[idx, 'status'] = "In Progress"
                            save_petitions()
                            st.success("Petition marked as In Progress!")
                            time.sleep(1)
                            st.rerun()
                
                st.markdown("---")

# Main function
def main():
    if not st.session_state.logged_in:
        show_login_page()
    else:
        if st.session_state.user_type == "user":
            show_user_dashboard()
        else:
            show_admin_dashboard()

if __name__ == "__main__":
    main()