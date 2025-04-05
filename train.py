import streamlit as st
import pymongo
import datetime
import qrcode
from PIL import Image
import io
import base64
import pandas as pd
import bcrypt
import os
from bson import ObjectId

# Initialize MongoDB connection
# For the hardcoded version, we'll simulate the database
# In a real implementation, you would connect to your actual MongoDB instance
# client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client["instagate_db"]

# Simulate collections with hardcoded data
students_data = [
    {
        "_id": ObjectId("60d0fe4f5311236168a109ca"),
        "name": "John Doe",
        "email": "john@example.com",
        "password": "password123",  # Plain text for demo
        "admno": "STU001",
        "phone": "1234567890",
        "department": "Computer Science",
        "hostel": "A Block"
    },
    {
        "_id": ObjectId("60d0fe4f5311236168a109cb"),
        "name": "Jane Smith",
        "email": "jane@example.com",
        "password": "password123",  # Plain text for demo
        "admno": "STU002",
        "phone": "9876543210",
        "department": "Electrical Engineering",
        "hostel": "B Block"
    }
]

parents_data = [
    {
        "_id": ObjectId("60d0fe4f5311236168a109cc"),
        "name": "Robert Doe",
        "email": "robert@example.com",
        "password": "password123",  # Plain text for demo
        "admno": "STU001",
        "phone": "5678901234"
    },
    {
        "_id": ObjectId("60d0fe4f5311236168a109cd"),
        "name": "Sarah Smith",
        "email": "sarah@example.com",
        "password": "password123",  # Plain text for demo
        "admno": "STU002",
        "phone": "6789012345"
    }
]

wardens_data = [
    {
        "_id": ObjectId("60d0fe4f5311236168a109ce"),
        "name": "Prof. Michael Johnson",
        "email": "michael@example.com",
        "password": "password123",  # Plain text for demo
        "hostel": "A Block",
        "phone": "7890123456"
    },
    {
        "_id": ObjectId("60d0fe4f5311236168a109cf"),
        "name": "Prof. Emily Williams",
        "email": "emily@example.com",
        "password": "password123",  # Plain text for demo
        "hostel": "B Block",
        "phone": "8901234567"
    }
]

gatepass_requests_data = [
    {
        "_id": ObjectId("60d0fe4f5311236168a109d0"),
        "student_id": "STU001",
        "student_name": "John Doe",
        "hostel": "A Block",
        "reason": "Going home for weekend",
        "destination": "Home",
        "departure_date": datetime.datetime(2025, 3, 15, 10, 0),
        "return_date": datetime.datetime(2025, 3, 17, 18, 0),
        "parent_approval": True,
        "warden_approval": False,
        "request_date": datetime.datetime(2025, 3, 12, 14, 30),
        "status": "Pending Warden Approval"
    }
]

# Helper functions

def hash_password(password):
    """Hash a password for storing."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(stored_password, provided_password):
    """Verify a stored password against a provided password."""
    # For demo purposes, just compare plain text passwords
    return stored_password == provided_password

def find_user_by_email(email, collection_data):
    """Find a user by email in the specified collection."""
    for user in collection_data:
        if user["email"] == email:
            return user
    return None

def find_student_by_admno(admno):
    """Find a student by admission number."""
    for student in students_data:
        if student["admno"] == admno:
            return student
    return None

def find_parent_by_admno(admno):
    """Find a parent by student admission number."""
    for parent in parents_data:
        if parent["admno"] == admno:
            return parent
    return None

def find_warden_by_hostel(hostel):
    """Find a warden by hostel."""
    for warden in wardens_data:
        if warden["hostel"] == hostel:
            return warden
    return None

def get_student_requests(student_id):
    """Get all gate pass requests for a student."""
    return [req for req in gatepass_requests_data if req["student_id"] == student_id]

def get_parent_requests(admno):
    """Get all gate pass requests for a parent to approve."""
    return [req for req in gatepass_requests_data if req["student_id"] == admno and req["parent_approval"] == False]

def get_warden_requests(hostel):
    """Get all gate pass requests for a warden to approve."""
    return [req for req in gatepass_requests_data if req["hostel"] == hostel and req["parent_approval"] == True and req["warden_approval"] == False]

def create_gatepass_request(student_id, student_name, hostel, reason, destination, departure_date, return_date):
    """Create a new gate pass request."""
    new_request = {
        "_id": ObjectId(),
        "student_id": student_id,
        "student_name": student_name,
        "hostel": hostel,
        "reason": reason,
        "destination": destination,
        "departure_date": departure_date,
        "return_date": return_date,
        "parent_approval": False,
        "warden_approval": False,
        "request_date": datetime.datetime.now(),
        "status": "Pending Parent Approval"
    }
    gatepass_requests_data.append(new_request)
    return new_request

def approve_request_parent(request_id):
    """Approve a gate pass request as a parent."""
    for req in gatepass_requests_data:
        if str(req["_id"]) == request_id:
            req["parent_approval"] = True
            req["status"] = "Pending Warden Approval"
            return True
    return False

def approve_request_warden(request_id):
    """Approve a gate pass request as a warden."""
    for req in gatepass_requests_data:
        if str(req["_id"]) == request_id:
            req["warden_approval"] = True
            req["status"] = "Approved"
            return True
    return False

def generate_qr_code(data):
    """Generate a QR code for the gate pass."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save image to a bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    
    # Encode image to base64
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

def generate_gate_pass_pdf(request):
    """Generate a gate pass PDF for download."""
    # Create QR code with request details
    qr_data = f"Student: {request['student_name']}\nAdm No: {request['student_id']}\nHostel: {request['hostel']}\nDestination: {request['destination']}\nDeparture: {request['departure_date'].strftime('%Y-%m-%d %H:%M')}\nReturn: {request['return_date'].strftime('%Y-%m-%d %H:%M')}"
    qr_image = generate_qr_code(qr_data)
    
    # Create a gate pass HTML
    html = f"""
    <div style="border: 2px solid black; padding: 20px; width: 500px; margin: auto;">
        <h2 style="text-align: center;">INSTAGATE - GATE PASS</h2>
        <p><strong>Name:</strong> {request['student_name']}</p>
        <p><strong>Admission No:</strong> {request['student_id']}</p>
        <p><strong>Hostel:</strong> {request['hostel']}</p>
        <p><strong>Reason:</strong> {request['reason']}</p>
        <p><strong>Destination:</strong> {request['destination']}</p>
        <p><strong>Departure Date:</strong> {request['departure_date'].strftime('%Y-%m-%d %H:%M')}</p>
        <p><strong>Return Date:</strong> {request['return_date'].strftime('%Y-%m-%d %H:%M')}</p>
        <p><strong>Status:</strong> {request['status']}</p>
        <div style="text-align: center;">
            <img src="data:image/png;base64,{qr_image}" style="width: 150px;">
        </div>
        <p style="text-align: center; margin-top: 20px;">This gate pass is valid only with QR code verification at the gate.</p>
    </div>
    """
    
    return html

# Streamlit UI

def main():
    st.set_page_config(
        page_title="Instagate - Gate Pass Management System",
        page_icon="🏫",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state variables if they don't exist
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
    
    # Sidebar for authentication
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=Instagate", width=150)
        st.title("Instagate")
        st.subheader("Gate Pass Management System")
        
        if not st.session_state.logged_in:
            auth_option = st.radio("Select Option", ["Login", "Register"])
            
            if auth_option == "Login":
                st.subheader("Login")
                login_role = st.selectbox("Login As", ["Student", "Parent", "Warden"])
                login_email = st.text_input("Email", key="login_email")
                login_password = st.text_input("Password", type="password", key="login_password")
                
                if st.button("Login", key="login_button"):
                    # Determine which collection to check based on role
                    if login_role == "Student":
                        user = find_user_by_email(login_email, students_data)
                        role = "student"
                    elif login_role == "Parent":
                        user = find_user_by_email(login_email, parents_data)
                        role = "parent"
                    else:  # Warden
                        user = find_user_by_email(login_email, wardens_data)
                        role = "warden"
                    
                    if user and verify_password(user["password"], login_password):
                        st.session_state.logged_in = True
                        st.session_state.user_role = role
                        st.session_state.user_data = user
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials!")
            
            else:  # Register
                st.subheader("Register")
                register_role = st.selectbox("Register As", ["Student", "Parent", "Warden"])
                register_name = st.text_input("Full Name", key="register_name")
                register_email = st.text_input("Email", key="register_email")
                register_phone = st.text_input("Phone Number", key="register_phone")
                register_password = st.text_input("Password", type="password", key="register_password")
                register_confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")
                
                # Additional fields based on role
                if register_role == "Student":
                    register_admno = st.text_input("Admission Number", key="register_admno")
                    register_department = st.text_input("Department", key="register_department")
                    register_hostel = st.selectbox("Hostel", ["A Block", "B Block", "C Block", "D Block"])
                elif register_role == "Parent":
                    register_admno = st.text_input("Student's Admission Number", key="register_student_admno")
                elif register_role == "Warden":
                    register_hostel = st.selectbox("Hostel", ["A Block", "B Block", "C Block", "D Block"])
                
                if st.button("Register", key="register_button"):
                    if register_password != register_confirm_password:
                        st.error("Passwords do not match!")
                    else:
                        # Create new user based on role
                        if register_role == "Student":
                            # Check if admission number already exists
                            if any(s["admno"] == register_admno for s in students_data):
                                st.error("Admission number already registered!")
                            else:
                                new_user = {
                                    "_id": ObjectId(),
                                    "name": register_name,
                                    "email": register_email,
                                    "password": hash_password(register_password),
                                    "admno": register_admno,
                                    "phone": register_phone,
                                    "department": register_department,
                                    "hostel": register_hostel
                                }
                                students_data.append(new_user)
                                st.success("Registration successful! Please login.")
                        
                        elif register_role == "Parent":
                            # Check if student admission number exists
                            student = find_student_by_admno(register_admno)
                            if not student:
                                st.error("Student admission number not found!")
                            # Check if parent for this student already exists
                            elif any(p["admno"] == register_admno for p in parents_data):
                                st.error("Parent for this student already registered!")
                            else:
                                new_user = {
                                    "_id": ObjectId(),
                                    "name": register_name,
                                    "email": register_email,
                                    "password": hash_password(register_password),
                                    "admno": register_admno,
                                    "phone": register_phone
                                }
                                parents_data.append(new_user)
                                st.success("Registration successful! Please login.")
                        
                        else:  # Warden
                            # Check if warden for this hostel already exists
                            if any(w["hostel"] == register_hostel for w in wardens_data):
                                st.error("Warden for this hostel already registered!")
                            else:
                                new_user = {
                                    "_id": ObjectId(),
                                    "name": register_name,
                                    "email": register_email,
                                    "password": hash_password(register_password),
                                    "hostel": register_hostel,
                                    "phone": register_phone
                                }
                                wardens_data.append(new_user)
                                st.success("Registration successful! Please login.")
        
        else:
            # Show logged in user info
            st.success(f"Logged in as {st.session_state.user_data['name']}")
            st.write(f"Role: {st.session_state.user_role.title()}")
            
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.user_role = None
                st.session_state.user_data = None
                st.rerun()
    
    # Main content area
    if not st.session_state.logged_in:
        # Show landing page for non-logged in users
        st.title("Welcome to Instagate")
        st.write("Instagate is a gate pass management system for educational institutions.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("For Students")
            st.write("- Request gate passes easily")
            st.write("- Track gate pass status")
            st.write("- Download approved gate passes")
        
        with col2:
            st.subheader("For Parents")
            st.write("- Approve student requests")
            st.write("- Stay informed about student movements")
            st.write("- Ensure student safety")
        
        with col3:
            st.subheader("For Wardens")
            st.write("- Manage all student requests")
            st.write("- Approve gate passes")
            st.write("- Maintain hostel discipline")
    
    else:
        # Show role-specific dashboard
        if st.session_state.user_role == "student":
            student_dashboard()
        elif st.session_state.user_role == "parent":
            parent_dashboard()
        else:  # Warden
            warden_dashboard()

def student_dashboard():
    st.title(f"Welcome, {st.session_state.user_data['name']}!")
    st.subheader("Student Dashboard")
    
    # Tabs for different sections
    tab1, tab2 = st.tabs(["Request Gate Pass", "My Requests"])
    
    with tab1:
        st.subheader("Request New Gate Pass")
        
        # Form for requesting gate pass
        with st.form("gatepass_request_form"):
            reason = st.text_area("Reason for Leave")
            destination = st.text_input("Destination")
            
            col1, col2 = st.columns(2)
            with col1:
                departure_date = st.date_input("Departure Date", min_value=datetime.date.today())
                departure_time = st.time_input("Departure Time", value=datetime.time(8, 0))
            
            with col2:
                return_date = st.date_input("Return Date", min_value=datetime.date.today())
                return_time = st.time_input("Return Time", value=datetime.time(18, 0))
            
            # Combine date and time
            departure_datetime = datetime.datetime.combine(departure_date, departure_time)
            return_datetime = datetime.datetime.combine(return_date, return_time)
            
            # Validate dates
            valid_dates = departure_datetime < return_datetime
            
            submit_button = st.form_submit_button("Submit Request")
            
            if submit_button:
                if not valid_dates:
                    st.error("Return date must be after departure date!")
                elif not reason or not destination:
                    st.error("All fields are required!")
                else:
                    # Create new gate pass request
                    new_request = create_gatepass_request(
                        st.session_state.user_data["admno"],
                        st.session_state.user_data["name"],
                        st.session_state.user_data["hostel"],
                        reason,
                        destination,
                        departure_datetime,
                        return_datetime
                    )
                    
                    st.success("Gate pass request submitted successfully!")
                    st.info("Your request has been sent to your parent for approval.")
    
    with tab2:
        st.subheader("My Gate Pass Requests")
        
        # Get all requests for the student
        student_requests = get_student_requests(st.session_state.user_data["admno"])
        
        if not student_requests:
            st.info("You haven't made any gate pass requests yet.")
        else:
            # Display requests in a table
            requests_df = pd.DataFrame([
                {
                    "Request ID": str(req["_id"]),
                    "Reason": req["reason"],
                    "Destination": req["destination"],
                    "Departure": req["departure_date"].strftime("%Y-%m-%d %H:%M"),
                    "Return": req["return_date"].strftime("%Y-%m-%d %H:%M"),
                    "Status": req["status"],
                    "Request Date": req["request_date"].strftime("%Y-%m-%d %H:%M")
                }
                for req in student_requests
            ])
            
            st.dataframe(requests_df)
            
            # Allow downloading approved gate passes
            approved_requests = [req for req in student_requests if req["parent_approval"] and req["warden_approval"]]
            
            if approved_requests:
                st.subheader("Approved Gate Passes")
                
                for req in approved_requests:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Destination:** {req['destination']}")
                        st.write(f"**Departure:** {req['departure_date'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Return:** {req['return_date'].strftime('%Y-%m-%d %H:%M')}")
                    
                    with col2:
                        # Generate gate pass for download
                        gate_pass_html = generate_gate_pass_pdf(req)
                        
                        # Display download button
                        if st.button(f"Download Gate Pass", key=f"download_{req['_id']}"):
                            st.markdown(gate_pass_html, unsafe_allow_html=True)
                            st.markdown("""
                                <style>
                                    @media print {
                                        body * {
                                            visibility: hidden;
                                        }
                                        .print-section, .print-section * {
                                            visibility: visible;
                                        }
                                        .print-section {
                                            position: absolute;
                                            left: 0;
                                            top: 0;
                                        }
                                    }
                                </style>
                                <script>
                                    window.print();
                                </script>
                            """, unsafe_allow_html=True)

def parent_dashboard():
    st.title(f"Welcome, {st.session_state.user_data['name']}!")
    st.subheader("Parent Dashboard")
    
    # Get student details
    student = find_student_by_admno(st.session_state.user_data["admno"])
    
    if student:
        st.write(f"**Student:** {student['name']}")
        st.write(f"**Admission No:** {student['admno']}")
        st.write(f"**Department:** {student['department']}")
        st.write(f"**Hostel:** {student['hostel']}")
    
    # Get pending requests
    pending_requests = get_parent_requests(st.session_state.user_data["admno"])
    
    if not pending_requests:
        st.info("No pending gate pass requests to approve.")
    else:
        st.subheader("Pending Gate Pass Requests")
        
        for i, req in enumerate(pending_requests):
            with st.expander(f"Request {i+1}: {req['reason']} - {req['destination']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Student:** {req['student_name']}")
                    st.write(f"**Reason:** {req['reason']}")
                    st.write(f"**Destination:** {req['destination']}")
                    st.write(f"**Departure:** {req['departure_date'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Return:** {req['return_date'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Request Date:** {req['request_date'].strftime('%Y-%m-%d %H:%M')}")
                
                with col2:
                    if st.button("Approve", key=f"approve_{req['_id']}"):
                        if approve_request_parent(str(req["_id"])):
                            st.success("Request approved successfully!")
                            st.info("Request forwarded to warden for final approval.")
                            st.rerun()
                        else:
                            st.error("Failed to approve request. Please try again.")
    
    # Show all student requests (history)
    all_student_requests = get_student_requests(st.session_state.user_data["admno"])
    
    if all_student_requests:
        st.subheader("All Gate Pass Requests")
        
        # Display requests in a table
        requests_df = pd.DataFrame([
            {
                "Request ID": str(req["_id"]),
                "Reason": req["reason"],
                "Destination": req["destination"],
                "Departure": req["departure_date"].strftime("%Y-%m-%d %H:%M"),
                "Return": req["return_date"].strftime("%Y-%m-%d %H:%M"),
                "Status": req["status"],
                "Request Date": req["request_date"].strftime("%Y-%m-%d %H:%M")
            }
            for req in all_student_requests
        ])
        
        st.dataframe(requests_df)

def warden_dashboard():
    st.title(f"Welcome, {st.session_state.user_data['name']}!")
    st.subheader("Warden Dashboard")
    
    st.write(f"**Hostel:** {st.session_state.user_data['hostel']}")
    
    # Get pending requests
    pending_requests = get_warden_requests(st.session_state.user_data["hostel"])
    
    # Tabs for different sections
    tab1, tab2 = st.tabs(["Pending Approvals", "All Gate Passes"])
    
    with tab1:
        if not pending_requests:
            st.info("No pending gate pass requests to approve.")
        else:
            st.subheader("Pending Gate Pass Requests")
            
            for i, req in enumerate(pending_requests):
                with st.expander(f"Request {i+1}: {req['student_name']} - {req['destination']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Student:** {req['student_name']}")
                        st.write(f"**Admission No:** {req['student_id']}")
                        st.write(f"**Reason:** {req['reason']}")
                        st.write(f"**Destination:** {req['destination']}")
                        st.write(f"**Departure:** {req['departure_date'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Return:** {req['return_date'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Request Date:** {req['request_date'].strftime('%Y-%m-%d %H:%M')}")
                        st.write("**Parent Approval:** Yes")
                    
                    with col2:
                        if st.button("Approve", key=f"approve_{req['_id']}"):
                            if approve_request_warden(str(req["_id"])):
                                st.success("Request approved successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to approve request. Please try again.")
    
    with tab2:
        # Get all hostel requests
        all_hostel_requests = [req for req in gatepass_requests_data if req["hostel"] == st.session_state.user_data["hostel"]]
        
        if not all_hostel_requests:
            st.info("No gate pass requests found for your hostel.")
        else:
            st.subheader("All Gate Pass Requests")
            
            # Display requests in a table
            requests_df = pd.DataFrame([
                {
                    "Student": req["student_name"],
                    "Admission No": req["student_id"],
                    "Reason": req["reason"],
                    "Destination": req["destination"],
                    "Departure": req["departure_date"].strftime("%Y-%m-%d %H:%M"),
                    "Return": req["return_date"].strftime("%Y-%m-%d %H:%M"),
                    "Status": req["status"],
                    "Parent Approval": "Yes" if req["parent_approval"] else "No",
                    "Warden Approval": "Yes" if req["warden_approval"] else "No"
                }
                for req in all_hostel_requests
            ])
            
            st.dataframe(requests_df)
            
            # Basic analytics
            total_requests = len(all_hostel_requests)
            approved_requests = len([req for req in all_hostel_requests if req["warden_approval"]])
            pending_requests = len([req for req in all_hostel_requests if not req["warden_approval"]])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Requests", total_requests)
            col2.metric("Approved", approved_requests)
            col3.metric("Pending", pending_requests)

if __name__ == "__main__":
    main()