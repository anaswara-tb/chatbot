import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import plotly.express as px

# Backend API URL - Update this with your actual backend URL
API_BASE_URL = "http://localhost:5000/api"

# Initialize session state
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'current_project' not in st.session_state:
    st.session_state.current_project = None

# Helper function for API calls with auth
def api_call(endpoint, method="GET", data=None, files=None):
    headers = {}
    if st.session_state.token:
        headers['Authorization'] = f'Bearer {st.session_state.token}'
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            if files:
                response = requests.post(url, headers=headers, files=files, data=data)
            else:
                headers['Content-Type'] = 'application/json'
                response = requests.post(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        
        return response
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# Logout function
def logout():
    st.session_state.token = None
    st.session_state.user_email = None
    st.session_state.current_project = None
    st.rerun()

# AUTHENTICATION PAGE
def auth_page():
    st.title("🤖 Chatbot NLU Trainer & Evaluator")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                response = api_call("/auth/login", "POST", {
                    "email": email,
                    "password": password
                })
                
                if response and response.status_code == 200:
                    data = response.json()
                    st.session_state.token = data['token']
                    st.session_state.user_email = email
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    
    with tab2:
        st.subheader("Register")
        with st.form("register_form"):
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_confirm = st.text_input("Confirm Password", type="password")
            submit_reg = st.form_submit_button("Register")
            
            if submit_reg:
                if reg_password != reg_confirm:
                    st.error("Passwords don't match")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    response = api_call("/auth/register", "POST", {
                        "email": reg_email,
                        "password": reg_password
                    })
                    
                    if response and response.status_code == 201:
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Registration failed. Email may already exist.")

# PROJECT DASHBOARD
def project_dashboard():
    st.title("🤖 NLU Trainer Dashboard")
    
    # Sidebar
    with st.sidebar:
        st.write(f"👤 **{st.session_state.user_email}**")
        if st.button("Logout"):
            logout()
        
        st.divider()
        
        # Create new project
        st.subheader("Create New Project")
        with st.form("new_project"):
            project_name = st.text_input("Project Name", placeholder="e.g., HR Bot")
            project_desc = st.text_area("Description", placeholder="Optional")
            if st.form_submit_button("Create Project"):
                response = api_call("/projects", "POST", {
                    "name": project_name,
                    "description": project_desc
                })
                
                if response and response.status_code == 201:
                    st.success(f"Project '{project_name}' created!")
                    st.rerun()
                else:
                    st.error("Failed to create project")
    
    # Main area - Project list
    response = api_call("/projects")
    
    if response and response.status_code == 200:
        projects = response.json()
        
        if len(projects) == 0:
            st.info("👈 Create your first project using the sidebar")
        else:
            st.subheader("Your Projects")
            
            cols = st.columns(3)
            for idx, project in enumerate(projects):
                with cols[idx % 3]:
                    with st.container(border=True):
                        st.markdown(f"### 📁 {project['name']}")
                        st.caption(f"Created: {project.get('createdAt', 'N/A')}")
                        if project.get('description'):
                            st.write(project['description'])
                        
                        if st.button("Open", key=f"open_{project['id']}"):
                            st.session_state.current_project = project
                            st.rerun()
    else:
        st.error("Failed to load projects")

# DATASET UPLOAD & ANALYSIS PAGE
def dataset_page():
    project = st.session_state.current_project
    
    # Header with back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back"):
            st.session_state.current_project = None
            st.rerun()
    with col2:
        st.title(f"📁 {project['name']}")
    
    st.divider()
    
    # Upload section
    st.subheader("📤 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Upload CSV or JSON file",
        type=['csv', 'json'],
        help="Upload your training data with intents and entities"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info(f"**File:** {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        with col2:
            if st.button("Process & Upload", type="primary"):
                with st.spinner("Processing dataset..."):
                    files = {'file': uploaded_file}
                    response = api_call(
                        f"/projects/{project['id']}/datasets",
                        "POST",
                        files=files
                    )
                    
                    if response and response.status_code == 201:
                        st.success("Dataset uploaded successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to upload dataset")
    
    st.divider()
    
    # Dataset analysis
    response = api_call(f"/projects/{project['id']}/analysis")
    
    if response and response.status_code == 200:
        analysis = response.json()
        
        st.subheader("📊 Dataset Overview")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", analysis.get('total_samples', 0))
        with col2:
            st.metric("Unique Intents", analysis.get('unique_intents', 0))
        with col3:
            st.metric("Entity Types", analysis.get('entity_types', 0))
        with col4:
            st.metric("Datasets", analysis.get('dataset_count', 0))
        
        st.divider()
        
        # Intent distribution
        if analysis.get('intent_distribution'):
            st.subheader("Intent Distribution")
            
            intent_data = analysis['intent_distribution']
            df_intents = pd.DataFrame([
                {'Intent': k, 'Count': v} 
                for k, v in intent_data.items()
            ]).sort_values('Count', ascending=False)
            
            fig = px.bar(
                df_intents,
                x='Intent',
                y='Count',
                title='Samples per Intent',
                color='Count',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Intent table
            st.dataframe(df_intents, use_container_width=True)
        
        # Entity distribution
        if analysis.get('entity_distribution'):
            st.subheader("Entity Types")
            
            entity_data = analysis['entity_distribution']
            df_entities = pd.DataFrame([
                {'Entity Type': k, 'Count': v}
                for k, v in entity_data.items()
            ]).sort_values('Count', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig_entities = px.pie(
                    df_entities,
                    values='Count',
                    names='Entity Type',
                    title='Entity Type Distribution'
                )
                st.plotly_chart(fig_entities, use_container_width=True)
            
            with col2:
                st.dataframe(df_entities, use_container_width=True)
        
        # Sample data preview
        if analysis.get('samples'):
            st.subheader("Sample Data")
            df_samples = pd.DataFrame(analysis['samples'])
            st.dataframe(df_samples.head(10), use_container_width=True)
    
    else:
        st.info("No dataset uploaded yet. Upload a dataset to see analysis.")

# MAIN APP
def main():
    st.set_page_config(
        page_title="NLU Trainer",
        page_icon="🤖",
        layout="wide"
    )
    
    # Check authentication
    if not st.session_state.token:
        auth_page()
    else:
        # Check if project is selected
        if st.session_state.current_project:
            dataset_page()
        else:
            project_dashboard()

if __name__ == "__main__":
    main()