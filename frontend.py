import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Chatbot NLU Trainer", layout="wide")

if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = None
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'login'
if 'selected_project' not in st.session_state:
    st.session_state.selected_project = None
if 'preview_dataset_id' not in st.session_state:
    st.session_state.preview_dataset_id = None

def make_request(endpoint, method='GET', data=None, files=None):
    headers = {}
    if st.session_state.token:
        headers['Authorization'] = f'Bearer {st.session_state.token}'
    
    url = f"{API_URL}{endpoint}"
    if method == 'GET':
        response = requests.get(url, headers=headers)
    elif method == 'POST':
        if files:
            response = requests.post(url, headers=headers, files=files)
        else:
            response = requests.post(url, headers=headers, json=data)
    
    if response.status_code in [200, 201]:
        return response.json()
    else:
        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
        return None

def login_page():
    st.title("🤖 Chatbot NLU Trainer & Evaluator")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary"):
            result = make_request("/login", "POST", {"email": email, "password": password})
            if result:
                st.session_state.token = result['token']
                st.session_state.user = result['user']
                st.session_state.current_view = 'projects'
                st.rerun()
    
    with tab2:
        st.subheader("Register")
        name = st.text_input("Name", key="register_name")
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        
        if st.button("Register", type="primary"):
            result = make_request("/register", "POST", {"name": name, "email": email, "password": password})
            if result:
                st.session_state.token = result['token']
                st.session_state.user = result['user']
                st.session_state.current_view = 'projects'
                st.rerun()

def projects_page():
    st.title("📁 Projects")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(f"Welcome, {st.session_state.user['name']}")
    with col2:
        if st.button("Logout"):
            st.session_state.token = None
            st.session_state.user = None
            st.session_state.current_view = 'login'
            st.session_state.selected_project = None
            st.rerun()
    
    st.divider()
    
    st.subheader("Create New Project")
    col1, col2 = st.columns([4, 1])
    with col1:
        project_name = st.text_input("Project Name", placeholder="e.g., HR Bot, Travel Bot")
    with col2:
        st.write("")
        st.write("")
        if st.button("Create", type="primary"):
            if project_name:
                result = make_request("/projects", "POST", {"name": project_name})
                if result:
                    st.success(f"Project '{project_name}' created!")
                    st.rerun()
    
    st.divider()
    st.subheader("Your Projects")
    
    projects = make_request("/projects")
    if projects:
        cols = st.columns(3)
        for idx, project in enumerate(projects):
            with cols[idx % 3]:
                with st.container():
                    st.markdown(f"### 📂 {project['name']}")
                    created_date = str(project['created_at'])
                    st.caption(f"Created: {created_date.split('T')[0] if 'T' in created_date else created_date[:10]}")
                    if st.button("Open", key=f"open_{project['id']}"):
                        st.session_state.selected_project = project
                        st.session_state.current_view = 'dashboard'
                        st.rerun()
                st.divider()
    else:
        st.info("No projects yet. Create your first project above!")

def preview_modal(dataset_id, dataset_name):
    """Show dataset preview in a modal-like expander"""
    result = make_request(f"/datasets/{dataset_id}/preview?limit=50")
    
    if result:
        st.subheader(f"📋 Preview: {dataset_name}")
        st.caption(f"Showing first {result['limit']} of {result['total_records']} records")
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(result['preview_records'])
        
        # Show as dataframe
        st.dataframe(df, use_container_width=True, height=400)
        
        # Option to download full data
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv_data,
                f"{dataset_name.replace('.', '_')}_preview.csv",
                "text/csv"
            )
        with col2:
            json_data = json.dumps(result['preview_records'], indent=2)
            st.download_button(
                "⬇️ Download JSON",
                json_data,
                f"{dataset_name.replace('.', '_')}_preview.json",
                "application/json"
            )

def dashboard_page():
    project = st.session_state.selected_project
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"📊 {project['name']}")
    with col2:
        if st.button("← Back to Projects"):
            st.session_state.current_view = 'projects'
            st.session_state.selected_project = None
            st.session_state.preview_dataset_id = None
            st.rerun()
    
    st.divider()
    
    st.subheader("📤 Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or JSON file", type=['csv', 'json'])
    
    if uploaded_file is not None:
        if st.button("Upload", type="primary"):
            files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
            result = make_request(f"/projects/{project['id']}/upload", "POST", files=files)
            if result:
                st.success(f"Dataset uploaded successfully!")
                st.json(result)
                st.rerun()
    
    st.divider()
    st.subheader("📈 Dataset Summary")
    
    datasets = make_request(f"/projects/{project['id']}/datasets")
    
    if datasets:
        for dataset in datasets:
            with st.expander(f"📄 {dataset['name']}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Records", dataset['record_count'])
                
                with col2:
                    st.metric("Intents", len(dataset['intents']))
                
                with col3:
                    st.metric("Entities", len(dataset['entities']))
                
                with col4:
                    st.write("")
                    if st.button("👁️ Preview", key=f"preview_{dataset['id']}"):
                        st.session_state.preview_dataset_id = dataset['id']
                        st.rerun()
                
                uploaded_date = str(dataset['uploaded_at'])
                st.write("**Uploaded:**", uploaded_date.split('T')[0] if 'T' in uploaded_date else uploaded_date[:10])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Intents:**")
                    for intent in dataset['intents']:
                        st.markdown(f"- `{intent}`")
                
                with col2:
                    st.write("**Entities:**")
                    for entity in dataset['entities']:
                        st.markdown(f"- `{entity}`")
                
                # Show preview if this dataset is selected
                if st.session_state.preview_dataset_id == dataset['id']:
                    st.divider()
                    preview_modal(dataset['id'], dataset['name'])
                    
                    if st.button("✖️ Close Preview", key=f"close_preview_{dataset['id']}"):
                        st.session_state.preview_dataset_id = None
                        st.rerun()
    else:
        st.info("No datasets uploaded yet. Upload your first dataset above!")

# Main app logic
if st.session_state.token is None:
    login_page()
elif st.session_state.current_view == 'projects':
    projects_page()
elif st.session_state.current_view == 'dashboard':
    dashboard_page()