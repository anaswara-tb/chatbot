import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
from io import StringIO

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
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            if files:
                # For file uploads, don't set Content-Type header (requests will set it with boundary)
                response = requests.post(url, headers=headers, files=files)
            else:
                # For JSON data
                headers['Content-Type'] = 'application/json'
                response = requests.post(url, headers=headers, json=data)
        
        # Log response for debugging
        logger_msg = f"Response status: {response.status_code}"
        
        if response.status_code in [200, 201]:
            try:
                return response.json()
            except json.JSONDecodeError as e:
                st.error(f"Server returned invalid JSON. Status: {response.status_code}")
                st.error(f"Response text: {response.text[:500]}")  # Show first 500 chars
                return None
        else:
            try:
                error_detail = response.json().get('detail', 'Unknown error')
            except:
                error_detail = response.text or 'Unknown error'
            st.error(f"Error ({response.status_code}): {error_detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the API server. Make sure it's running on http://localhost:8000")
        return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def download_from_url(url):
    """Download file from URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content.decode('utf-8')
    except Exception as e:
        st.error(f"Failed to download from URL: {str(e)}")
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
    
    # Tab for file upload vs URL
    upload_tab1, upload_tab2 = st.tabs(["📁 Upload File", "🔗 From URL"])
    
    with upload_tab1:
        uploaded_file = st.file_uploader("Choose a CSV or JSON file", type=['csv', 'json'], key="file_upload")
        
        if uploaded_file is not None:
            if st.button("Upload File", type="primary", key="upload_file_btn"):
                with st.spinner("Uploading..."):
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
                    result = make_request(f"/projects/{project['id']}/upload", "POST", files=files)
                    if result:
                        st.success(f"✅ Dataset uploaded successfully!")
                        with st.expander("Upload Details", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Records", result['record_count'])
                            with col2:
                                st.metric("Intents", len(result['intents']))
                            with col3:
                                st.metric("Entities", len(result['entities']))
                        st.rerun()
    
    with upload_tab2:
        st.info("💡 Supported sources: HuggingFace datasets, GitHub raw files, Google Drive (public), Dropbox, etc.")
        
        url = st.text_input(
            "Dataset URL", 
            placeholder="e.g., https://huggingface.co/datasets/.../raw/main/data.csv",
            key="url_input"
        )
        
        if url:
            # Extract filename from URL
            filename = url.split('/')[-1]
            if not filename.endswith(('.csv', '.json')):
                st.warning("⚠️ URL must point to a .csv or .json file")
            else:
                if st.button("Upload from URL", type="primary", key="upload_url_btn"):
                    with st.spinner("Downloading and uploading..."):
                        content = download_from_url(url)
                        if content:
                            files = {'file': (filename, content.encode())}
                            result = make_request(f"/projects/{project['id']}/upload", "POST", files=files)
                            if result:
                                st.success(f"✅ Dataset uploaded successfully from URL!")
                                with st.expander("Upload Details", expanded=True):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Records", result['record_count'])
                                    with col2:
                                        st.metric("Intents", len(result['intents']))
                                    with col3:
                                        st.metric("Entities", len(result['entities']))
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
                    if dataset['intents']:
                        for intent in dataset['intents']:
                            st.markdown(f"- `{intent}`")
                    else:
                        st.caption("No intents found")
                
                with col2:
                    st.write("**Entities:**")
                    if dataset['entities']:
                        for entity in dataset['entities']:
                            st.markdown(f"- `{entity}`")
                    else:
                        st.caption("No entities found")
                
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