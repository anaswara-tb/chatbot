import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Chatbot NLU Trainer", layout="wide")

# Session state
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = None
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'login'
if 'selected_project' not in st.session_state:
    st.session_state.selected_project = None
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = None
if 'preview_dataset_id' not in st.session_state:
    st.session_state.preview_dataset_id = None
if 'current_annotation_index' not in st.session_state:
    st.session_state.current_annotation_index = 0
if 'annotations' not in st.session_state:
    st.session_state.annotations = []
if 'predicted_entities' not in st.session_state:
    st.session_state.predicted_entities = []
if 'models' not in st.session_state:
    st.session_state.models = []

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
                # files should be a dict like {'file': (filename, file_bytes, 'application/json')}
                response = requests.post(url, headers=headers, files=files)
            else:
                headers['Content-Type'] = 'application/json'
                response = requests.post(url, headers=headers, json=data)
        
        if response.status_code in [200, 201]:
            return response.json()
        else:
            try:
                error_detail = response.json().get('detail', 'Unknown error')
            except Exception:
                error_detail = response.text
            st.error(f"Error ({response.status_code}): {error_detail}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API server")
        return None
    except Exception as e:
        st.error(f"Request failed: {str(e)}")
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
                    created_date = str(project.get('created_at', ''))
                    st.caption(f"Created: {created_date.split('T')[0] if 'T' in created_date else created_date[:10]}")
                    if st.button("Open", key=f"open_{project['id']}"):
                        st.session_state.selected_project = project
                        st.session_state.current_view = 'dashboard'
                        st.rerun()
                st.divider()
    else:
        st.info("No projects yet. Create your first project above!")

def preview_modal(dataset_id, dataset_name):
    result = make_request(f"/datasets/{dataset_id}/preview?limit=50")
    
    if result:
        st.subheader(f"📋 Preview: {dataset_name}")
        st.caption(f"Showing first {result['limit']} of {result['total_records']} records")
        
        data = result['preview_records']
        for record in data:
            if 'entities' in record:
                ents = record['entities']
                if isinstance(ents, list):
                    record['entities'] = ", ".join([str(e) for e in ents]) if ents else ""
                elif isinstance(ents, str):
                    record['entities'] = ents
                else:
                    record['entities'] = str(ents) if ents is not None else ""
        df = pd.DataFrame(data)

        st.dataframe(df, use_container_width=True, height=400)
        
        col1, col2 = st.columns(2)
        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button("⬇️ Download CSV", csv_data, f"{dataset_name}_preview.csv", "text/csv")
        with col2:
            json_data = json.dumps(result['preview_records'], indent=2)
            st.download_button("⬇️ Download JSON", json_data, f"{dataset_name}_preview.json", "application/json")

def dashboard_page():
    project = st.session_state.selected_project
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"📊 {project['name']}")
    with col2:
        if st.button("← Back to Projects"):
            st.session_state.current_view = 'projects'
            st.session_state.selected_project = None
            st.rerun()
    
    st.divider()
    
    st.subheader("📤 Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV or JSON file", type=['csv', 'json'])
    
    if uploaded_file is not None:
        if st.button("Upload File", type="primary"):
            with st.spinner("Uploading..."):
                # requests expects files to be in tuple form (filename, bytes, content_type)
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or 'application/octet-stream')}
                result = make_request(f"/projects/{project['id']}/upload", "POST", files=files)
                if result:
                    st.success(f"✅ Dataset uploaded!")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Records", result['record_count'])
                    with col2:
                        st.metric("Intents", len(result['intents']))
                    with col3:
                        st.metric("Entities", len(result['entities']))
                    st.rerun()
    
    st.divider()
    st.subheader("📈 Your Datasets")
    
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
                    if st.button("🏷️ Annotate", key=f"annotate_{dataset['id']}"):
                        st.session_state.selected_dataset = dataset
                        st.session_state.current_view = 'annotation'
                        st.session_state.current_annotation_index = 0
                        st.session_state.annotations = []
                        st.rerun()
                
                uploaded_date = str(dataset.get('uploaded_at', ''))
                st.write("**Uploaded:**", uploaded_date.split('T')[0] if 'T' in uploaded_date else uploaded_date[:10])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Intents:**")
                    if dataset['intents']:
                        for intent in dataset['intents']:
                            st.markdown(f"- `{intent}`")
                    else:
                        st.caption("No intents - Start annotating!")
                
                with col2:
                    st.write("**Entities:**")
                    if dataset['entities']:
                        for entity in dataset['entities']:
                            st.markdown(f"- `{entity}`")
                    else:
                        st.caption("No entities - Start annotating!")
                
                if st.session_state.preview_dataset_id == dataset['id']:
                    st.divider()
                    preview_modal(dataset['id'], dataset['name'])
                    if st.button("✖️ Close", key=f"close_{dataset['id']}"):
                        st.session_state.preview_dataset_id = None
                        st.rerun()
    else:
        st.info("No datasets uploaded yet. Upload your first dataset above!")

    st.divider()
    # Models panel
    st.subheader("🧠 Trained Models")
    if st.button("Refresh Models"):
        res = make_request("/models")
        if res:
            st.session_state.models = res.get('models', [])
    if st.session_state.models:
        for m in st.session_state.models:
            with st.expander(f"Model: {m['model_name']} (Dataset {m['dataset_id']})", expanded=False):
                st.write("Trained on:", m.get('trained_on', ''))
                st.write("Accuracy:", m.get('accuracy', 'N/A'))
                st.write("Path:", m.get('path', ''))
                st.write("---")
                st.text_input("Enter text to test this model", key=f"test_input_{m['id']}")
                if st.button("Run Test", key=f"run_test_{m['id']}"):
                    txt = st.session_state.get(f"test_input_{m['id']}", "")
                    if txt:
                        res = make_request(f"/predict_trained/{m['id']}", "POST", {"text": txt})
                        if res:
                            st.success(f"Predicted intent: {res.get('intent')}")
                            st.write("Entities:")
                            for ent in res.get('entities', []):
                                st.write(f"- {ent['text']} → {ent['label']} ({ent['start']}:{ent['end']})")
    else:
        st.info("No models yet. Train models from the annotation screen.")

def annotation_page():
    dataset = st.session_state.selected_dataset
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"🏷️ Annotate: {dataset['name']}")
    with col2:
        if st.button("← Back"):
            st.session_state.current_view = 'dashboard'
            st.session_state.selected_dataset = None
            st.session_state.annotations = []
            st.rerun()
    
    st.divider()
    
    # Load annotations
    if not st.session_state.annotations:
        result = make_request(f"/datasets/{dataset['id']}/annotations?skip=0&limit=1000")
        if result:
            st.session_state.annotations = result['annotations']
    
    annotations = st.session_state.annotations
    
    if not annotations:
        st.info("No records to annotate")
        return
    
    current_idx = st.session_state.current_annotation_index
    total = len(annotations)
    
    if current_idx >= total:
        st.success("🎉 All records processed (annotated or skipped)!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export JSON", use_container_width=True):
                result = make_request(f"/datasets/{dataset['id']}/export?format=json")
                if result:
                    json_str = json.dumps(result['data'], indent=2)
                    st.download_button("Download JSON", json_str, f"{dataset['name']}_annotated.json", "application/json")
        
        with col2:
            if st.button("📥 Export CSV", use_container_width=True):
                result = make_request(f"/datasets/{dataset['id']}/export?format=csv")
                if result:
                    df = pd.DataFrame(result['data'])
                    csv_str = df.to_csv(index=False)
                    st.download_button("Download CSV", csv_str, f"{dataset['name']}_annotated.csv", "text/csv")
        
        st.divider()
        # Train model button
        if st.button("🧠 Train Model (spaCy)", use_container_width=True):
            with st.spinner("Training model... this may take a while depending on data size"):
                res = make_request(f"/train_model/{dataset['id']}", "POST", {})
                if res:
                    st.success("Model trained successfully!")
                    st.write("Model name:", res.get("model_name"))
                    st.write("Accuracy:", res.get("accuracy"))
                    st.write("Path:", res.get("path"))
                    # Refresh models
                    mres = make_request("/models")
                    if mres:
                        st.session_state.models = mres.get('models', [])
        st.divider()
        return
    
    current_record = annotations[current_idx]
    
    st.progress(current_idx / total, text=f"Progress: {current_idx}/{total}")
    
    st.divider()
    
    st.subheader("📝 Current Text")
    st.info(current_record['text'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 AI Prediction (spaCy small)")
        if st.button("🔮 Predict with spaCy", use_container_width=True):
            with st.spinner("Predicting..."):
                result = make_request("/annotate/predict", "POST", {"text": current_record['text']})
                if result:
                    st.success("Prediction complete!")
                    st.write("**Detected Entities:**")
                    if result['entities']:
                        for ent in result['entities']:
                            st.markdown(f"- **{ent['text']}** → `{ent['label']}`")
                    else:
                        st.caption("No entities detected")
                    st.session_state['predicted_entities'] = result['entities']
    
    with col2:
        st.subheader("✏️ Manual Annotation")
        
        intent = st.text_input(
            "Intent",
            value=current_record.get('intent', '') or '',
            placeholder="e.g., book_flight, cancel_order",
            key=f"intent_{current_idx}"
        )
        
        st.write("**Entities (JSON)**")
        default_entities = st.session_state.get('predicted_entities', [])
        if not default_entities and current_record.get('entities'):
            default_entities = current_record['entities']
        
        # Ensure we show a JSON string
        try:
            default_entities_text = json.dumps(default_entities, indent=2)
        except Exception:
            default_entities_text = "[]"
        
        entities_json = st.text_area(
            "Entities",
            value=default_entities_text,
            height=150,
            placeholder='[{"text": "Paris", "label": "destination", "start": 10, "end": 15}]',
            key=f"entities_{current_idx}"
        )
    
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("⬅️ Previous", disabled=(current_idx == 0), use_container_width=True):
            st.session_state.current_annotation_index -= 1
            st.rerun()
    
    with col2:
        if st.button("➡️ Skip", use_container_width=True):
            st.session_state.current_annotation_index += 1
            st.rerun()
    
    with col3:
        if st.button("💾 Save & Next", type="primary", use_container_width=True):
            try:
                entities_list = json.loads(entities_json) if entities_json else []
                
                save_data = {
                    "dataset_id": dataset['id'],
                    "record_id": current_record['id'],
                    "text": current_record['text'],
                    "intent": intent,
                    "entities": entities_list
                }
                
                result = make_request("/annotate/save", "POST", save_data)
                if result:
                    st.success("✅ Saved!")
                    st.session_state.current_annotation_index += 1
                    st.session_state.pop('predicted_entities', None)
                    st.rerun()
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
    
    with col4:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.current_annotation_index = 0
            st.session_state.annotations = []
            st.rerun()

# Main
if st.session_state.token is None:
    login_page()
elif st.session_state.current_view == 'projects':
    projects_page()
elif st.session_state.current_view == 'dashboard':
    dashboard_page()
elif st.session_state.current_view == 'annotation':
    annotation_page()
