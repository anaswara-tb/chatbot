from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List
import jwt
import bcrypt
from datetime import datetime, timedelta
import json
import csv
from io import StringIO
import databases
import sqlalchemy
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = "mysql+aiomysql://anas:12121@localhost:3306/nlu_trainer"
SYNC_DATABASE_URL = "mysql+pymysql://anas:12121@localhost:3306/nlu_trainer"
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# Define tables
users_table = sqlalchemy.Table(
    "users",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String(100)),
    sqlalchemy.Column("email", sqlalchemy.String(255), unique=True),
    sqlalchemy.Column("password", sqlalchemy.String(255)),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow)
)

projects_table = sqlalchemy.Table(
    "projects",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String(100)),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id")),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow)
)

datasets_table = sqlalchemy.Table(
    "datasets",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String(255)),
    sqlalchemy.Column("project_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("projects.id")),
    sqlalchemy.Column("record_count", sqlalchemy.Integer),
    sqlalchemy.Column("uploaded_at", sqlalchemy.DateTime, default=datetime.utcnow)
)

intents_table = sqlalchemy.Table(
    "intents",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("dataset_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("datasets.id")),
    sqlalchemy.Column("intent_name", sqlalchemy.String(100))
)

entities_table = sqlalchemy.Table(
    "entities",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("dataset_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("datasets.id")),
    sqlalchemy.Column("entity_name", sqlalchemy.String(100))
)

# Create tables
try:
    engine = sqlalchemy.create_engine(SYNC_DATABASE_URL)
    metadata.create_all(engine)
    logger.info("Tables created successfully")
except Exception as e:
    logger.error(f"Failed to create tables: {str(e)}")
    raise

# FastAPI app
app = FastAPI(title="Chatbot NLU Trainer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = "your-secret-key-change-in-production"
security = HTTPBearer()

# Models
class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ProjectCreate(BaseModel):
    name: str

class TokenResponse(BaseModel):
    token: str
    user: dict

# Helper functions
def create_token(user_id: int, email: str):
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def parse_csv_content(content: str):
    csv_reader = csv.DictReader(StringIO(content))
    return list(csv_reader)

def extract_metadata(data: List[dict]):
    intents = set()
    entities = set()
    for record in data:
        if 'intent' in record:
            intents.add(record['intent'])
        if 'entities' in record:
            ents = record['entities'].split(',') if isinstance(record['entities'], str) else []
            entities.update([e.strip() for e in ents])
    return list(intents), list(entities)

# Endpoints
@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.post("/register", response_model=TokenResponse)
async def register(user: UserRegister):
    existing = await database.fetch_one(users_table.select().where(users_table.c.email == user.email))
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt()).decode()
    query = users_table.insert().values(
        name=user.name,
        email=user.email,
        password=hashed_password
    )
    user_id = await database.execute(query)
    
    token = create_token(user_id, user.email)
    return {"token": token, "user": {"id": user_id, "name": user.name, "email": user.email}}

@app.post("/login", response_model=TokenResponse)
async def login(user: UserLogin):
    db_user = await database.fetch_one(users_table.select().where(users_table.c.email == user.email))
    if not db_user or not bcrypt.checkpw(user.password.encode(), db_user['password'].encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(db_user['id'], db_user['email'])
    return {"token": token, "user": {"id": db_user['id'], "name": db_user['name'], "email": db_user['email']}}

@app.post("/projects")
async def create_project(project: ProjectCreate, payload: dict = Depends(verify_token)):
    query = projects_table.insert().values(name=project.name, user_id=payload['user_id'])
    project_id = await database.execute(query)
    return {"id": project_id, "name": project.name, "user_id": payload['user_id']}

@app.get("/projects")
async def get_projects(payload: dict = Depends(verify_token)):
    query = projects_table.select().where(projects_table.c.user_id == payload['user_id'])
    projects = await database.fetch_all(query)
    return [dict(p) for p in projects]

@app.post("/projects/{project_id}/upload")
async def upload_dataset(project_id: int, file: UploadFile = File(...), payload: dict = Depends(verify_token)):
    content = await file.read()
    content_str = content.decode('utf-8')
    
    if file.filename.endswith('.json'):
        data = json.loads(content_str)
        if not isinstance(data, list):
            data = data.get('data', [])
    elif file.filename.endswith('.csv'):
        data = parse_csv_content(content_str)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    intents, entities = extract_metadata(data)
    
    dataset_query = datasets_table.insert().values(
        name=file.filename,
        project_id=project_id,
        record_count=len(data)
    )
    dataset_id = await database.execute(dataset_query)
    
    for intent in intents:
        await database.execute(intents_table.insert().values(dataset_id=dataset_id, intent_name=intent))
    
    for entity in entities:
        await database.execute(entities_table.insert().values(dataset_id=dataset_id, entity_name=entity))
    
    return {"dataset_id": dataset_id, "intents": intents, "entities": entities, "record_count": len(data)}

@app.get("/projects/{project_id}/datasets")
async def get_datasets(project_id: int, payload: dict = Depends(verify_token)):
    datasets = await database.fetch_all(datasets_table.select().where(datasets_table.c.project_id == project_id))
    result = []
    for dataset in datasets:
        intents = await database.fetch_all(intents_table.select().where(intents_table.c.dataset_id == dataset['id']))
        entities = await database.fetch_all(entities_table.select().where(entities_table.c.dataset_id == dataset['id']))
        result.append({
            **dict(dataset),
            "intents": [i['intent_name'] for i in intents],
            "entities": [e['entity_name'] for e in entities]
        })
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)