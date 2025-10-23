# filename: app.py
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
from sqlalchemy.dialects.mysql import LONGTEXT   # <-- NEW
import logging

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Database
# -------------------------------------------------
DATABASE_URL = "mysql+aiomysql://anas:12121@localhost:3306/nlu_trainer"
SYNC_DATABASE_URL = "mysql+pymysql://anas:12121@localhost:3306/nlu_trainer"

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# -------------------------------------------------
# Tables
# -------------------------------------------------
users_table = sqlalchemy.Table(
    "users",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String(100)),
    sqlalchemy.Column("email", sqlalchemy.String(255), unique=True),
    sqlalchemy.Column("password", sqlalchemy.String(255)),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
)

projects_table = sqlalchemy.Table(
    "projects",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String(100)),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id")),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, default=datetime.utcnow),
)

datasets_table = sqlalchemy.Table(
    "datasets",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String(255)),
    sqlalchemy.Column("project_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("projects.id")),
    sqlalchemy.Column("record_count", sqlalchemy.Integer),
    sqlalchemy.Column("data", LONGTEXT),                 # <-- LONGTEXT (4 GB)
    sqlalchemy.Column("uploaded_at", sqlalchemy.DateTime, default=datetime.utcnow),
)

intents_table = sqlalchemy.Table(
    "intents",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("dataset_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("datasets.id")),
    sqlalchemy.Column("intent_name", sqlalchemy.String(100)),
)

entities_table = sqlalchemy.Table(
    "entities",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("dataset_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("datasets.id")),
    sqlalchemy.Column("entity_name", sqlalchemy.String(100)),
)

# -------------------------------------------------
# Create tables (idempotent)
# -------------------------------------------------
try:
    engine = sqlalchemy.create_engine(SYNC_DATABASE_URL)
    metadata.create_all(engine)
    logger.info("Tables ensured")
except Exception as e:
    logger.error(f"Failed to create tables: {e}")
    raise

# -------------------------------------------------
# FastAPI
# -------------------------------------------------
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

# -------------------------------------------------
# Pydantic models
# -------------------------------------------------
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

# -------------------------------------------------
# JWT helpers
# -------------------------------------------------
def create_token(user_id: int, email: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=7),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# -------------------------------------------------
# CSV / JSON helpers
# -------------------------------------------------
def parse_csv_content(content: str) -> List[dict]:
    reader = csv.DictReader(StringIO(content))
    return list(reader)

def extract_metadata(data: List[dict]) -> tuple[list[str], list[str]]:
    intents = set()
    entities = set()
    for rec in data:
        if "intent" in rec:
            intents.add(rec["intent"])
        if "entities" in rec:
            ents = rec["entities"]
            if isinstance(ents, str):
                ents = [e.strip() for e in ents.split(",") if e.strip()]
            entities.update(ents)
    return list(intents), list(entities)

# -------------------------------------------------
# Size guard (100 MB)
# -------------------------------------------------
MAX_DATA_SIZE_BYTES = 100 * 1024 * 1024   # 100 MB

# -------------------------------------------------
# Startup / shutdown
# -------------------------------------------------
@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

# -------------------------------------------------
# Auth endpoints
# -------------------------------------------------
@app.post("/register", response_model=TokenResponse)
async def register(user: UserRegister):
    existing = await database.fetch_one(
        users_table.select().where(users_table.c.email == user.email)
    )
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt()).decode()
    query = users_table.insert().values(
        name=user.name, email=user.email, password=hashed
    )
    user_id = await database.execute(query)

    token = create_token(user_id, user.email)
    return {"token": token, "user": {"id": user_id, "name": user.name, "email": user.email}}

@app.post("/login", response_model=TokenResponse)
async def login(user: UserLogin):
    db_user = await database.fetch_one(
        users_table.select().where(users_table.c.email == user.email)
    )
    if not db_user or not bcrypt.checkpw(user.password.encode(), db_user["password"].encode()):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(db_user["id"], db_user["email"])
    return {
        "token": token,
        "user": {"id": db_user["id"], "name": db_user["name"], "email": db_user["email"]},
    }

# -------------------------------------------------
# Project endpoints
# -------------------------------------------------
@app.post("/projects")
async def create_project(project: ProjectCreate, payload: dict = Depends(verify_token)):
    q = projects_table.insert().values(name=project.name, user_id=payload["user_id"])
    pid = await database.execute(q)
    return {"id": pid, "name": project.name, "user_id": payload["user_id"]}

@app.get("/projects")
async def list_projects(payload: dict = Depends(verify_token)):
    q = projects_table.select().where(projects_table.c.user_id == payload["user_id"])
    rows = await database.fetch_all(q)
    return [dict(r) for r in rows]

# -------------------------------------------------
# Dataset upload
# -------------------------------------------------
@app.post("/projects/{project_id}/upload")
async def upload_dataset(
    project_id: int,
    file: UploadFile = File(...),
    payload: dict = Depends(verify_token),
):
    # ---- 1. Verify project ownership ----
    proj = await database.fetch_one(
        projects_table.select().where(
            (projects_table.c.id == project_id)
            & (projects_table.c.user_id == payload["user_id"])
        )
    )
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found or access denied")

    # ---- 2. Read file ----
    content_bytes = await file.read()
    content_str = content_bytes.decode("utf-8")

    # ---- 3. Parse according to extension ----
    if file.filename.lower().endswith(".json"):
        try:
            data = json.loads(content_str)
            # Accept top-level list or common wrappers
            if isinstance(data, dict):
                if "data" in data:
                    data = data["data"]
                elif "examples" in data:
                    data = data["examples"]
                else:
                    raise ValueError("JSON must contain a list under 'data' or 'examples'")
            if not isinstance(data, list):
                raise ValueError("JSON root must be a list")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    elif file.filename.lower().endswith(".csv"):
        data = parse_csv_content(content_str)
        if not data:
            raise HTTPException(status_code=400, detail="CSV is empty")
    else:
        raise HTTPException(status_code=400, detail="Only .csv and .json are supported")

    if not data:
        raise HTTPException(status_code=400, detail="No records found")

    # ---- 4. Size guard ----
    json_str = json.dumps(data)
    if len(json_str.encode("utf-8")) > MAX_DATA_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset too large (> {MAX_DATA_SIZE_BYTES // (1024*1024)} MB)",
        )

    # ---- 5. Extract metadata ----
    intents, entities = extract_metadata(data)

    # ---- 6. Persist ----
    ds_ins = datasets_table.insert().values(
        name=file.filename,
        project_id=project_id,
        record_count=len(data),
        data=json_str,
    )
    dataset_id = await database.execute(ds_ins)

    # intents / entities
    for i in intents:
        await database.execute(intents_table.insert().values(dataset_id=dataset_id, intent_name=i))
    for e in entities:
        await database.execute(entities_table.insert().values(dataset_id=dataset_id, entity_name=e))

    return {
        "dataset_id": dataset_id,
        "intents": intents,
        "entities": entities,
        "record_count": len(data),
    }

# -------------------------------------------------
# List datasets (summary)
# -------------------------------------------------
@app.get("/projects/{project_id}/datasets")
async def list_datasets(project_id: int, payload: dict = Depends(verify_token)):
    rows = await database.fetch_all(
        datasets_table.select().where(datasets_table.c.project_id == project_id)
    )
    result = []
    for ds in rows:
        intents = await database.fetch_all(
            intents_table.select().where(intents_table.c.dataset_id == ds["id"])
        )
        entities = await database.fetch_all(
            entities_table.select().where(entities_table.c.dataset_id == ds["id"])
        )
        result.append(
            {
                **dict(ds),
                "intents": [i["intent_name"] for i in intents],
                "entities": [e["entity_name"] for e in entities],
                "data": None,  # omit heavy payload
            }
        )
    return result

# -------------------------------------------------
# Dataset preview (first N records)
# -------------------------------------------------
@app.get("/datasets/{dataset_id}/preview")
async def preview_dataset(dataset_id: int, limit: int = 50, payload: dict = Depends(verify_token)):
    ds = await database.fetch_one(
        datasets_table.select().where(datasets_table.c.id == dataset_id)
    )
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    data = json.loads(ds["data"])
    return {
        "dataset_id": dataset_id,
        "name": ds["name"],
        "total_records": len(data),
        "preview_records": data[:limit],
        "limit": limit,
    }

# -------------------------------------------------
# Run with uvicorn
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)