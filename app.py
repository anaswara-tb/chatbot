from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import jwt
import bcrypt
from datetime import datetime, timedelta
import json
import csv
from io import StringIO
import databases
import sqlalchemy
from sqlalchemy.dialects.mysql import LONGTEXT
import logging
import spacy

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database
DATABASE_URL = "mysql+aiomysql://anas:12121@localhost:3306/nlu_trainer"
SYNC_DATABASE_URL = "mysql+pymysql://anas:12121@localhost:3306/nlu_trainer"

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# Tables
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
    sqlalchemy.Column(
        "project_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("projects.id")
    ),
    sqlalchemy.Column("record_count", sqlalchemy.Integer),
    sqlalchemy.Column("data", LONGTEXT),
    sqlalchemy.Column("uploaded_at", sqlalchemy.DateTime, default=datetime.utcnow),
)

intents_table = sqlalchemy.Table(
    "intents",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column(
        "dataset_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("datasets.id")
    ),
    sqlalchemy.Column("intent_name", sqlalchemy.String(100)),
)

entities_table = sqlalchemy.Table(
    "entities",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column(
        "dataset_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("datasets.id")
    ),
    sqlalchemy.Column("entity_name", sqlalchemy.String(100)),
)

# Annotations table
annotations_table = sqlalchemy.Table(
    "annotations",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column(
        "dataset_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("datasets.id")
    ),
    sqlalchemy.Column("text", sqlalchemy.Text),
    sqlalchemy.Column("intent", sqlalchemy.String(100)),
    sqlalchemy.Column("entities", sqlalchemy.Text),
    sqlalchemy.Column("is_annotated", sqlalchemy.Boolean, default=False),
    sqlalchemy.Column("annotated_at", sqlalchemy.DateTime),
)

# Create tables
try:
    engine = sqlalchemy.create_engine(SYNC_DATABASE_URL)
    metadata.create_all(engine)
    logger.info("Tables ensured")
except Exception as e:
    logger.error(f"Failed to create tables: {e}")
    raise

# FastAPI
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

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    nlp = None


# Pydantic models
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


class PredictRequest(BaseModel):
    text: str


class SaveAnnotationRequest(BaseModel):
    dataset_id: int
    record_id: int
    text: str
    intent: str
    entities: List[dict]


# JWT helpers
def create_token(user_id: int, email: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=7),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# CSV / JSON helpers
def parse_csv_content(content: str) -> List[dict]:
    reader = csv.DictReader(StringIO(content))
    return list(reader)


def extract_metadata(data: List[dict]) -> tuple[list[str], list[str]]:
    intents = set()
    entities = set()
    for rec in data:
        if "intent" in rec and rec["intent"]:
            intent_val = str(rec["intent"]).strip()
            if intent_val:
                intents.add(intent_val)

        if "entities" in rec and rec["entities"]:
            ents = rec["entities"]
            if isinstance(ents, str):
                ents = [e.strip() for e in ents.split(",") if e.strip()]
            elif isinstance(ents, list):
                ents = [str(e).strip() for e in ents if str(e).strip()]
            entities.update(ents)

    return list(intents), list(entities)


MAX_DATA_SIZE_BYTES = 100 * 1024 * 1024


# Startup / shutdown
@app.on_event("startup")
async def startup():
    await database.connect()


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


# Auth endpoints
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
    return {
        "token": token,
        "user": {"id": user_id, "name": user.name, "email": user.email},
    }


@app.post("/login", response_model=TokenResponse)
async def login(user: UserLogin):
    db_user = await database.fetch_one(
        users_table.select().where(users_table.c.email == user.email)
    )
    if not db_user or not bcrypt.checkpw(
        user.password.encode(), db_user["password"].encode()
    ):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(db_user["id"], db_user["email"])
    return {
        "token": token,
        "user": {
            "id": db_user["id"],
            "name": db_user["name"],
            "email": db_user["email"],
        },
    }


# Project endpoints
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


# Dataset upload
@app.post("/projects/{project_id}/upload")
async def upload_dataset(
    project_id: int, file: UploadFile = File(...), payload: dict = Depends(verify_token)
):
    proj = await database.fetch_one(
        projects_table.select().where(
            (projects_table.c.id == project_id)
            & (projects_table.c.user_id == payload["user_id"])
        )
    )
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    content_bytes = await file.read()
    content_str = content_bytes.decode("utf-8")

    if file.filename.lower().endswith(".json"):
        try:
            data = json.loads(content_str)
            if isinstance(data, dict):
                if "data" in data:
                    data = data["data"]
                elif "examples" in data:
                    data = data["examples"]
            if not isinstance(data, list):
                raise ValueError("JSON must be a list")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    elif file.filename.lower().endswith(".csv"):
        data = parse_csv_content(content_str)
    else:
        raise HTTPException(status_code=400, detail="Only .csv and .json supported")

    if not data:
        raise HTTPException(status_code=400, detail="No records found")

    json_str = json.dumps(data)
    if len(json_str.encode("utf-8")) > MAX_DATA_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Dataset too large")

    intents, entities = extract_metadata(data)

    ds_ins = datasets_table.insert().values(
        name=file.filename,
        project_id=project_id,
        record_count=len(data),
        data=json_str,
    )
    dataset_id = await database.execute(ds_ins)

    # Store annotations for each record
    for idx, record in enumerate(data):
        text = record.get("text", "").strip()
        # Support missing intent/entities
        intent = record.get("intent", "").strip() if "intent" in record else None
        ents = record.get("entities") if "entities" in record else None
        # Assign default empty list to entities if missing/blank
        if not ents:
            ents = []
        # Only create annotation if text exists
        if text:
            await database.execute(
                annotations_table.insert().values(
                    dataset_id=dataset_id,
                    text=text,
                    intent=intent,
                    entities=json.dumps(ents),
                    is_annotated=bool(intent),
                )
            )

    for i in intents:
        await database.execute(
            intents_table.insert().values(dataset_id=dataset_id, intent_name=i)
        )
    for e in entities:
        await database.execute(
            entities_table.insert().values(dataset_id=dataset_id, entity_name=e)
        )

    return {
        "dataset_id": dataset_id,
        "intents": intents,
        "entities": entities,
        "record_count": len(data),
    }


# List datasets
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
                "data": None,
            }
        )
    return result


# Dataset preview
@app.get("/datasets/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: int, limit: int = 50, payload: dict = Depends(verify_token)
):
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


# ===== ANNOTATION ENDPOINTS =====


@app.post("/annotate/predict")
async def predict_entities(
    request: PredictRequest, payload: dict = Depends(verify_token)
):
    """Use spaCy to predict entities"""
    if not nlp:
        raise HTTPException(status_code=500, detail="spaCy model not loaded")

    doc = nlp(request.text)

    entities = []
    for ent in doc.ents:
        entities.append(
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
        )

    return {
        "text": request.text,
        "entities": entities,
        "tokens": [token.text for token in doc],
    }


@app.get("/datasets/{dataset_id}/annotations")
async def get_annotations(
    dataset_id: int,
    skip: int = 0,
    limit: int = 10,
    payload: dict = Depends(verify_token),
):
    """Get annotations for dataset"""
    query = (
        annotations_table.select()
        .where(annotations_table.c.dataset_id == dataset_id)
        .offset(skip)
        .limit(limit)
    )

    rows = await database.fetch_all(query)

    result = []
    for row in rows:
        result.append(
            {
                "id": row["id"],
                "text": row["text"],
                "intent": row["intent"],
                "entities": json.loads(row["entities"]) if row["entities"] else [],
                "is_annotated": row["is_annotated"],
            }
        )

    # Get total count
    count_query = (
        sqlalchemy.select([sqlalchemy.func.count()])
        .select_from(annotations_table)
        .where(annotations_table.c.dataset_id == dataset_id)
    )
    total = await database.fetch_val(count_query)

    return {"annotations": result, "total": total, "skip": skip, "limit": limit}


@app.post("/annotate/save")
async def save_annotation(
    request: SaveAnnotationRequest, payload: dict = Depends(verify_token)
):
    """Save user annotation"""
    update_query = (
        annotations_table.update()
        .where(annotations_table.c.id == request.record_id)
        .values(
            intent=request.intent,
            entities=json.dumps(request.entities),
            is_annotated=True,
            annotated_at=datetime.utcnow(),
        )
    )

    await database.execute(update_query)

    return {"message": "Annotation saved successfully"}


@app.get("/datasets/{dataset_id}/export")
async def export_annotations(
    dataset_id: int, format: str = "json", payload: dict = Depends(verify_token)
):
    """Export annotated data"""
    query = annotations_table.select().where(
        (annotations_table.c.dataset_id == dataset_id)
        & (annotations_table.c.is_annotated == True)
    )

    rows = await database.fetch_all(query)

    data = []
    for row in rows:
        data.append(
            {
                "text": row["text"],
                "intent": row["intent"],
                "entities": json.loads(row["entities"]) if row["entities"] else [],
            }
        )

    if format == "json":
        return {"data": data, "count": len(data)}
    elif format == "csv":
        csv_data = []
        for item in data:
            entities_str = json.dumps(item["entities"])
            csv_data.append(
                {
                    "text": item["text"],
                    "intent": item["intent"],
                    "entities": entities_str,
                }
            )
        return {"data": csv_data, "count": len(csv_data)}
    else:
        raise HTTPException(status_code=400, detail="Format must be json or csv")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
