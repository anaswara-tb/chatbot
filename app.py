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
from spacy.util import minibatch, compounding
import random
import os
import shutil

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

# Model metadata table
model_metadata_table = sqlalchemy.Table(
    "model_metadata",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("dataset_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("datasets.id")),
    sqlalchemy.Column("model_name", sqlalchemy.String(255)),
    sqlalchemy.Column("trained_on", sqlalchemy.DateTime, default=datetime.utcnow),
    sqlalchemy.Column("accuracy", sqlalchemy.Float),
    sqlalchemy.Column("path", sqlalchemy.String(255)),
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

# Load spaCy small model as a helper (not the trained model)
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load spaCy model en_core_web_sm: {e}")
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


from pydantic import field_validator

class SaveAnnotationRequest(BaseModel):
    dataset_id: int
    record_id: int
    text: str
    intent: str
    entities: List[dict]

    @field_validator("entities", mode="before")
    def parse_entities(cls, v):
        """Allow stringified JSON or list for entities"""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return v



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
    sqlalchemy.select(sqlalchemy.func.count())
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


# ====== TRAINING & MODEL MANAGEMENT ======

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def entities_to_spacy_format(text: str, entities: List[dict]):
    """Converts stored entity dicts into spaCy training spans.
    Accepts either entities with explicit start/end or entities with text+label.
    """
    spans = []
    for ent in entities:
        # If ent already has start and end
        if isinstance(ent, dict) and "start" in ent and "end" in ent and "label" in ent:
            try:
                s = int(ent["start"])
                e = int(ent["end"])
                lbl = ent["label"]
                # validate bounds
                if 0 <= s < e <= len(text):
                    spans.append((s, e, lbl))
                    continue
            except Exception:
                pass

        # fallback: find substring occurrences of the text
        if isinstance(ent, dict) and "text" in ent and "label" in ent:
            sub = str(ent["text"]).strip()
            lbl = ent["label"]
            if sub:
                idx = text.find(sub)
                if idx != -1:
                    spans.append((idx, idx + len(sub), lbl))
                    continue
        # if format is [text, label] or tuple
        if isinstance(ent, (list, tuple)) and len(ent) >= 2:
            sub = str(ent[0]).strip()
            lbl = ent[1]
            idx = text.find(sub)
            if idx != -1:
                spans.append((idx, idx + len(sub), lbl))

    # ensure spans don't overlap and are valid
    final_spans = []
    for s, e, lbl in spans:
        if s < 0 or e > len(text) or s >= e:
            continue
        final_spans.append((s, e, lbl))
    return final_spans


@app.post("/train_model/{dataset_id}")
async def train_model(dataset_id: int, payload: dict = Depends(verify_token)):
    """
    Train a spaCy NER + textcat model on the annotated dataset.
    Saves model to disk and stores metadata.
    """

    # Fetch annotated records
    query = annotations_table.select().where(
        (annotations_table.c.dataset_id == dataset_id)
        & (annotations_table.c.is_annotated == True)
    )
    rows = await database.fetch_all(query)
    if not rows:
        raise HTTPException(status_code=400, detail="No annotated records to train on")

    examples = []
    intents = set()
    for r in rows:
        text = r["text"]
        intent = r["intent"] or ""
        ents = json.loads(r["entities"]) if r["entities"] else []
        intents.add(intent)
        spans = entities_to_spacy_format(text, ents)
        examples.append({"text": text, "intent": intent, "entities": spans})

    # Convert to training formats
    ner_training = []
    textcat_training = []
    for ex in examples:
        ner_training.append((ex["text"], {"entities": ex["entities"]}))
        # Textcat single-label: label mapping
        # spaCy textcat expects dict label->bool. We'll use intent as label.
        textcat_training.append((ex["text"], {ex["intent"]: True}))

    # Basic split train/test
    combined = list(zip(ner_training, textcat_training))
    random.shuffle(combined)
    split = int(0.8 * len(combined)) or (len(combined) - 1)
    train_data = combined[:split]
    dev_data = combined[split:] if split < len(combined) else []

    # Prepare spaCy pipeline: blank English model
    model_name = f"model_dataset_{dataset_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    model_path = os.path.join(MODELS_DIR, model_name)

    # Create blank model
    try:
        train_nlp = spacy.blank("en")
    except Exception:
        # fallback to loading small model and disabling components
        if nlp:
            train_nlp = nlp.__class__.from_config(nlp.config)
        else:
            train_nlp = spacy.blank("en")

    # add NER
    if "ner" not in train_nlp.pipe_names:
        ner = train_nlp.add_pipe("ner")
    else:
        ner = train_nlp.get_pipe("ner")

    # add labels to NER
    all_entity_labels = set()
    for _, (ner_ex, _) in enumerate(train_data):
        entities = ner_ex[1].get("entities", [])
        for s, e, lbl in entities:
            all_entity_labels.add(lbl)
            ner.add_label(lbl)
    # Also from dev set
    for _, (ner_ex, _) in enumerate(dev_data):
        entities = ner_ex[1].get("entities", [])
        for s, e, lbl in entities:
            all_entity_labels.add(lbl)
            ner.add_label(lbl)

    # add textcat (single label)
    if "textcat" not in train_nlp.pipe_names:
        textcat = train_nlp.add_pipe("textcat", last=True)
        # configure for single-label classification
        for intent_label in intents:
            if intent_label:
                textcat.add_label(intent_label)
    else:
        textcat = train_nlp.get_pipe("textcat")

    # Prepare optimizer
    optimizer = train_nlp.begin_training()
    n_iter = 12
    # Convert training data into spaCy format for training loops
    spacy_train_ner = [(t[0], t[1]) for (t, _) in train_data]
    spacy_train_textcat = [(t[0], t[1]) for (_, t) in train_data]

    # Combine training into tuples (text, {"entities": [...]}, {"cats": {...}})
    combined_train = []
    for (text_ner, ann_ner), (text_cat, ann_cat) in zip(spacy_train_ner, spacy_train_textcat):
        cats = ann_cat
        combined_train.append((text_ner, {"entities": ann_ner.get("entities", [])}, cats))

    logger.info(f"Starting training for model {model_name} on {len(combined_train)} examples")

    try:
        for itn in range(n_iter):
            random.shuffle(combined_train)
            batches = minibatch(combined_train, size=compounding(4.0, 32.0, 1.5))
            losses = {}
            from spacy.training import Example

            from spacy.training import Example

            for batch in batches:
                examples = []
                for t in batch:
                    text = t[0]
                    ann = t[1]
                    cats = t[2]
                    doc = train_nlp.make_doc(text)
                    example = Example.from_dict(doc, {"entities": ann["entities"], "cats": cats})
                    examples.append(example)
                    train_nlp.update(examples, sgd=optimizer, drop=0.2, losses=losses)


            logger.info(f"Iteration {itn+1}/{n_iter} Losses: {losses}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    # Evaluation on dev set (simple accuracy for textcat)
    acc = None
    if dev_data:
        correct = 0
        total = 0
        for (ner_ex, _), (_, cat_ex) in dev_data:
            text = ner_ex[0]
            preds = train_nlp(text)
            # predict highest scoring label for textcat
            if hasattr(preds, "cats"):
                if len(preds.cats) == 0:
                    # no cat predictions
                    total += 1
                    continue
                # pick label with max score
                pred_label = max(preds.cats.items(), key=lambda x: x[1])[0]
                # find true label (only one key in cat_ex is True)
                true_label = None
                for k, v in cat_ex.items():
                    if v:
                        true_label = k
                        break
                if true_label == pred_label:
                    correct += 1
                total += 1
        acc = (correct / total) if total > 0 else None

    # Save model
    try:
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        train_nlp.to_disk(model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save model: {e}")

    # store metadata
    meta_ins = model_metadata_table.insert().values(
        dataset_id=dataset_id,
        model_name=model_name,
        trained_on=datetime.utcnow(),
        accuracy=acc,
        path=model_path,
    )
    model_id = await database.execute(meta_ins)

    return {
        "message": "Training completed",
        "model_id": model_id,
        "model_name": model_name,
        "accuracy": acc,
        "path": model_path,
    }


@app.get("/models")
async def list_models(payload: dict = Depends(verify_token)):
    rows = await database.fetch_all(model_metadata_table.select().order_by(model_metadata_table.c.trained_on.desc()))
    result = []
    for r in rows:
        result.append({
            "id": r["id"],
            "dataset_id": r["dataset_id"],
            "model_name": r["model_name"],
            "trained_on": r["trained_on"].isoformat() if r["trained_on"] else None,
            "accuracy": r["accuracy"],
            "path": r["path"],
        })
    return {"models": result}


@app.post("/predict_trained/{model_id}")
async def predict_trained(model_id: int, request: PredictRequest, payload: dict = Depends(verify_token)):
    row = await database.fetch_one(model_metadata_table.select().where(model_metadata_table.c.id == model_id))
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")
    path = row["path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail="Model files missing on disk")

    try:
        model_nlp = spacy.load(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    doc = model_nlp(request.text)

    # Entities
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })

    # Intent/textcat
    intent = None
    if hasattr(doc, "cats") and doc.cats:
        intent = max(doc.cats.items(), key=lambda x: x[1])[0]

    return {
        "text": request.text,
        "intent": intent,
        "entities": entities
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
