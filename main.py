from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import joblib
import json
import pandas as pd
from pydantic import BaseModel, EmailStr
from utils import process_and_extract
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import uuid 

app = FastAPI()

# --- 1. CORS SETTINGS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. CONFIGURATION & FILE PATHS ---
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
HISTORY_FILE = BASE_DIR / "history.json"
USER_FILE = BASE_DIR / "users.json"
ADMIN_SECRET_KEY = "HCMC2_3-COS30049" 

MODEL_PATH = MODELS_DIR / "ds1_best_svm.pkl"
model_pipeline = None

if MODEL_PATH.exists():
    try:
        model_pipeline = joblib.load(str(MODEL_PATH))
        print("✅ Model loaded")
    except Exception as e:
        print("❌ Model load failed:", e)
        model_pipeline = None
else:
    print(f"⚠️ Warning: Model file not found at {MODEL_PATH}")

# --- 3. DATA MODELS ---

class UserRegistration(BaseModel):
    username: str
    password: str
    email: EmailStr
    country: str
    day_of_birth: str
    sex: str 
    admin_key: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class MailInput(BaseModel):
    content: str
    username: str = "Guest"

class UpdateHistoryLabel(BaseModel):
    id: str
    new_label: str 
    role: str

# --- 4. HELPERS ---

def load_json(file_path: Path):
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except:
            return []

def save_json(file_path: Path, data: list):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# --- 5. AUTH ---

@app.post("/register")
async def register_user(user: UserRegistration):
    users = load_json(USER_FILE)
    
    if any(u['username'] == user.username for u in users):
        raise HTTPException(status_code=400, detail="Username already exists")
    
    role = "admin" if user.admin_key == ADMIN_SECRET_KEY else "user"
    
    user_data = user.dict()
    user_data["role"] = role
    del user_data["admin_key"]
    
    users.append(user_data)
    save_json(USER_FILE, users)
    return {"status": "success", "role": role}

@app.post("/login")
async def login_user(credentials: UserLogin):
    users = load_json(USER_FILE)
    user = next((u for u in users if u['username'] == credentials.username and u['password'] == credentials.password), None)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    return {"status": "success", "username": user["username"], "role": user["role"]}

# --- 6. PREDICT ---

@app.post("/predict")
async def predict(input_data: MailInput):

    clean_text, features = process_and_extract(input_data.content)

    if model_pipeline is None:
        label = "Spam" if "http" in input_data.content.lower() else "Ham"
    else:
        input_df = pd.DataFrame([{
            'clean_body': clean_text, 
            'has_url': features[0],
            'has_dangerous_file': features[1], 
            'has_common_file': features[2],
            'has_other_file': features[3],
            'has_html': features[4]
        }])

        prediction = int(model_pipeline.predict(input_df)[0])
        label = "Spam" if prediction == 1 else "Ham"

    result = {
        "id": str(uuid.uuid4()), 
        "username": input_data.username,
        "prediction": label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "details": {
            "url": bool(features[0]),
            "file": bool(features[1]),
            "html": bool(features[4])
        },
        "content_preview": input_data.content[:100]
    }

    history = load_json(HISTORY_FILE)
    history.append(result)
    save_json(HISTORY_FILE, history)

    return {"status": "success", "data": result}

# --- 7. HISTORY ---

@app.get("/history")
async def get_history(username: str, role: str):
    history = load_json(HISTORY_FILE)
    if role == "admin":
        return history
    
    user_history = []
    for item in history:
        if item.get("username") == username:
            user_history.append(item)
    return user_history

# 🔥 FIX: DELETE ALL
@app.delete("/history")
async def delete_all_history(username: str, role: str):
    history = load_json(HISTORY_FILE)

    if role == "admin":
        save_json(HISTORY_FILE, [])
        return {"status": "success"}

    new_history = []
    for item in history:
        if item.get("username") != username:
            new_history.append(item)

    save_json(HISTORY_FILE, new_history)

    return {"status": "success"}

# 🔥 FIX: DELETE ONE
@app.delete("/history/{item_id}")
async def delete_history(item_id: str, username: str, role: str):
    history = load_json(HISTORY_FILE)
    new_history = []
    item_found = False
    
    for item in history:
        if item["id"] == item_id:
            item_found = True
            if role != "admin" and item["username"] != username:
                raise HTTPException(status_code=403)
            continue 
        new_history.append(item)
    
    if not item_found:
        raise HTTPException(status_code=404)
        
    save_json(HISTORY_FILE, new_history)
    return {"status": "success"}

@app.get("/stats")
async def get_stats():
    history = load_json(HISTORY_FILE)

    total = len(history)

    spam = sum(1 for item in history if item["prediction"].lower() == "spam")
    ham = sum(1 for item in history if item["prediction"].lower() == "ham")

    # 🔥 1. TIMELINE THEO GIỜ
    timeline_dict = {}

    for item in history:
        try:
            hour = item["timestamp"].split(" ")[1].split(":")[0]  # HH
        except:
            continue  # tránh crash nếu data lỗi

        if hour not in timeline_dict:
            timeline_dict[hour] = {"spam": 0, "ham": 0}

        if item["prediction"].lower() == "spam":
            timeline_dict[hour]["spam"] += 1
        else:
            timeline_dict[hour]["ham"] += 1

    timeline = [
        {
            "date": f"{int(h):02d}:00",
            "spam": v["spam"],
            "ham": v["ham"]
        }
        for h, v in sorted(timeline_dict.items(), key=lambda x: int(x[0]))
    ]

    # 🔥 2. SPAM MALWARE ANALYSIS
    spam_malware = 0
    spam_clean = 0

    for item in history:
        if item["prediction"].lower() == "spam":
            details = item.get("details", {})

            if details.get("url") or details.get("file"):
                spam_malware += 1
            else:
                spam_clean += 1

    # 🔥 3. RECENT
    recent = list(reversed(history[-5:]))

    return {
        "total": int(total),
        "spam": int(spam),
        "ham": int(ham),
        "spam_malware": int(spam_malware),
        "spam_clean": int(spam_clean),
        "timeline": timeline,
        "recent": recent
    }