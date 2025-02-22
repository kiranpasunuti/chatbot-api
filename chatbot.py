from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os

# Configure API key
genai.configure(api_key='AIzaSyC0uefxG-FBjr83Jj-RWGwSFUuvz_59gCk')

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    generation_config=generation_config,
    safety_settings=safety_settings
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use DATABASE_URL from environment or default to Render PostgreSQL URL
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://chatdb_qlfr_user:fda5wxAJpSWW8qvWTGzjNLcFuajHc3Yu@dpg-cuspqqrtq21c73b7hcrg-a.singapore-postgres.render.com/chatdb_qlfr")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define table model
class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(String, nullable=False)
    response = Column(String, nullable=False)

# Create tables only if they don't exist
def create_tables():
    from sqlalchemy import inspect
    inspector = inspect(engine)
    if "chat_history" not in inspector.get_table_names():
        Base.metadata.create_all(bind=engine)

# Ensure the table is created on startup
create_tables()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    id: int
    prompt: str
    response: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    response = model.generate_content(request.message)
    original_reply = response.text

    if original_reply:
        chat_entry = ChatHistory(prompt=request.message, response=original_reply)
        db.add(chat_entry)
        db.commit()
        db.refresh(chat_entry)

    return {"code": 200, "data": original_reply}

@app.get("/chat/history", response_model=list[ChatResponse])
async def get_chat_history(db: Session = Depends(get_db)):
    chat_entries = db.query(ChatHistory).all()
    return chat_entries
