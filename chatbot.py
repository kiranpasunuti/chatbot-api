from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import asyncio

# Configure Gemini AI API
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

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup (SQLite)
DATABASE_URL = "sqlite:///./chat_history1.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model for chat history
class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(String, nullable=False)
    response = Column(String, nullable=False)

Base.metadata.create_all(bind=engine)

# Request model
class ChatRequest(BaseModel):
    message: str

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to stream responses word by word
async def generate_response_stream(message: str, db: Session):
    response = model.generate_content(message)
    original_reply = response.text

    # Store response in the database
    if original_reply:
        chat_entry = ChatHistory(prompt=message, response=original_reply)
        db.add(chat_entry)
        db.commit()

    # Stream words one by one
    words = original_reply.split()
    for word in words:
        yield word + " "
        await asyncio.sleep(0.1)  # Adjust delay for smooth streaming

# Chatbot API endpoint with streaming response
@app.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    return StreamingResponse(generate_response_stream(request.message, db), media_type="text/event-stream")
