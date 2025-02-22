from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import markdown

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

# Chatbot API endpoint
@app.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    response = model.generate_content(request.message)
    original_reply = response.text

    # Ensure response is formatted correctly using Markdown
    formatted_reply = markdown.markdown(original_reply)

    # Store response in the database
    if original_reply:
        chat_entry = ChatHistory(prompt=request.message, response=formatted_reply)
        db.add(chat_entry)
        db.commit()

    return {"code": 200, "data": formatted_reply}
