from fastapi import FastAPI, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
import shutil

# Configure Gemini API
API_KEY = 'AIzaSyC0uefxG-FBjr83Jj-RWGwSFUuvz_59gCk'
genai.configure(api_key=API_KEY)

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

# Initialize models
gemini_text_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    generation_config=generation_config,
    safety_settings=safety_settings
)
gemini_vision_model = genai.GenerativeModel("gemini-1.5-pro-vision")

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
LOCAL_DATABASE_URL = "sqlite:///chatbot.db"
local_engine = create_engine(LOCAL_DATABASE_URL, connect_args={"check_same_thread": False})
LocalSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=local_engine)
Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)

# Create tables
def create_tables():
    Base.metadata.create_all(bind=local_engine)
create_tables()

# Database dependency
def get_local_db():
    db = LocalSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Request model
class ChatRequest(BaseModel):
    message: str

# Chat endpoint (text)
@app.post("/chat")
async def chat(request: ChatRequest, local_db: Session = Depends(get_local_db)):
    response = gemini_text_model.generate_content(request.message)
    reply = response.text

    chat_entry = ChatHistory(prompt=request.message, response=reply)
    local_db.add(chat_entry)
    local_db.commit()

    return JSONResponse(content={"code": 200, "data": reply})

# Chatbot with Gemini Pro-Vision (image handling)
@app.post("/chatbot-images")
async def chatbot_images(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    with open(temp_path, "rb") as img_file:
        response = gemini_vision_model.generate_content([img_file])
        reply = response.text
    
    os.remove(temp_path)
    return JSONResponse(content={"code": 200, "data": reply})

# Chat history endpoint
@app.get("/chat/history")
async def get_chat_history(local_db: Session = Depends(get_local_db)):
    history = local_db.query(ChatHistory).all()
    return {"code": 200, "history": [{"prompt": h.prompt, "response": h.response} for h in history]}

# Run server
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
