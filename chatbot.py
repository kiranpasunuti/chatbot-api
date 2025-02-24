from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from sqlalchemy import create_engine, Column, Integer, String,Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os

# ✅ Install MySQL Driver (pymysql)
try:
    import pymysql
except ModuleNotFoundError:
    os.system("pip install pymysql")

# ✅ Configure API key for Gemini AI
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

# ✅ FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REMOTE_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://chatdb_qlfr_user:fda5wxAJpSWW8qvWTGzjNLcFuajHc3Yu@dpg-cuspqqrtq21c73b7hcrg-a.singapore-postgres.render.com/chatdb_qlfr"
)

LOCAL_DATABASE_URL = os.getenv(
    "LOCAL_DATABASE_URL",
    "mysql+pymysql://root:12345@localhost:3306/chatbot"
) 

remote_engine = create_engine(REMOTE_DATABASE_URL, pool_pre_ping=True)
local_engine = create_engine(LOCAL_DATABASE_URL, pool_pre_ping=True)
RemoteSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=remote_engine)
LocalSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=local_engine)

Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)  # Change from String to Text

# ✅ Create tables in both databases
def create_tables():
    from sqlalchemy import inspect
    for engine in [remote_engine, local_engine]:
        inspector = inspect(engine)
        if "chat_history" not in inspector.get_table_names():
            Base.metadata.create_all(bind=engine)

# Ensure table creation on startup
create_tables()

# ✅ Request model
class ChatRequest(BaseModel):
    message: str

# ✅ Dependency to get DB session
def get_remote_db():
    db = RemoteSessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_local_db():
    db = LocalSessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ Function to format response properly
def format_response(response_text: str) -> str:
    """Formats response to ensure proper markdown structure like ChatGPT."""
    lines = response_text.split("\n")
    formatted_lines = []
    
    for line in lines:
        if line.startswith("#"):  # Headings on a separate line
            formatted_lines.append(f"\n{line}\n")
        elif line.startswith("//") or line.startswith("#") or line.startswith("--"):  # Comments on a new line
            formatted_lines.append(f"\n{line}\n")
        elif "```" in line:  # Ensure code blocks are properly formatted
            formatted_lines.append(f"\n{line}\n")
        else:
            formatted_lines.append(line)

    return "\n".join(formatted_lines)

# ✅ Chat endpoint (Store data in both MySQL & PostgreSQL)
@app.post("/chat")
async def chat(request: ChatRequest, remote_db: Session = Depends(get_remote_db), local_db: Session = Depends(get_local_db)):
    response = model.generate_content(request.message)
    original_reply = response.text

    if original_reply:
        # ✅ Create separate objects for each database
        remote_chat_entry = ChatHistory(prompt=request.message, response=original_reply)
        local_chat_entry = ChatHistory(prompt=request.message, response=original_reply)

        # ✅ Store in Remote PostgreSQL
        remote_db.add(remote_chat_entry)
        remote_db.commit()

        # ✅ Store in Local MySQL
        local_db.add(local_chat_entry)
        local_db.commit()

    # ✅ Format response for proper display
    formatted_reply = format_response(original_reply)

    return JSONResponse(content={"code": 200, "data": formatted_reply})

# ✅ Fetch chat history (from both MySQL & PostgreSQL)
@app.get("/chat/history")
async def get_chat_history(remote_db: Session = Depends(get_remote_db), local_db: Session = Depends(get_local_db)):
    remote_history = remote_db.query(ChatHistory).all()
    local_history = local_db.query(ChatHistory).all()

    return {
        "code": 200,
        "remote_history": [{"prompt": h.prompt, "response": h.response} for h in remote_history],
        "local_history": [{"prompt": h.prompt, "response": h.response} for h in local_history],
    }

# ✅ Ensure the correct port for Render
port = int(os.getenv("PORT", 10000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
