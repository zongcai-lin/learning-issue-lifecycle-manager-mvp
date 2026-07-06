from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from analyzer import analyze_learning_conversation


app = FastAPI()

# 解决跨域问题（5000端口访问8000端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    conversation_text: str


@app.get("/")
def root():
    return {"message": "Learning Issue MVP backend is running"}


@app.post("/analyze")
def analyze_conversation(request: AnalyzeRequest):
    result = analyze_learning_conversation(request.conversation_text)
    return result