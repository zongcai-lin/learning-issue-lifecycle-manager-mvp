from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAIError
from pydantic import ValidationError

from backend.analyzer import (
    AnalyzerConfigurationError,
    AnalyzerResponseError,
    InvalidConversationError,
    NoGoalThreadsError,
    analyze_learning_conversation,
)
from backend.schemas import AnalysisResult, AnalyzeRequest


app = FastAPI(title="Learning Issue Lifecycle Manager MVP")

# Local-development CORS configuration. Restrict this list before deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5000",
        "http://localhost:5000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/")
def root():
    return {"message": "Learning Issue MVP backend is running"}


@app.post("/analyze", response_model=AnalysisResult)
def analyze_conversation(request: AnalyzeRequest) -> AnalysisResult:
    try:
        return analyze_learning_conversation(request.conversation_text)
    except InvalidConversationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NoGoalThreadsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AnalyzerConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (AnalyzerResponseError, ValidationError) as exc:
        raise HTTPException(
            status_code=502,
            detail="The analyzer returned an invalid structured response",
        ) from exc
    except OpenAIError as exc:
        raise HTTPException(
            status_code=502,
            detail="The OpenAI API request failed. Please try again later.",
        ) from exc
