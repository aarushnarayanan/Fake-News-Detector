import sys
import os
import re
import hashlib
import json
import datetime
import requests
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session
import trafilatura

# Add project root to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.predict import FakeNewsPredictor
from backend.database import get_db, init_db, Prediction, Feedback

# Simple in-memory cache for scraping results
# Format: {url: {"data": ScrapeResponse, "timestamp": datetime.datetime}}
SCRAPE_CACHE = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("Application starting: Initializing shared resources...")
    init_db()
    # Initialize the predictor once here
    app.state.predictor = FakeNewsPredictor()
    yield
    # --- SHUTDOWN ---
    print("Application shutting down...")

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Fake News Detector API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS config
origins = [
    "http://localhost:5173", # Vite default
    "http://localhost:3000",
    "*", # For development, tighten in prod
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---

class ScrapeRequest(BaseModel):
    url: str = Field(..., title="Article URL")

class ScrapeResponse(BaseModel):
    title: str
    text: str

class AnalyzeRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=500, title="News Title")
    text: str = Field(..., max_length=3000, title="News Body")

class EvidenceItem(BaseModel):
    text: str
    score: float

class AnalyzeResponse(BaseModel):
    id: int
    label: str
    probability: float
    overall_prediction: float
    highlighted_text: str
    evidence: List[EvidenceItem]

class FeedbackRequest(BaseModel):
    prediction_id: int
    user_label: str # "REAL" / "FAKE"
    comments: Optional[str] = None

class HistoryItem(BaseModel):
    id: int
    title: Optional[str] = None
    text_preview: str
    label: str
    probability: float
    created_at: str

# --- Helpers ---

def sanitize_input(text: str) -> str:
    # Remove HTML tags
    clean = re.sub(r'<.*?>', '', text)
    # Basic URL "scraping" placeholder - if user sends a URL, we might want to fetch it.
    # For now, we assume direct text input as per V1 requirements.
    # If the user input is ONLY a URL, we could implement a scraper.
    # Requirement says "paste a url or a body of text".
    # I will add a simple note that URL fetching is NOT implemented in this turn unless simple text.
    return clean.strip()

# --- Endpoints ---

@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("5/minute") # Rate limit: 5 requests per minute per IP
def analyze_text(request: Request, payload: AnalyzeRequest, db: Session = Depends(get_db)):
    predictor = request.app.state.predictor
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    clean_title = sanitize_input(payload.title) if payload.title else ""
    clean_body = sanitize_input(payload.text)
    
    if not clean_body:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    if len(clean_body) > 3000:
        raise HTTPException(status_code=400, detail="Text is too long (limit: 3000 characters).")

    # Run Prediction
    try:
        # Pass both title and body to the model
        result = predictor.predict(body=clean_body, title=clean_title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    # Save to DB
    # Combine title and body for hash to ensure uniqueness
    combined_text = f"{clean_title}\n{clean_body}"
    input_hash = hashlib.sha1(combined_text.encode("utf-8")).hexdigest()
    
    db_prediction = Prediction(
        input_text_hash=input_hash,
        title=clean_title if clean_title else None,
        input_text_preview=clean_body[:100],
        label=result['label'],
        probability=result['probability']
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)

    return AnalyzeResponse(
        id=db_prediction.id,
        label=result['label'],
        probability=result['probability'],
        overall_prediction=result['overall_prediction'],
        highlighted_text=result['highlighted_text'],
        evidence=[EvidenceItem(text=e['text'], score=e['score']) for e in result['evidence']]
    )

@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest, db: Session = Depends(get_db)):
    feedback = Feedback(
        prediction_id=payload.prediction_id,
        user_label=payload.user_label,
        comments=payload.comments
    )
    db.add(feedback)
    db.commit()
    return {"status": "success", "message": "Feedback received"}

@app.get("/history", response_model=List[HistoryItem])
def get_history(limit: int = 10, db: Session = Depends(get_db)):
    predictions = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(limit).all()
    return [
        HistoryItem(
            id=p.id,
            title=p.title,
            text_preview=p.input_text_preview,
            label=p.label,
            probability=p.probability,
            created_at=p.created_at.strftime("%Y-%m-%d %H:%M")
        ) for p in predictions
    ]

@app.post("/scrape", response_model=ScrapeResponse)
@limiter.limit("5/minute")
def scrape_article(request: Request, payload: ScrapeRequest):
    url = payload.url.strip()
    
    # Check Cache first
    now = datetime.datetime.now()
    if url in SCRAPE_CACHE:
        cache_entry = SCRAPE_CACHE[url]
        if (now - cache_entry["timestamp"]).total_seconds() < CACHE_TTL_SECONDS:
            print(f"Returning cached results for: {url}")
            return cache_entry["data"]

    try:
        # User-Agent to impersonate a real browser (Chrome on Mac)
        browser_user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        
        # Use requests directly for better control over headers/timeouts
        response = requests.get(url, headers={"User-Agent": browser_user_agent}, timeout=10)
        
        if not response.ok:
            raise HTTPException(status_code=400, detail=f"Failed to reach the URL (Status: {response.status_code}). The site might be blocking the request.")
        
        downloaded = response.text
        
        # Extract content using trafilatura with metadata
        result = trafilatura.extract(downloaded, include_comments=False, include_tables=False, output_format='json')
        
        if not result:
            raise HTTPException(status_code=400, detail="The page loaded, but no article content could be found.")
            
        data = json.loads(result)
        title = data.get('title', 'Unknown Title')
        text = data.get('text', '')

        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from the article body.")
            
        response_data = ScrapeResponse(title=title, text=text)
        
        # Update Cache
        SCRAPE_CACHE[url] = {
            "data": response_data,
            "timestamp": now
        }
        
        return response_data
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Scraping error: {str(e)}")

@app.get("/")
def health_check(request: Request):
    return {"status": "ok", "model_loaded": getattr(request.app.state, 'predictor', None) is not None}
