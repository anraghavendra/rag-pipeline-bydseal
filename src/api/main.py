import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import our RAG pipeline
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="A RAG (Retrieval-Augmented Generation) pipeline for answering questions about BYD Seal",
    version="1.0.0"
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
try:
    pipeline = RAGPipeline()
except Exception as e:
    pipeline = None

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask", min_length=1, max_length=500)

class Citation(BaseModel):
    source: str
    doc_id: str
    chunk_id: str
    title: Optional[str] = None
    channel: Optional[str] = None
    views: Optional[str] = None
    subscribers: Optional[str] = None
    type: str = ""

class QuestionResponse(BaseModel):
    answer: str
    status: str
    citations: List[Citation]


# API Endpoints

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an answer from the RAG pipeline"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    try:
        # Process the question through our RAG pipeline
        result = pipeline.ask(request.question)
        
        # Convert citations to proper format
        citations = []
        for citation in result['citations']:
            # Only include fields that have actual values
            citation_data = {
                "source": citation['source'],
                "doc_id": citation['doc_id'],
                "chunk_id": citation['chunk_id'],
                "type": citation.get('type', '')
            }
            
            # Only add optional fields if they have values
            if citation.get('title'):
                citation_data["title"] = citation['title']
            if citation.get('channel'):
                citation_data["channel"] = citation['channel']
            if citation.get('views'):
                citation_data["views"] = citation['views']
            if citation.get('subscribers'):
                citation_data["subscribers"] = citation['subscribers']
            
            citations.append(Citation(**citation_data))
        
        return QuestionResponse(
            answer=result['answer'],
            status=result['status'],
            citations=citations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")



@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Pipeline API is running",
        "docs": "/docs",
        "health": "/health",
        "ask": "/ask"
    }


@app.get("/health")
async def health():
    """Cheap health check that also validates the model can be called."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    try:
        # Minimal, low-cost model ping
        result = pipeline._analyze_question_and_plan_search("ping")  # noqa: SLF001
        return {"status": "ok", "model": pipeline.model, "llm": result in ["refuse", "facts_only", "external_safe"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
