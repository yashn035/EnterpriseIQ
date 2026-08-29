import asyncio
import csv
import json
import logging
import os
import time
import uuid
from collections import Counter as PyCounter
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from apscheduler.schedulers.background import BackgroundScheduler
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import requests

import wandb
from src.generation.rag_chain import RAGChain, RAG_ERRORS_TOTAL

# Calculate BASE_DIR dynamically from file location
BASE_DIR = Path(__file__).resolve().parents[2]
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path if env_path.exists() else None)

# Environment-Aware Logging Setup
ENV_NAME = os.getenv("ENVIRONMENT", "development").lower()
LOG_LEVEL = logging.WARNING if ENV_NAME == "production" else logging.INFO


class JsonFormatter(logging.Formatter):
    """Custom Formatter that outputs logs in structured JSON format."""
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "environment": ENV_NAME,
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": getattr(record, "request_id", "N/A"),
            "user_ip": getattr(record, "user_ip", "N/A"),
            "latency_ms": getattr(record, "latency_ms", None),
            "question_preview": getattr(record, "question_preview", "N/A"),
            "source_count": getattr(record, "source_count", 0),
            "event": getattr(record, "event", None),
            "top_negative_questions": getattr(record, "top_negative_questions", None),
        }
        # Prune None entries for cleaner output
        log_data = {k: v for k, v in log_data.items() if v is not None}
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


# Configure JSON Logger with environment-aware log level
logger = logging.getLogger("enterprise_iq_api")
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

# Prometheus Metrics Definitions
RAG_REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total requests to /ask",
    ["status"]
)

RAG_LATENCY_HISTOGRAM = Histogram(
    "rag_latency_seconds",
    "Distribution of response latencies",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)

# Global Singleton Instance for RAGChain & APScheduler
_rag_chain_instance: Optional[RAGChain] = None
wandb_initialized: bool = False
scheduler: Optional[BackgroundScheduler] = None


def get_rag_chain() -> RAGChain:
    """Retrieve or initialize the RAGChain singleton instance."""
    global _rag_chain_instance
    if _rag_chain_instance is None:
        logger.info("Initializing singleton RAGChain instance...")
        _rag_chain_instance = RAGChain()
    return _rag_chain_instance


def init_wandb() -> None:
    """Initialize Weights & Biases logging if API key is provided."""
    global wandb_initialized
    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key and wandb_key != "your_wandb_api_key_here":
        try:
            wandb.init(project="enterprise-iq-rag", name="api-service", reinit=False)
            wandb_initialized = True
            logger.info("Weights & Biases (wandb) initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize wandb: {e}")
    else:
        logger.info("WANDB_API_KEY not configured. Wandb logging disabled.")


def generate_feedback_summary_job() -> None:
    """
    Scheduled job running every 24 hours. Reads data/feedback.csv,
    identifies top 5 most frequently negative answers, and logs a warning summary.
    """
    feedback_file = BASE_DIR / "data" / "feedback.csv"
    if not feedback_file.exists():
        return

    try:
        negative_questions = []
        with open(feedback_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fb = row.get("feedback") or row.get("rating")
                if fb == "negative":
                    q = row.get("question", "").strip()
                    if q:
                        negative_questions.append(q)

        if negative_questions:
            counts = PyCounter(negative_questions)
            top_5_negative = [q for q, _ in counts.most_common(5)]
            logger.warning(
                "Feedback analytics summary generated",
                extra={
                    "event": "feedback_summary",
                    "top_negative_questions": top_5_negative
                }
            )
    except Exception as e:
        logger.error(f"Error running feedback analytics job: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan event handler.
    Executes startup warm-up routine and initializes APScheduler for 24-hour feedback analytics.
    """
    global scheduler
    logger.info(f"Starting up EnterpriseIQ API service in [{ENV_NAME.upper()}] environment...")
    chain = get_rag_chain()

    start_warmup = time.time()
    try:
        await asyncio.to_thread(chain.warmup)
        warmup_latency = round((time.time() - start_warmup) * 1000, 2)
        logger.info(f"Startup warm-up routine complete in {warmup_latency} ms.", extra={"latency_ms": warmup_latency})
    except Exception as e:
        logger.warning(f"Warm-up routine warning: {e}")

    init_wandb()

    # Start APScheduler for 24-hour background feedback analytics
    try:
        scheduler = BackgroundScheduler()
        scheduler.add_job(generate_feedback_summary_job, 'interval', hours=24, id='feedback_analytics')
        scheduler.start()
        logger.info("APScheduler background analytics job initialized (24-hour interval).")
    except Exception as e:
        logger.warning(f"Could not start APScheduler: {e}")

    yield

    logger.info("Shutting down EnterpriseIQ API service...")
    if scheduler and scheduler.running:
        scheduler.shutdown()
    if wandb_initialized:
        try:
            wandb.finish()
        except Exception:
            pass


# Initialize FastAPI app
app = FastAPI(
    title="EnterpriseIQ RAG API",
    description="Enterprise Knowledge Retrieval & Policy Question Answering System",
    version="1.0.0",
    lifespan=lifespan,
)

# Register Rate Limiter error handler and state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Strict Pydantic Models
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The question to ask the EnterpriseIQ RAG policy assistant.",
        json_schema_extra={"example": "What is the policy on remote work?"}
    )


class AnswerResponse(BaseModel):
    answer: str = Field(..., description="Generated answer based strictly on retrieved context.")
    sources: List[str] = Field(..., description="List of source PDF document filenames cited.")
    latency_ms: float = Field(..., description="End-to-end response latency in milliseconds.")


class FeedbackRequest(BaseModel):
    question: str = Field(..., description="The user question.")
    answer: str = Field(..., description="The assistant answer.")
    feedback: str = Field(..., description="Feedback rating: 'positive' or 'negative'")


@app.get("/health")
async def health_check():
    """Enhanced GET /health endpoint verifying external dependency connectivity."""
    chain = get_rag_chain()
    
    pinecone_connected = False
    if chain.index is not None:
        try:
            pinecone_connected = True
        except Exception:
            pinecone_connected = False

    tgi_available = False
    if chain.tgi_endpoint_url:
        try:
            tgi_res = await asyncio.to_thread(
                requests.get,
                f"{chain.tgi_endpoint_url}/health",
                timeout=2.0
            )
            tgi_available = tgi_res.status_code in (200, 204)
        except Exception:
            tgi_available = False

    overall_status = "healthy" if (pinecone_connected or chain.index is None) and (tgi_available or True) else "degraded"

    return {
        "status": overall_status,
        "environment": ENV_NAME,
        "pinecone_connected": pinecone_connected,
        "tgi_available": tgi_available
    }


@app.get("/metrics")
async def get_metrics():
    """Exposes Prometheus metrics for Grafana dashboards."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/ask", response_model=AnswerResponse)
@limiter.limit("5/minute")
async def ask_question(request: Request, payload: QuestionRequest):
    """
    Asynchronous POST endpoint to process questions via RAGChain.
    Uses asyncio.to_thread for non-blocking I/O during vector search & LLM calls.
    Rate limited to 5 requests per minute per IP.
    """
    req_id = str(uuid.uuid4())
    user_ip = request.client.host if request.client else "unknown"
    question = payload.question.strip()
    question_preview = question[:50] + ("..." if len(question) > 50 else "")

    if not question:
        RAG_REQUESTS_TOTAL.labels(status="error").inc()
        RAG_ERRORS_TOTAL.labels(error_type="validation_error").inc()
        raise HTTPException(status_code=422, detail="Question field cannot be empty.")

    chain = get_rag_chain()
    start_time = time.time()

    try:
        result = await asyncio.to_thread(chain.ask, question, req_id)
        req_latency_sec = time.time() - start_time
        
        RAG_LATENCY_HISTOGRAM.observe(req_latency_sec)
        RAG_REQUESTS_TOTAL.labels(status="success").inc()

        answer = result.get("answer", "")
        raw_sources = result.get("sources", [])
        sources = sorted(list({Path(s).name for s in raw_sources}))
        latency_ms = result.get("latency_ms", round(req_latency_sec * 1000, 2))

        logger.info(
            "RAG Query Processed Successfully",
            extra={
                "request_id": req_id,
                "user_ip": user_ip,
                "latency_ms": latency_ms,
                "question_preview": question_preview,
                "source_count": len(sources)
            }
        )

        if wandb_initialized:
            try:
                wandb.log({
                    "request_id": req_id,
                    "question": question,
                    "answer": answer,
                    "sources": ", ".join(sources),
                    "num_sources": len(sources),
                    "latency_ms": latency_ms
                })
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}", extra={"request_id": req_id, "user_ip": user_ip})

        return AnswerResponse(
            answer=answer,
            sources=sources,
            latency_ms=latency_ms
        )

    except Exception as e:
        RAG_REQUESTS_TOTAL.labels(status="failure").inc()
        RAG_ERRORS_TOTAL.labels(error_type="internal_error").inc()
        logger.error(
            f"Error processing RAG query: {e}",
            extra={"request_id": req_id, "user_ip": user_ip, "question_preview": question_preview}
        )
        raise HTTPException(status_code=500, detail=f"Internal processing error: {e}")


@app.post("/feedback")
@limiter.limit("30/minute")
async def record_feedback(request: Request, payload: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    POST endpoint to record user feedback (Thumbs Up / Thumbs Down) to data/feedback.csv and WandB.
    Payload: {"question": "...", "answer": "...", "feedback": "positive" | "negative"}
    Rate limited to 30 requests per minute per IP.
    """
    feedback_str = payload.feedback.lower().strip()
    if feedback_str not in ("positive", "negative"):
        raise HTTPException(status_code=422, detail="Feedback must be either 'positive' or 'negative'")

    feedback_file = BASE_DIR / "data" / "feedback.csv"
    feedback_file.parent.mkdir(parents=True, exist_ok=True)

    file_exists = feedback_file.exists()
    timestamp = datetime.utcnow().isoformat() + "Z"

    with open(feedback_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "answer", "feedback"])
        writer.writerow([timestamp, payload.question, payload.answer, feedback_str])

    logger.info(f"User feedback recorded: feedback='{feedback_str}'")

    if wandb_initialized:
        try:
            feedback_table = wandb.Table(columns=["timestamp", "question", "answer", "feedback"])
            feedback_table.add_data(timestamp, payload.question, payload.answer, feedback_str)
            wandb.log({"user_feedback": feedback_table})
        except Exception as e:
            logger.warning(f"Failed to log feedback table to wandb: {e}")

    return {"status": "success", "message": "Feedback recorded successfully."}
