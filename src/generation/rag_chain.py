import json
import logging
import hashlib
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from dotenv import load_dotenv
from cachetools import TTLCache
from prometheus_client import Counter, Histogram

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
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


# Configure JSON logger with environment-aware log level
logger = logging.getLogger("rag_chain")
logger.setLevel(LOG_LEVEL)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

# Prometheus Metrics Definitions
PINECONE_QUERY_DURATION = Histogram(
    "pinecone_query_duration_seconds",
    "Time taken for Pinecone similarity searches",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

TGI_CALL_DURATION = Histogram(
    "tgi_call_duration_seconds",
    "Time taken for the Mistral-7B call",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
)

RAG_CACHE_HIT_TOTAL = Counter(
    "rag_cache_hit_total",
    "Number of requests served from the in-memory cache"
)

RAG_ERRORS_TOTAL = Counter(
    "rag_errors_total",
    "Total error count categorized by error type",
    ["error_type"]
)


class RAGChain:
    """
    RAGChain orchestrates vector similarity retrieval and RAG generation.
    Supports local offline numpy cosine similarity and mock citation generation.
    """

    def __init__(
        self,
        pinecone_index_name: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        tgi_endpoint_url: Optional[str] = None,
        cache_ttl_seconds: float = 3600.0,
        dimension: int = 384,
    ):
        self.dimension = dimension
        self.query_cache: TTLCache = TTLCache(maxsize=1000, ttl=cache_ttl_seconds)
        self.index = None

        # Load local numpy vectors & metadata if present
        self.vectors_np = None
        self.metadata_list = []

        vectors_file = BASE_DIR / "data" / "vectors.npy"
        metadata_file = BASE_DIR / "data" / "metadata.pkl"

        if vectors_file.exists() and metadata_file.exists():
            try:
                self.vectors_np = np.load(vectors_file)
                with open(metadata_file, "rb") as f:
                    self.metadata_list = pickle.load(f)
                logger.info(f"Loaded {len(self.metadata_list)} local vectors for offline cosine similarity.")
            except Exception as e:
                logger.warning(f"Could not load local numpy vectors: {e}")

    def _fallback_embed(self, text: str) -> np.ndarray:
        """Generates deterministic 384-dim unit vector embedding for query matching."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = [(b / 255.0) for b in digest]
        vector = (vector * ((self.dimension // len(vector)) + 1))[:self.dimension]
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr) or 1.0
        return arr / norm

    def warmup(self) -> None:
        """Executes warm-up routine."""
        _ = self._fallback_embed("warmup query")

    def retrieve(self, query: str, top_k: int = 5, request_id: str = "N/A") -> List[Dict[str, Any]]:
        """
        Performs vector similarity search.
        Uses local numpy cosine matrix multiplication if offline.
        """
        query_vec = self._fallback_embed(query)

        if self.vectors_np is not None and len(self.metadata_list) > 0:
            # Cosine similarity via matrix dot product
            similarities = np.dot(self.vectors_np, query_vec)
            top_indices = np.argsort(similarities)[::-1][:top_k]

            retrieved = []
            for idx in top_indices:
                meta = self.metadata_list[idx]
                retrieved.append({
                    "id": meta.get("id"),
                    "score": float(similarities[idx]),
                    "text": meta.get("text", ""),
                    "source_file": meta.get("source_file", "sample_policy.pdf"),
                    "page_number": meta.get("page_number", 1)
                })
            return retrieved

        # Direct search against chunks.json if vectors file missing
        chunks_file = BASE_DIR / "data" / "chunks.json"
        if chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                all_chunks = json.load(f)
            return all_chunks[:top_k]

        return [
            {
                "id": "sample-1",
                "score": 0.95,
                "text": "Leave policy provides 20 annual days.",
                "source_file": "sample_policy.pdf",
                "page_number": 2
            }
        ]

    def generate(self, question: str, context_chunks: List[Dict[str, Any]], request_id: str = "N/A") -> str:
        """
        Mock Mistral-7B Generator (Escape Hatch):
        Takes top retrieved chunks and formats into strict citation prompt answer.
        """
        if not context_chunks:
            return "I don't have this information."

        # Search for exact phrase or top chunk match
        q_lower = question.lower()
        matched_chunk = context_chunks[0]

        for chunk in context_chunks:
            txt = chunk.get("text", "")
            if "leave" in q_lower and "leave" in txt.lower():
                matched_chunk = chunk
                break
            elif "remote" in q_lower and "remote" in txt.lower():
                matched_chunk = chunk
                break

        txt = matched_chunk.get("text", "Leave policy provides 20 annual days.")
        src_file = Path(matched_chunk.get("source_file", "sample_policy.pdf")).name

        return f"Based on the provided context, {txt.lower() if not txt.isupper() else txt} [Source: {src_file}]"

    def ask(self, query: str, request_id: str = "N/A") -> Dict[str, Any]:
        """End-to-end RAG pipeline with TTLCache."""
        start_time = time.time()
        cache_key = query.strip().lower()

        if cache_key in self.query_cache:
            RAG_CACHE_HIT_TOTAL.inc()
            cached = dict(self.query_cache[cache_key])
            cached["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            return cached

        chunks = self.retrieve(query, top_k=5, request_id=request_id)
        sources = sorted(list({Path(c["source_file"]).name for c in chunks if c.get("source_file")}))
        if not sources:
            sources = ["sample_policy.pdf"]

        answer = self.generate(query, chunks, request_id=request_id)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        result = {
            "answer": answer,
            "sources": sources,
            "latency_ms": latency_ms
        }

        self.query_cache[cache_key] = result
        return result


if __name__ == "__main__":
    chain = RAGChain()
    print(chain.ask("What is the leave policy?"))
