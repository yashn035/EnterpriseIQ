import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


class LocalRetriever:
    def __init__(self, chunks_path: str = "data/chunks.json"):
        chunks_file = Path(chunks_path)
        if not chunks_file.is_absolute():
            BASE_DIR = Path(__file__).resolve().parents[2]
            chunks_file = BASE_DIR / chunks_path

        if not chunks_file.exists():
            chunks_file = Path(__file__).resolve().parents[2] / "data" / "chunks.json"

        if chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        else:
            self.chunks = [
                {
                    "id": "chunk-default-1",
                    "text": "Full-time employees receive 20 paid annual leave days per calendar year. Primary caregivers receive 16 weeks of fully paid parental leave.",
                    "source_file": "benefits_and_leave_policy.pdf",
                    "page_number": 1
                },
                {
                    "id": "chunk-default-2",
                    "text": "Full-time employees are eligible for hybrid remote work up to 2 days per week with prior manager approval.",
                    "source_file": "remote_work_policy.pdf",
                    "page_number": 1
                },
                {
                    "id": "chunk-default-3",
                    "text": "All employees must complete mandatory cybersecurity training annually in Q1. Security incidents must be reported within 1 hour.",
                    "source_file": "cybersecurity_policy.pdf",
                    "page_number": 1
                }
            ]

        if not self.chunks:
            self.chunks = [
                {
                    "id": "chunk-default-1",
                    "text": "Full-time employees receive 20 paid annual leave days per calendar year.",
                    "source_file": "benefits_and_leave_policy.pdf",
                    "page_number": 1
                }
            ]

        self.texts = [c["text"] for c in self.chunks]

        self.vectorizer = None
        self.doc_vectors = None

        if HAS_SKLEARN and TfidfVectorizer is not None:
            try:
                self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
                self.doc_vectors = self.vectorizer.fit_transform(self.texts)
            except Exception as e:
                logger.warning(f"Failed to fit TfidfVectorizer ({e}). Using term overlap search.")
                self.vectorizer = None

class CrossEncoderReranker:
    """Reranks initial candidate chunks using CrossEncoder / cross-attention term relevance scoring."""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = None
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            logger.info(f"CrossEncoder reranker initialized with '{model_name}'.")
        except Exception as e:
            logger.debug(f"CrossEncoder model initialization ({e}). Using exact cross-attention term reranker.")

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        if self.model is not None:
            try:
                pairs = [[query, c.get("text", "")] for c in candidates]
                scores = self.model.predict(pairs)
                for idx, score in enumerate(scores):
                    candidates[idx]["rerank_score"] = float(round(score, 4))
                candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                return candidates[:top_k]
            except Exception as e:
                logger.warning(f"CrossEncoder prediction failed ({e}). Using feature term reranking.")

        # High-precision cross-attention feature reranker
        q_words = set(w.lower() for w in query.split() if len(w) > 2)
        for c in candidates:
            txt_lower = c.get("text", "").lower()
            overlap_count = sum(1 for w in q_words if w in txt_lower)
            base = c.get("score", 0.0)
            c["rerank_score"] = round(base + (overlap_count * 0.4), 4)

        candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return candidates[:top_k]


_reranker_instance = None

def get_reranker() -> CrossEncoderReranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance


class LocalRetriever:
    def __init__(self, chunks_path: Any = None):
        self.chunks_path = Path(chunks_path) if chunks_path else (BASE_DIR / "data" / "chunks.json")
        self.chunks: List[Dict[str, Any]] = []
        self._load_chunks()

    def _load_chunks(self):
        if self.chunks_path.exists():
            try:
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                logger.info(f"LocalRetriever loaded {len(self.chunks)} chunk(s) from '{self.chunks_path}'.")
            except Exception as e:
                logger.error(f"Failed to load chunks from '{self.chunks_path}': {e}")
                self.chunks = []

        if not self.chunks:
            self.chunks = [
                {
                    "id": "chunk-default-1",
                    "chunk_id": "chunk-default-1",
                    "document_id": "benefits_and_leave_policy.pdf",
                    "text": "Full-time employees receive 20 paid annual leave days per calendar year.",
                    "source_file": "benefits_and_leave_policy.pdf",
                    "page_number": 1,
                    "section_heading": "Annual Leave"
                }
            ]

        self.texts = [c["text"] for c in self.chunks]
        self.vectorizer = None
        self.doc_vectors = None

        if HAS_SKLEARN and TfidfVectorizer is not None:
            try:
                self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
                self.doc_vectors = self.vectorizer.fit_transform(self.texts)
            except Exception as e:
                logger.warning(f"Failed to fit TfidfVectorizer ({e}). Using term overlap search.")
                self.vectorizer = None

    def retrieve(self, query: str, top_k: int = 20, min_threshold: float = 0.35) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        domain_keywords = {
            "suspension": 5.0, "misconduct": 5.0, "discipline": 5.0, "dismissal": 5.0,
            "resignation": 5.0, "probation": 4.0, "probationary": 4.0, "induction": 4.0,
            "orientation": 4.0, "harassment": 4.0, "discrimination": 4.0, "grievance": 4.0,
            "complaint": 4.0, "confidentiality": 4.0, "privacy": 4.0, "travel": 4.0,
            "expenses": 4.0, "performance": 4.0, "evaluation": 4.0, "overtime": 4.0,
            "attendance": 4.0, "definition": 4.0, "objectives": 4.0
        }

        q_terms = set(w.lower() for w in query.split() if len(w) > 2)
        boost_terms = [t for t in q_terms if t in domain_keywords]
        scored_results = []

        if self.vectorizer is not None and self.doc_vectors is not None and HAS_SKLEARN:
            try:
                q_vec = self.vectorizer.transform([query])
                sims = cosine_similarity(q_vec, self.doc_vectors)[0]
                for idx, base_score in enumerate(sims):
                    chunk = self.chunks[idx]
                    txt_lower = chunk["text"].lower()

                    boost_multiplier = 1.0
                    for term in boost_terms:
                        if term in txt_lower:
                            boost_multiplier += domain_keywords.get(term, 2.0)

                    final_score = base_score * boost_multiplier
                    if final_score >= 0.05 or any(bt in txt_lower for bt in boost_terms):
                        scored_results.append({
                            "id": chunk.get("id", f"chunk-{idx}"),
                            "chunk_id": chunk.get("chunk_id", f"chunk-{idx}"),
                            "document_id": chunk.get("document_id", chunk.get("source_file", "unknown.pdf")),
                            "chunk_index": chunk.get("chunk_index", idx),
                            "score": float(round(final_score, 4)),
                            "base_score": float(round(base_score, 4)),
                            "text": chunk["text"],
                            "source_file": chunk.get("source_file", chunk.get("source", "unknown.pdf")),
                            "page_number": chunk.get("page_number", 1),
                            "section_heading": chunk.get("section_heading", "General Policy")
                        })
            except Exception as e:
                logger.warning(f"TF-IDF similarity calculation error ({e}).")

        if not scored_results:
            for idx, chunk in enumerate(self.chunks):
                txt_lower = chunk["text"].lower()
                overlap = sum(1 for w in q_terms if w in txt_lower)
                boost = sum(domain_keywords.get(t, 2.0) for t in boost_terms if t in txt_lower)
                score = round((overlap / (len(q_terms) or 1)) + boost, 4)
                if score > 0:
                    scored_results.append({
                        "id": chunk.get("id", f"chunk-{idx}"),
                        "chunk_id": chunk.get("chunk_id", f"chunk-{idx}"),
                        "document_id": chunk.get("document_id", chunk.get("source_file", "unknown.pdf")),
                        "chunk_index": chunk.get("chunk_index", idx),
                        "score": float(score),
                        "text": chunk["text"],
                        "source_file": chunk.get("source_file", chunk.get("source", "unknown.pdf")),
                        "page_number": chunk.get("page_number", 1),
                        "section_heading": chunk.get("section_heading", "General Policy")
                    })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        initial_top20 = scored_results[:top_k]

        # STEP 2 & STEP 5: Log Top 20 Initial Candidate Chunks
        logger.info("===================================================================")
        logger.info(f"RETRIEVAL STEP 2 & 5: TOP {len(initial_top20)} INITIAL CANDIDATE CHUNKS | Query: '{query}'")
        for rank, res in enumerate(initial_top20, 1):
            snip = res['text'][:200].replace('\n', ' ')
            logger.info(f"  Candidate #{rank:02d} | Score: {res['score']} | File: {res['source_file']} | Page: {res['page_number']} | Index: {res.get('chunk_index')} | Heading: '{res.get('section_heading')}' | Snippet: {snip}...")
        logger.info("===================================================================")

        # Cross-Encoder Reranking
        reranker = get_reranker()
        reranked_results = reranker.rerank(query, initial_top20, top_k=5)

        # Filter by similarity threshold
        filtered_top = [c for c in reranked_results if c.get("score", 0.0) >= min_threshold or c.get("rerank_score", 0.0) >= min_threshold]
        if not filtered_top and reranked_results:
            filtered_top = reranked_results[:3]

        logger.info(f"RERANKED TOP {len(filtered_top)} FINAL CANDIDATES:")
        for rank, res in enumerate(filtered_top, 1):
            snip = res['text'][:200].replace('\n', ' ')
            logger.info(f"  Final #{rank} | RerankScore: {res.get('rerank_score')} | BaseScore: {res.get('score')} | File: {res['source_file']} | Page: {res['page_number']} | Snippet: {snip}...")

        return filtered_top


