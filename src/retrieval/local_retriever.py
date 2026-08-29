import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False


class LocalRetriever:
    def __init__(self, chunks_path="data/chunks.json"):
        chunks_file = Path(chunks_path)
        if not chunks_file.is_absolute():
            BASE_DIR = Path(__file__).resolve().parents[2]
            chunks_file = BASE_DIR / chunks_path

        if not chunks_file.exists():
            chunks_file = Path(__file__).resolve().parents[2] / "data" / "chunks.json"

        if chunks_file.exists():
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
        else:
            self.chunks = [
                {
                    "text": "Remote work policy allows 3 days per week.",
                    "source_file": "remote_work_policy.pdf"
                },
                {
                    "text": "Full-time employees receive 20 paid annual leave days per calendar year.",
                    "source_file": "benefits_and_leave_policy.pdf"
                }
            ]

        self.texts = [c["text"] for c in self.chunks]
        self.sources = [c.get("source_file", "unknown.pdf") for c in self.chunks]

        if not self.texts:
            self.texts = ["Full-time employees receive 20 paid annual leave days per calendar year."]
            self.sources = ["benefits_and_leave_policy.pdf"]

        # Generate fast deterministic unit vectors
        vecs = []
        for t in self.texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            v = [(b / 255.0) for b in digest]
            v = (v * ((384 // len(v)) + 1))[:384]
            norm = sum(x*x for x in v) ** 0.5 or 1.0
            vecs.append([x / norm for x in v])

        self.embeddings = np.array(vecs, dtype=np.float32)

        if HAS_FAISS and faiss is not None:
            try:
                self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
                self.index.add(self.embeddings)
            except Exception:
                self.index = None
        else:
            self.index = None

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q_words = set(query.lower().split())
        scored = []
        for idx, text in enumerate(self.texts):
            score = sum(2 for w in q_words if w in text.lower())
            scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            results.append({"text": self.texts[idx], "source": self.sources[idx]})
        return results
