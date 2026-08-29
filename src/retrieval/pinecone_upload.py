import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv

# Calculate BASE_DIR dynamically from file location
BASE_DIR = Path(__file__).resolve().parents[2]
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path if env_path.exists() else None)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_unit_vector(text: str, dimension: int = 384) -> List[float]:
    """Generates a normalized 384-dim unit vector based on SHA256 bytes for offline search."""
    import hashlib
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vec = [(b / 255.0) for b in digest]
    vec = (vec * ((dimension // len(vec)) + 1))[:dimension]
    norm = sum(x*x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


def upload_chunks_to_pinecone(
    chunks_file: Any = None,
    batch_size: int = 100,
    target_dimension: int = 384,
) -> int:
    """
    Mock Pinecone escape hatch: saves vector embeddings to data/vectors.npy
    and metadata to data/metadata.pkl for offline local retrieval.
    """
    chunks_path = Path(chunks_file) if chunks_file else (BASE_DIR / "data" / "chunks.json")
    if not chunks_path.exists():
        logger.error(f"Chunks file '{chunks_path}' not found.")
        return 0

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)

    if not chunks:
        logger.warning(f"No chunks found in '{chunks_path}'.")
        return 0

    vectors_list = []
    metadata_list = []

    for chunk in chunks:
        vec = generate_unit_vector(chunk.get("text", ""), target_dimension)
        vectors_list.append(vec)
        metadata_list.append({
            "id": chunk.get("id"),
            "source_file": chunk.get("source_file", "sample_policy.pdf"),
            "page_number": chunk.get("page_number", 1),
            "text": chunk.get("text", "")
        })

    vectors_np = np.array(vectors_list, dtype=np.float32)
    vectors_file = BASE_DIR / "data" / "vectors.npy"
    metadata_file = BASE_DIR / "data" / "metadata.pkl"

    vectors_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_file, vectors_np)

    with open(metadata_file, "wb") as f:
        pickle.dump(metadata_list, f)

    logger.info(f"Saved {len(chunks)} local vector embeddings to '{vectors_file.name}' and metadata to '{metadata_file.name}'.")
    return len(chunks)


if __name__ == "__main__":
    upload_chunks_to_pinecone()
