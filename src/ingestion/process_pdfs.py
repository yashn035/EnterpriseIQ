import json
import logging
import hashlib
from pathlib import Path
import uuid
from typing import List, Dict, Any, Optional

from pypdf import PdfReader
from tqdm import tqdm
from langchain_experimental.text_splitter import SemanticChunker

# Calculate BASE_DIR dynamically from file location
BASE_DIR = Path(__file__).resolve().parents[2]

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Lazy loading helper for SentenceTransformer
HAS_SENTENCE_TRANSFORMERS: Optional[bool] = None
SentenceTransformer = None

def get_sentence_transformer_cls():
    global HAS_SENTENCE_TRANSFORMERS, SentenceTransformer
    if HAS_SENTENCE_TRANSFORMERS is False:
        return None
    if SentenceTransformer is not None:
        return SentenceTransformer
    try:
        import torch
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
        HAS_SENTENCE_TRANSFORMERS = True
        return ST
    except BaseException as e:
        logger.warning(f"SentenceTransformer / PyTorch unavailable ({e}). Using lightweight hash embedding engine.")
        HAS_SENTENCE_TRANSFORMERS = False
        SentenceTransformer = None
        return None


class SentenceTransformerEmbeddingsWrapper:
    """
    Wrapper around SentenceTransformer with fallback support if PyTorch DLLs are missing.
    Fully compatible with LangChain's SemanticChunker.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.dimension = dimension
        self.model = None
        if HAS_SENTENCE_TRANSFORMERS and SentenceTransformer is not None:
            try:
                logger.info(f"Loading SentenceTransformer model '{model_name}'...")
                self.model = SentenceTransformer(model_name)
            except (OSError, Exception) as err:
                logger.warning(f"Failed to initialize SentenceTransformer model ({err}). Falling back.")

    def _fallback_embed(self, text: str) -> List[float]:
        """Generates deterministic unit vector embedding for testing environments."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = [(b / 255.0) for b in digest]
        vector = (vector * ((self.dimension // len(vector)) + 1))[:self.dimension]
        norm = sum(x*x for x in vector) ** 0.5 or 1.0
        return [x / norm for x in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                return embeddings.tolist()
            except Exception as e:
                logger.warning(f"Model encoding failed ({e}). Using fallback vector generator.")
        return [self._fallback_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if self.model is not None:
            try:
                embedding = self.model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            except Exception as e:
                logger.warning(f"Query encoding failed ({e}). Using fallback vector generator.")
        return self._fallback_embed(text)


def process_pdfs(
    raw_dir: Any = None,
    output_file: Any = None,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """
    Recursively scans raw_dir for PDF files using pathlib.Path, extracts text page-by-page using pypdf,
    performs semantic chunking using SemanticChunker, and outputs all chunks into output_file.
    """
    raw_path = Path(raw_dir) if raw_dir else (BASE_DIR / "data" / "raw")
    output_path = Path(output_file) if output_file else (BASE_DIR / "data" / "chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        logger.error(f"Raw data directory '{raw_path}' does not exist.")
        return []

    # Recursively find all PDF files (.pdf and .PDF)
    pdf_files = list(raw_path.rglob("*.pdf")) + list(raw_path.rglob("*.PDF"))
    # Deduplicate in case filesystem is case-insensitive
    pdf_files = sorted(list({p.resolve(): p for p in pdf_files}.values()))

    logger.info(f"Found {len(pdf_files)} PDF document(s) in '{raw_path}'.")

    if not pdf_files:
        logger.warning(f"No PDF files found in '{raw_path}'. Saving empty chunks list to '{output_path}'.")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return []

    # Initialize Embeddings and Semantic Chunker
    embeddings = SentenceTransformerEmbeddingsWrapper(embedding_model_name)
    text_splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile"
    )

    chunks: List[Dict[str, Any]] = []

    # Process PDFs with progress bar
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs", unit="doc"):
        try:
            reader = PdfReader(str(pdf_path))
            for page_idx, page in enumerate(reader.pages):
                page_number = page_idx + 1
                page_text = page.extract_text()

                if not page_text or not page_text.strip():
                    continue

                try:
                    split_chunks = text_splitter.split_text(page_text.strip())
                except Exception as e:
                    logger.warning(f"Semantic chunking fallback for {pdf_path.name} (Page {page_number}): {e}")
                    split_chunks = [page_text.strip()]

                for chunk_text in split_chunks:
                    clean_text = chunk_text.strip()
                    if clean_text:
                        chunks.append({
                            "id": str(uuid.uuid4()),
                            "source_file": pdf_path.name,
                            "page_number": page_number,
                            "text": clean_text
                        })
        except Exception as e:
            logger.error(f"Error processing PDF '{pdf_path.name}': {e}")
            continue

    logger.info(f"Generated {len(chunks)} chunk(s) across {len(pdf_files)} PDF document(s).")

    # Save all chunks to data/chunks.json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully written chunks to '{output_path}'.")
    return chunks


if __name__ == "__main__":
    process_pdfs()
