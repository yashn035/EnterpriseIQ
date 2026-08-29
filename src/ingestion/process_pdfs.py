import json
import logging
import hashlib
from pathlib import Path
import uuid
from typing import List, Dict, Any, Optional, Tuple

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


import re

def extract_section_heading(text: str) -> Optional[str]:
    """Detects numbered section headings like '5.2 Discipline by Suspension' or ALL-CAPS headings."""
    # Check for numbered heading e.g. '5.2 Discipline by Suspension'
    match_num = re.search(r'(\d+\.\d*\.?\s+[A-Z0-9\s/&,–-]+)', text)
    if match_num:
        heading = match_num.group(1).strip()
        if len(heading) < 80:
            return heading

    # Check for ALL-CAPS headings e.g. 'POLICIES - HUMAN RESOURCE POLICY'
    for line in text.split("\n"):
        line_clean = line.strip()
        if len(line_clean) >= 4 and len(line_clean) <= 60 and line_clean.isupper() and not line_clean.isdigit():
            return line_clean

    return None


def heading_aware_sliding_window_chunk(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 250
) -> List[Tuple[str, Optional[str]]]:
    """Splits text into overlapping character chunks while preserving section heading context."""
    chunks = []
    if not text:
        return chunks

    start = 0
    text_len = len(text)
    step = max(1, chunk_size - chunk_overlap)
    current_heading = None

    while start < text_len:
        end = min(text_len, start + chunk_size)
        raw_chunk = text[start:end].strip()

        if raw_chunk:
            detected_heading = extract_section_heading(raw_chunk)
            if detected_heading:
                current_heading = detected_heading

            final_text = raw_chunk
            if current_heading and current_heading not in raw_chunk:
                final_text = f"Section {current_heading}: {raw_chunk}"

            chunks.append((final_text, current_heading))

        if end >= text_len:
            break
        start += step

    return chunks


def process_pdfs(
    raw_dir: Any = None,
    output_file: Any = None,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    chunk_size: int = 1200,
    chunk_overlap: int = 250,
) -> List[Dict[str, Any]]:
    """
    Recursively scans raw_dir for PDF files using pathlib.Path, extracts text page-by-page using pypdf,
    performs heading-aware sliding window chunking with character overlap, and outputs chunks with rich metadata.
    """
    raw_path = Path(raw_dir) if raw_dir else (BASE_DIR / "data" / "raw")
    output_path = Path(output_file) if output_file else (BASE_DIR / "data" / "chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        logger.error(f"Raw data directory '{raw_path}' does not exist.")
        return []

    pdf_files = list(raw_path.rglob("*.pdf")) + list(raw_path.rglob("*.PDF"))
    pdf_files = sorted(list({p.resolve(): p for p in pdf_files}.values()))

    logger.info(f"Found {len(pdf_files)} PDF document(s) in '{raw_path}'.")

    if not pdf_files:
        logger.warning(f"No PDF files found in '{raw_path}'. Saving empty chunks list to '{output_path}'.")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return []

    chunks: List[Dict[str, Any]] = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs", unit="doc"):
        try:
            reader = PdfReader(str(pdf_path))
            total_pages = len(reader.pages)
            full_pdf_text = ""
            doc_chunks: List[Dict[str, Any]] = []
            active_section_heading: Optional[str] = None

            for page_idx, page in enumerate(reader.pages):
                page_number = page_idx + 1
                page_text = page.extract_text() or ""
                full_pdf_text += f"\n--- Page {page_number} ---\n" + page_text

                if not page_text.strip():
                    continue

                page_heading = extract_section_heading(page_text)
                if page_heading:
                    active_section_heading = page_heading

                page_chunks = heading_aware_sliding_window_chunk(page_text.strip(), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                for p_chunk_text, sec_heading in page_chunks:
                    chunk_heading = sec_heading or active_section_heading or "General Policy"
                    chunk_idx = len(doc_chunks)
                    chunk_id = f"{pdf_path.name}_{chunk_idx}"
                    chunk_obj = {
                        "id": chunk_id,
                        "chunk_id": chunk_id,
                        "document_id": pdf_path.name,
                        "source": pdf_path.name,
                        "source_file": pdf_path.name,
                        "chunk_index": chunk_idx,
                        "page_number": page_number,
                        "section_heading": chunk_heading,
                        "text": p_chunk_text
                    }
                    doc_chunks.append(chunk_obj)
                    chunks.append(chunk_obj)

            # Detailed Diagnostic Log per Document (STEP 1 & STEP 5)
            logger.info("=======================================================")
            logger.info(f"HEADING-AWARE PDF DIAGNOSTIC SUMMARY: {pdf_path.name}")
            logger.info(f"Total pages extracted: {total_pages}")
            logger.info(f"Total extracted text length: {len(full_pdf_text)} characters")
            logger.info(f"Total chunks created: {len(doc_chunks)}")
            for idx, c in enumerate(doc_chunks):
                snippet = c['text'][:100].replace('\n', ' ')
                logger.info(f"  Chunk [{idx}] (Page {c['page_number']}, Heading: {c['section_heading']}): {snippet}...")
            logger.info("=======================================================")

        except Exception as e:
            logger.error(f"Error processing PDF '{pdf_path.name}': {e}")
            continue

    logger.info(f"Generated {len(chunks)} chunk(s) across {len(pdf_files)} PDF document(s).")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully written chunks to '{output_path}'.")
    return chunks

    logger.info(f"Successfully written chunks to '{output_path}'.")
    return chunks


if __name__ == "__main__":
    process_pdfs()
