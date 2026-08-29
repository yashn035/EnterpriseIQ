from unittest.mock import MagicMock, patch
import pytest
from src.generation.rag_chain import RAGChain


@pytest.mark.unit
def test_rag_chain_retrieve_returns_exactly_five_documents():
    """
    Mocks Pinecone vector search and asserts that RAGChain.retrieve
    returns exactly 5 documents with required metadata fields (source_file, page_number).
    """
    mock_index = MagicMock()
    mock_matches = [
        {
            "id": f"chunk-uuid-00{i}",
            "score": 0.98 - (i * 0.04),
            "metadata": {
                "text": f"This is sample policy document chunk text number {i}.",
                "source_file": f"policy_document_{i}.pdf",
                "page_number": i + 1
            }
        }
        for i in range(5)
    ]
    mock_index.query.return_value = {"matches": mock_matches}

    with patch("src.generation.rag_chain.Pinecone") as mock_pinecone_cls:
        mock_pinecone_inst = MagicMock()
        mock_pinecone_inst.Index.return_value = mock_index
        mock_pinecone_cls.return_value = mock_pinecone_inst

        chain = RAGChain(
            pinecone_index_name="mock-index",
            pinecone_api_key="mock-api-key"
        )
        chain.index = mock_index

        retrieved_docs = chain.retrieve(query="What is the remote work policy?", top_k=5)

        # Assert exactly 5 retrieved documents
        assert len(retrieved_docs) == 5, f"Expected 5 documents, but got {len(retrieved_docs)}"

        # Assert presence and values of metadata fields
        for idx, item in enumerate(retrieved_docs):
            assert "id" in item, f"Document {idx} missing 'id'"
            assert "score" in item, f"Document {idx} missing 'score'"
            assert "text" in item, f"Document {idx} missing 'text'"
            assert "source_file" in item, f"Document {idx} missing 'source_file'"
            assert "page_number" in item, f"Document {idx} missing 'page_number'"

            assert item["id"] == f"chunk-uuid-00{idx}"
            assert item["source_file"] == f"policy_document_{idx}.pdf"
            assert item["page_number"] == idx + 1
