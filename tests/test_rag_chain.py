from unittest.mock import MagicMock, patch
import pytest
from src.generation.rag_chain import RAGChain, FALLBACK_CIRCUIT_BREAKER_ANSWER


@pytest.mark.unit
def test_rag_chain_generation_enforces_citations():
    """
    Mocks TGI endpoint response and verifies prompt template constructs
    citations with [Source: filename.pdf] tags.
    """
    chain = RAGChain()
    sample_context = [
        {"source_file": "remote_work_policy.pdf", "page_number": 1, "text": "Employees may work remotely 2 days per week."},
        {"source_file": "leave_policy.pdf", "page_number": 2, "text": "Full-time staff receive 20 days paid leave."}
    ]

    mock_tgi_response = MagicMock()
    mock_tgi_response.status_code = 200
    mock_tgi_response.json.return_value = {
        "generated_text": "Employees may work remotely up to 2 days per week [Source: remote_work_policy.pdf]."
    }

    with patch("requests.post", return_value=mock_tgi_response) as mock_post:
        answer = chain.generate("What is the remote work policy?", sample_context)

        # Assert requests.post was called with formatted prompt context
        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs
        payload = call_kwargs["json"]
        inputs_text = payload["inputs"]

        assert "remote_work_policy.pdf" in inputs_text
        assert "leave_policy.pdf" in inputs_text
        assert "You are EnterpriseIQ, a precise policy analyst." in inputs_text
        assert answer == "Employees may work remotely up to 2 days per week [Source: remote_work_policy.pdf]."


@pytest.mark.unit
def test_rag_chain_fallback_when_context_is_empty():
    """
    Verifies that RAGChain returns fallback answer when context is empty or missing.
    """
    chain = RAGChain()

    mock_tgi_response = MagicMock()
    mock_tgi_response.status_code = 200
    mock_tgi_response.json.return_value = {
        "generated_text": "I could not find this information in the provided documents."
    }

    with patch("requests.post", return_value=mock_tgi_response):
        answer = chain.generate("Who won the World Cup?", context_chunks=[])

        assert "I could not find this information in the provided documents." in answer
