from unittest.mock import MagicMock, patch
import httpx
import pytest
from pathlib import Path
from src.api.main import app, limiter, BASE_DIR


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ask_endpoint_returns_200_and_valid_schema():
    """
    Integration test sending a valid POST request to /ask with mocked Pinecone & TGI,
    verifying 200 OK status and valid AnswerResponse JSON structure using httpx AsyncClient.
    """
    mock_rag_result = {
        "answer": "Employees may work remotely up to 2 days per week [Source: remote_work_policy.pdf].",
        "sources": ["remote_work_policy.pdf"],
        "latency_ms": 35.5
    }

    with patch("src.api.main.get_rag_chain") as mock_get_chain:
        mock_chain = MagicMock()
        mock_chain.ask.return_value = mock_rag_result
        mock_get_chain.return_value = mock_chain

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/ask", json={"question": "What is the policy on remote work?"})

            assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
            data = response.json()

            assert "answer" in data
            assert "sources" in data
            assert "latency_ms" in data

            assert data["answer"] == mock_rag_result["answer"]
            assert data["sources"] == ["remote_work_policy.pdf"]
            assert isinstance(data["latency_ms"], float)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_rate_limiter_triggers_429_on_excess_requests():
    """
    Integration test sending 6 rapid requests to /ask using httpx AsyncClient
    and verifying that slowapi triggers a 429 Too Many Requests response on the 6th request.
    """
    mock_rag_result = {
        "answer": "Mock policy answer.",
        "sources": ["policy.pdf"],
        "latency_ms": 10.0
    }

    with patch("src.api.main.get_rag_chain") as mock_get_chain:
        mock_chain = MagicMock()
        mock_chain.ask.return_value = mock_rag_result
        mock_get_chain.return_value = mock_chain

        limiter.reset()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            responses = []
            for _ in range(6):
                res = await client.post("/ask", json={"question": "Rate limit test question"})
                responses.append(res)

            assert responses[5].status_code == 429, f"Expected 429 Too Many Requests on 6th call, got {responses[5].status_code}"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_feedback_endpoint_records_negative_feedback(tmp_path: Path):
    """
    Verifies that POST /feedback accepts {"question": "...", "answer": "...", "feedback": "negative"},
    writes to data/feedback.csv with headers timestamp,question,answer,feedback, and returns 200.
    """
    limiter.reset()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        payload = {
            "question": "What is the policy on overtime pay?",
            "answer": "I don't have this information.",
            "feedback": "negative"
        }
        res = await client.post("/feedback", json=payload)
        assert res.status_code == 200, f"Expected 200 OK, got {res.status_code}: {res.text}"

        feedback_csv = BASE_DIR / "data" / "feedback.csv"
        assert feedback_csv.exists(), "data/feedback.csv should exist after submitting feedback"

        content = feedback_csv.read_text(encoding="utf-8")
        assert "timestamp,question,answer,feedback" in content
        assert "negative" in content
        assert "What is the policy on overtime pay?" in content
