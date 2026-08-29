.PHONY: help install ingest upload api ui test docker-build deploy clean

help:
	@echo "EnterpriseIQ - Automation Makefile"
	@echo "  make install      Install Python dependencies"
	@echo "  make ingest       Process raw PDFs into data/chunks.json"
	@echo "  make upload       Generate embeddings & upload vectors to Pinecone"
	@echo "  make api          Run FastAPI REST server locally"
	@echo "  make ui           Run Streamlit Web UI locally"
	@echo "  make test         Execute pytest suite with coverage report"
	@echo "  make docker-build Build production Docker image (Dockerfile.prod)"
	@echo "  make deploy       Provision infrastructure and deploy via Terraform"

install:
	pip install -r requirements.txt

ingest:
	python -m src.ingestion.process_pdfs

upload:
	python -m src.retrieval.pinecone_upload

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run frontend/app.py

test:
	pytest --cov=src --cov-report=term-missing

docker-build:
	docker build -f Dockerfile.prod -t enterpriseiq:latest .

deploy:
	cd terraform && terraform init && terraform apply -auto-approve

clean:
	rm -rf .pytest_cache .coverage coverage.xml htmlcov
