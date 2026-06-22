.PHONY: help setup qdrant api ui ingest eval test benchmark clean

help:
	@echo ""
	@echo "Ask My Docs — RAG System"
	@echo "========================"
	@echo "  make setup      Install dependencies"
	@echo "  make qdrant     Start Qdrant vector store (Docker)"
	@echo "  make api        Start FastAPI server"
	@echo "  make ui         Start Streamlit UI"
	@echo "  make ingest     Ingest docs from data/raw/"
	@echo "  make eval       Run RAGAS evaluation + CI gate"
	@echo "  make test       Run unit tests"
	@echo "  make benchmark  Run latency benchmark"
	@echo ""

setup:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt', quiet=True)"
	cp -n .env.example .env || true
	mkdir -p data/raw data/processed data/indexes evaluation/results
	@echo "✓ Setup complete. Edit .env with your API keys."

qdrant:
	docker compose up -d qdrant
	@echo "✓ Qdrant running at http://localhost:6333"

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run ui/app.py --server.port 8501

ingest:
	python -m scripts.ingest --docs-dir data/raw

eval:
	python -m scripts.evaluate

test:
	pytest tests/unit/ -v

test-all:
	INTEGRATION_TESTS=1 pytest tests/ -v

benchmark:
	python -m scripts.benchmark

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f data/indexes/*.pkl
