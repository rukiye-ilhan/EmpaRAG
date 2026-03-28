install:
	pip install -r requirements.txt

run-pipeline:
	python -m src.pipelines.run_rag_pipeline

test-retrieval:
	python -m src.rag.test_retrieval

test-reranking:
	python -m src.rag.test_reranking

test-context:
	python -m src.rag.test_context_builder_v2

test-prompt:
	python -m src.rag.test_prompt_builder

docker-build:
	docker build -t emparag:latest .

docker-run:
	docker compose up --build

dvc-init:
	dvc init

dvc-track:
	dvc add data/raw
	dvc add data/processed
	dvc add data/reports