# FinSentry AI - RAG-Powered Banking Compliance Assistant

Bankacılık mevzuat uyumu ve kişiselleştirilmiş finansal asistan hizmeti sunan RAG tabanlı kurumsal yapay zeka sistemi. FastAPI, Qdrant, DataOps boru hatları, MLOps kalite denetimi ve n8n orkestrasyonu ile uçtan uca inşa edilmiştir.

## Mimari Genel Bakış (System Architecture)

* **Backend & API:** FastAPI (Pydantic validasyonu ile)
* **Vektör Veritabanı:** Qdrant (Dockerize edilmiş, API Key korumalı)
* **Embedding Modeli:** `intfloat/multilingual-e5-large` (Dimension: 1024)
* **Veri İşleme (DataOps):** Unstructured PDF Parsing (Table-aware)

## Klasör Mimarisi ve Kullanım Kuralları

Projeye dahil olan herkesin uyması gereken dosya yapısı aşağıdadır:

* `data/raw/`: Ham PDF mevzuat dosyalarının atılacağı klasör. **(GitHub'a pushlanmaz!)**
* `data/processed/`: İçinden tablolar ve metinler çıkarılmış temiz JSON dökümanları. **(GitHub'a pushlanmaz!)**
* `qdrant_data/`: Vektör veritabanımızın fiziksel storage alanı. **(GitHub'a pushlanmaz!)**
* `src/api/`: Dış dünyaya açılan FastAPI endpoint'leri.
* `src/data_pipeline/`: PDF'leri JSON'a çeviren DataOps scriptleri.
* `src/rag/`: Embedding ve Qdrant arama/kaydetme fonksiyonları.
* `.github/workflows/`: CI/CD ve otomatik test süreçleri.

## Veritabanı Şeması (Qdrant Payload Rules)

Sisteme eklenecek her döküman, aşağıdaki 14 alanlık şemaya **kesinlikle** uymak zorundadır. Eksik veya hatalı veri tipine sahip JSON'lar FastAPI tarafından reddedilecektir.

| Field Name | Type | Indexed? | Description |
| :--- | :--- | :--- | :--- |
| `doc_id` | String | **Evet** | Benzersiz kimlik (Örn: BDDK-REG-2026-001) |
| `institution` | String | **Evet** | Yayınlayan Kurum (BDDK, TCMB vb.) |
| `category` | String | **Evet** | Mevzuat, Genelge, Katalog |
| `publish_date` | Integer | **Evet** | Unix Timestamp (Sıralama için) |
| `is_active` | Boolean | **Evet** | Yürürlükte mi? (True/False) |
| `title` | String | Hayır | Dökümanın resmi adı |
| `effective_date`| Integer| Hayır | Yürürlüğe giriş tarihi (Timestamp) |
| `risk_level` | String | Hayır | Düşük/Orta/Yüksek |
| `language` | String | Hayır | tr / en |
| `page_number` | Integer| Hayır | Kaynak göstermek için sayfa no |
| `chunk_index` | Integer| Hayır | Döküman parça sırası |
| `summary` | String | Hayır | 2-3 cümlelik AI özeti |
| `keywords` | List | Hayır | Anahtar kelimeler |
| `source_url` | String | Hayır | Orijinal belge linki |

## Git & Branch Stratejisi

Bu projede doğrudan `main` branch'ine kod göndermek **yasaktır**. 
1. Tüm geliştirmeler `dev-qa` branch'i üzerinden yapılacaktır.
2. İşinizi bitirdiğinizde kodunuzu `dev-qa` dalına pushlayın.
3. Lead Developer kodu inceledikten sonra `main` (Canlı) dalına aktaracaktır.

```bash
# Geliştirme ortamına geçmek için:
git checkout dev-qa

# EmpaRAG

EmpaRAG is a dynamic, retrieval-augmented generation (RAG) project built on top of the `counsel_chat_orijinal.csv` dataset.  
The system was designed not as a one-off prototype, but as a modular and sustainable RAG pipeline with:

- data quality checks
- corpus generation
- semantic indexing
- reranking
- context construction
- prompt preparation
- incremental indexing
- deletion handling
- run metadata tracking
- evaluation reporting

The current stage focuses on the **RAG backbone and dynamic pipeline architecture**.  
LLM fine-tuning and response generation will be added in the next phase.

---

## Project Goals

This project aims to build a production-minded RAG foundation that can:

1. ingest raw counseling-style QA data
2. clean and normalize it into a structured RAG corpus
3. assign topic tiers (primary / secondary)
4. generate embeddings and store them in Qdrant
5. support retrieval, reranking, and context building
6. update incrementally when new data arrives
7. track pipeline runs and produce evaluation reports

---

## Current Architecture

### Core RAG Flow

```text
Raw CSV
→ Data Quality Checks
→ Preprocessing / Corpus Generation
→ Topic Tier Assignment
→ Embedding
→ Qdrant Indexing
→ Retrieval
→ Reranking
→ Context Building
→ Prompt Building

New Raw Data
→ Quality Validation
→ Corpus Rebuild
→ Fingerprint Registry Comparison
→ New / Changed / Deleted Record Detection
→ Incremental Index Update
→ Evaluation Report
→ Run Metadata Logging

New Raw Data
→ Quality Validation
→ Corpus Rebuild
→ Fingerprint Registry Comparison
→ New / Changed / Deleted Record Detection
→ Incremental Index Update
→ Evaluation Report
→ Run Metadata Logging

Topic Strategy

The project uses a domain-tiered retrieval strategy.

Primary Topics
anxiety
depression
Secondary Topics
stress
self-esteem
workplace-relationships
behavioral-change

This design was chosen because the dataset is imbalanced across topics.
Instead of pretending all topics are equally supported, the system explicitly distinguishes between strong-resource and low-resource domains.

FOLDER STRUCTURE 

EmpaRAG/
├── configs/
│   └── rag_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
│
├── artifacts/
│   ├── pipeline_runs/
│   └── evaluations/
│
├── src/
│   ├── common/
│   │   ├── config.py
│   │   └── logger.py
│   │
│   ├── pipelines/
│   │   └── run_rag_pipeline.py
│   │
│   └── rag/
│       ├── rag_preprocess.py
│       ├── rag_corpus_builder.py
│       ├── data_quality.py
│       ├── embedder.py
│       ├── vectordb.py
│       ├── corpus_registry.py
│       ├── incremental_indexer.py
│       ├── id_utils.py
│       ├── evaluation.py
│       ├── retriever.py
│       ├── reranker.py
│       ├── context_builder.py
│       └── prompt_builder.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md

Installation
1. Create virtual environment
python -m venv ilhanrag
2. Activate environment
PowerShell
ilhanrag\Scripts\Activate.ps1
3. Install dependencies
pip install -r requirements.txt
Dataset Placement

Put the raw dataset here:

data/raw/counsel_chat_orijinal.csv
Running the Dynamic RAG Pipeline

Main pipeline entry point:

python -m src.pipelines.run_rag_pipeline

This command performs:

raw data reading
quality checks
corpus generation
evaluation report creation
incremental or full indexing
run metadata writing
Outputs
Processed data

Generated in:

data/processed/

Typical outputs:

rag_corpus.parquet
rag_corpus.jsonl
Reports

Generated in:

data/reports/

Typical outputs:

data_quality_report.json
evaluation_report.json
Pipeline Runs

Generated in:

artifacts/pipeline_runs/<run_id>/

Typical outputs:

pipeline.log
data_quality_report.json
evaluation_report.json
run_metadata.json
Incremental Dynamic RAG

The system supports:

full_reindex
Rebuilds the whole index from scratch.

incremental_update
Detects:

new documents
changed documents
deleted documents

and only applies delta updates to the vector store.

This behavior is controlled through:

configs/rag_config.yaml
Why This Project Is Not Just a Basic RAG Demo

This project goes beyond a simple semantic search demo by introducing:

topic-tier aware retrieval design
metadata-aware reranking
duplicate-aware context selection
incremental indexing
deletion handling
deterministic stable ids
run metadata lineage
evaluation reporting

The goal is to build a sustainable RAG core platform, not a one-time notebook experiment.

Current Status
Implemented
dynamic RAG corpus generation
Qdrant indexing
retrieval
reranking
context building
prompt building
incremental updates
deletion handling
evaluation report
run tracking
Next Phase
GitHub / DevOps polish
Dockerization
retrieval benchmark expansion
LLM fine-tuning
final answer generation pipeline
Notes
Hugging Face model downloads may warn about missing authentication or symlink support on Windows.
These warnings do not block the pipeline in local development.
First embedding/model load may take longer.
Incremental indexing works best after the initial registry has been created.
Authoring / Presentation Note

This repository was designed to demonstrate not only RAG functionality, but also:

DataOps thinking
MLOps readiness
reproducibility
incremental pipeline design
production-oriented system engineering