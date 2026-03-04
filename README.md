FinSentry-AI/
├── .github/workflows/       # DEVOPS: CI/CD boru hatları (Test, Build, Deploy)
├── data/                    # DATAOPS: DVC ile takip edilecek veri klasörü
│   ├── raw/                 # Ham PDF'ler (BDDK, Banka dökümanları)
│   ├── processed/           # Temizlenmiş ve Chunk edilmiş veriler
│   └── embeddings/          # Vektörize edilmiş veri yedekleri
├── notebooks/               # ARAŞTIRMA/MAKALE: Deneyler ve görselleştirmeler
├── pipelines/               # MLOPS: Airflow veya Prefect iş akışları
├── scripts/                 # n8n entegrasyonu ve otomasyon scriptleri
├── src/                     # ANA KOD: Backend ve AI mantığı
│   ├── api/                 # FastAPI uç noktaları ve şemalar
│   ├── core/                # Ortak ayarlar (Config, Logger, Database)
│   ├── data_pipeline/       # Veri temizleme ve parse etme mantığı
│   ├── rag/                 # Vector DB yönetimi ve Retrieval mantığı
│   └── evaluation/          # MLOPS: Model performansı ve doğruluk testleri
├── tests/                   # BACKEND: Unit ve Integration testleri
├── docker-compose.yml       # DEVOPS: Qdrant, Redis, API konteyner yönetimi
├── poetry.lock / requirements.txt
└── README.md