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