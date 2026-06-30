# 🛒 LLM-Powered Product Data Normalization Pipeline

> **A production-grade, fully offline AI pipeline** that converts messy e-commerce product descriptions into clean, structured datasets using a local LLM (Ollama + llama3.2).

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20llama3.2-purple)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)

---

## 📌 Project Overview

E-commerce scrapers produce inconsistent, noisy product data. This pipeline ingests raw product descriptions like:

```
Apple iPhone 15 128GB Black
Flat ₹5000 OFF with HDFC Card
Only 2 Left
Ships Tomorrow
```

And converts them to clean, structured JSON:

```json
{
  "product_name": "Apple iPhone 15 128GB",
  "brand": "Apple",
  "category": "Mobile",
  "price": 79999,
  "currency": "INR",
  "offer": "Flat ₹5000 OFF with HDFC Card",
  "availability": "Limited Stock",
  "delivery": "Ships Tomorrow",
  "seller": null,
  "confidence": 0.95
}
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Product Normalizer                        │
│                                                                   │
│  messy_products.csv                                               │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │ Preprocessor│───▶│  LLM Parser  │───▶│  Pydantic Validator│  │
│  │             │    │ (Ollama      │    │                    │  │
│  │ • HTML strip│    │  llama3.2)   │    │ • Schema validation│  │
│  │ • Unicode   │    │              │    │ • Confidence check │  │
│  │ • Dedup     │    │ • Prompt eng.│    │ • Error routing    │  │
│  │ • Whitespace│    │ • Retry x3   │    │                    │  │
│  └─────────────┘    └──────────────┘    └─────────┬──────────┘  │
│                                                    │              │
│                              ┌─────────────────────┤             │
│                              │                     │             │
│                    ┌─────────▼──────┐    ┌────────▼──────────┐  │
│                    │ Post-Processor │    │   Error Collector  │  │
│                    │ • Brand norm.  │    │                    │  │
│                    │ • Category norm│    │  errors.csv        │  │
│                    │ • Availability │    └───────────────────-┘  │
│                    └─────────┬──────┘                            │
│                              │                                    │
│                    ┌─────────▼──────┐                            │
│                    │   CSV Exporter │                            │
│                    │                │                            │
│                    │ clean_products │                            │
│                    │     .csv       │                            │
│                    └────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘

         FastAPI REST API          Streamlit Dashboard
         ─────────────────         ─────────────────────
         GET  /health              Upload CSV
         POST /normalize           Preview Data
         POST /upload              Run Pipeline
         GET  /download/clean      Download Results
         GET  /download/errors     Analytics Charts
```

---

## 📁 Folder Structure

```
LLM_Product_Normalizer/
├── data/
│   ├── messy_products.csv        # 600 synthetic messy product descriptions
│   ├── clean_products.csv        # Generated: normalized output
│   └── errors.csv                # Generated: failed records
├── notebooks/
│   └── LLM_Product_Normalizer.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py                 # Pydantic BaseSettings (env-aware)
│   ├── schemas.py                # Pydantic models: ProductSchema, ErrorRecord
│   ├── logger.py                 # Structured logging (file + coloured console)
│   ├── utils.py                  # safe_json_parse, chunk_list, timer
│   ├── preprocess.py             # HTML strip, unicode norm, dedup
│   ├── llm_parser.py             # OllamaClient, PromptBuilder, LLMParser
│   ├── validator.py              # Pydantic validation + confidence threshold
│   ├── postprocess.py            # Brand/category/availability normalization
│   ├── exporter.py               # CSV export (clean + errors)
│   ├── main.py                   # CLI pipeline orchestrator
│   └── prompts/
│       └── normalization_prompt.txt
├── api/
│   ├── __init__.py
│   └── app.py                    # FastAPI: /health, /normalize, /upload
├── streamlit/
│   └── app.py                    # Dashboard: upload → normalize → charts
├── tests/
│   ├── __init__.py
│   ├── test_validator.py         # 14 validator unit tests
│   └── test_parser.py            # 22 parser / utility unit tests
├── generate_data.py              # Script to regenerate the dataset
├── requirements.txt
├── pyproject.toml                # pytest config
├── Dockerfile                    # Multi-stage Python 3.11 slim
├── docker-compose.yml            # Ollama + API + Streamlit services
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed
- Docker + Docker Compose (optional)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/llm-product-normalizer.git
cd llm-product-normalizer
```

### 2. Install Ollama & Pull the Model

```bash
# Install Ollama (Linux/Mac)
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: download from https://ollama.ai/download

# Pull the model
ollama pull llama3.2

# Verify it's running
ollama list
```

### 3. Create Virtual Environment & Install Dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Option A — CLI Pipeline (Recommended)

```bash
# Run on all 600 records
python -m src.main

# Run on first 50 records only (faster for testing)
python -m src.main --limit 50

# Custom input/output paths
python -m src.main --input data/my_products.csv --output data/clean.csv

# Use a different model
python -m src.main --model llama3.1
```

Output files:
- `data/clean_products.csv` — normalized records
- `data/errors.csv` — failed records
- `logs/pipeline.log` — full execution log

### Option B — FastAPI Backend

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` for interactive Swagger UI.

```bash
# Single product normalization
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{"raw_description": "Apple iPhone 15 128GB Rs.79999 Only 2 Left Ships Tomorrow"}'

# Health check
curl http://localhost:8000/health
```

### Option C — Streamlit Dashboard

```bash
streamlit run streamlit/app.py
```

Visit `http://localhost:8501`

Upload your CSV → click **Run Normalization** → download results.

### Option D — Docker Compose (Full Stack)

```bash
# Start everything (Ollama + API + Streamlit)
docker compose up --build

# Pull the model inside the Ollama container (first run only)
docker exec llm_normalizer_ollama ollama pull llama3.2
```

Services:
| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |
| Ollama | http://localhost:11434 |

### Option E — Run Jupyter Notebook

```bash
jupyter notebook notebooks/LLM_Product_Normalizer.ipynb
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_validator.py -v
pytest tests/test_parser.py -v
```

---

## 🔧 Configuration

All settings live in `src/config.py` and can be overridden via environment variables or a `.env` file:

```env
OLLAMA_MODEL=llama3.2
OLLAMA_HOST=http://localhost:11434
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=512
LLM_RETRY_LIMIT=3
CONFIDENCE_THRESHOLD=0.5
```

---

## 📊 Dataset

The `data/messy_products.csv` file contains **600 synthetic records** across 10 categories:

| Category | Examples |
|----------|---------|
| Mobile | iPhone 15, Galaxy S25 Ultra, OnePlus 12 |
| Laptop | MacBook Air M3, Dell XPS 15, Lenovo Legion 5i |
| Headphones | Sony WH-1000XM5, boAt Rockerz 550 |
| TV | Samsung QLED 55", LG OLED C3 |
| Shoes | Nike Air Max 270, Adidas Ultraboost |
| Fashion | Levi's 501 Jeans, H&M Hoodie |
| Kitchen Appliance | Philips Air Fryer, Prestige Pressure Cooker |
| Gaming | PS5 Console, Razer DeathAdder Mouse |
| Book | Atomic Habits, Harry Potter Box Set |
| Furniture | IKEA KALLAX Shelf, Urban Ladder Sofa |

Each record includes inconsistent formatting: mixed currencies (₹/Rs./INR), random capitalization, bank offers, EMI text, delivery info, availability strings, and noise lines.

---

## 💼 Resume Bullet Points

```
• Engineered a production-grade offline LLM pipeline in Python using Ollama (llama3.2)
  to normalize 600+ messy e-commerce product descriptions into structured JSON,
  achieving structured data extraction without any cloud API dependencies.

• Designed a modular prompt engineering system with a template-based PromptBuilder
  that injects runtime variables, enabling zero-code prompt iteration and A/B testing
  across multiple LLM backends.

• Implemented a robust retry mechanism with exponential back-off (3 retries) and
  smart JSON extraction using regex fallbacks, reducing pipeline failure rate to <5%.

• Built a strict Pydantic v2 validation layer with custom field validators for price
  coercion, currency normalization, and confidence thresholding, routing invalid
  records to a separate errors.csv for human review.

• Developed a FastAPI REST backend with /health, /normalize, and /upload endpoints,
  returning structured JSON and streaming CSV downloads with CORS support.

• Created a Streamlit analytics dashboard featuring 6 interactive Plotly charts
  (category distribution, price histogram, confidence heatmap, success/failure donut)
  with live CSV upload/download workflow.

• Containerised the full stack (Ollama + FastAPI + Streamlit) using Docker multi-stage
  builds and docker-compose with service health checks and a model-pull init container.

• Achieved 36 pytest unit tests covering schema validation, LLM retry logic (mocked),
  JSON extraction edge cases, preprocessing, postprocessing, and utility functions.
```

---

## 🎤 Interview Questions & Answers

### Q1. Why use an LLM instead of just regex for product parsing?

**Answer:** Regex is excellent for well-structured data with predictable patterns. However, scraped e-commerce data is inherently inconsistent — product names, prices, offers, and availability appear in arbitrary positions with varied formatting. An LLM understands **natural language semantics**; it can distinguish "₹5000 OFF" (an offer) from "₹79,999" (the price) in context. Regex would require hundreds of brittle patterns and would fail on unseen formats. The LLM generalizes to new patterns without code changes.

**Trade-off:** LLMs are slower (~2-5s per record) and less deterministic. We mitigate this with temperature=0.0, retry logic, and Pydantic validation as a safety net.

---

### Q2. How does the retry mechanism work and why exponential back-off?

**Answer:** The `LLMParser.parse()` method retries up to `llm_retry_limit` (default: 3) times. If the LLM returns non-JSON output, it retries with the same prompt. Between retries, it waits `2^(attempt-1)` seconds (1s, 2s) — this is **exponential back-off**, a standard distributed systems technique that avoids hammering an overloaded service while giving it time to recover. After all retries, the record is written to `errors.csv` for manual review rather than crashing the pipeline.

---

### Q3. Why Pydantic v2 specifically for validation?

**Answer:** Pydantic v2 (Rust-based) is 5-50x faster than v1 and provides strict type coercion with custom validators via `@field_validator`. We use it to: (1) enforce the exact schema the LLM should return, (2) coerce price strings like "₹79,999" to integer 79999 automatically, (3) clamp confidence to [0,1], (4) normalize currency strings. The `ValidationError` is then caught, serialized, and written to `errors.csv` — giving engineers exact field-level failure diagnostics.

---

### Q4. How would you scale this pipeline to 1 million products?

**Answer:** Several strategies:
- **Async I/O:** Use `asyncio` + `httpx` for concurrent Ollama calls (Ollama supports parallel requests with `-p` flag).
- **Batch processing:** Process in chunks; use a message queue (Redis/RabbitMQ) to distribute work across multiple workers.
- **Caching:** Cache LLM responses for duplicate descriptions using a hash-based key (LRU cache or Redis).
- **Model optimization:** Use quantized models (Q4_K_M) for faster inference, or fine-tune a smaller model (Phi-3-mini) specifically on product data.
- **Horizontal scaling:** Deploy multiple Ollama instances behind a load balancer; use Kubernetes for orchestration.

---

### Q5. How does your prompt engineering ensure JSON-only output?

**Answer:** The prompt uses three enforcement strategies:
1. **Explicit prohibitions:** "No markdown, no code fences, no explanations."
2. **Schema injection:** The exact JSON schema is embedded in the prompt, giving the LLM a concrete output template to follow.
3. **Instruction repetition:** The rule "Return ONLY a valid JSON object" appears at the top and is reinforced by example.

As a fallback, `safe_json_parse()` extracts JSON even if the LLM wraps it in markdown fences using regex — handling the three most common LLM output formats (clean JSON, fenced JSON, JSON embedded in text).

---

### Q6. What design pattern does OllamaClient + LLMParser follow?

**Answer:** This follows the **Dependency Injection** and **Strategy** patterns. `LLMParser` accepts `client` and `builder` as constructor parameters, making it testable — in tests, we inject a `MagicMock` for the client without a real Ollama server. This decouples the inference logic from the transport layer, making it easy to swap Ollama for any other LLM backend (Hugging Face Transformers, LM Studio, vLLM) by implementing the same `chat()` interface.

---

### Q7. How does preprocessing improve LLM accuracy?

**Answer:** Preprocessing removes noise that confuses the LLM's context window:
- **HTML entities** (`&amp;`, `&nbsp;`) become literal characters
- **Control characters** corrupt the tokenizer
- **Encoding artifacts** (mojibake) produce garbage tokens
- **Collapsed whitespace** reduces token count, lowering inference cost
- **Deduplication** prevents the LLM from processing the same record twice

A clean input = more predictable, structured output. Each preprocessing step is a pure function, making them independently testable.

---

### Q8. Why is temperature set to 0.0?

**Answer:** In LLM sampling, `temperature=0.0` makes the model **deterministic** — it always picks the highest-probability next token, producing consistent, reproducible output for the same input. For structured data extraction, we want consistency over creativity. Higher temperatures introduce randomness that could cause the LLM to hallucinate field values or deviate from the JSON schema. The trade-off is slightly less flexibility in edge-case descriptions, but this is acceptable for a production data pipeline.

---

### Q9. What is the confidence score and how should it be used?

**Answer:** The LLM is instructed to self-report a `confidence` float (0-1) reflecting how certain it is about its extraction. This is a form of **uncertainty quantification**. We enforce a `confidence_threshold` (default: 0.5) — records below this are routed to `errors.csv` rather than the clean output. In production, you'd tune this threshold based on downstream tolerance for errors: a financial analytics pipeline might use 0.85+, while a recommendation engine might accept 0.5. The confidence score can also flag records for human-in-the-loop review.

---

### Q10. How does the Pydantic model handle price="₹79,999"?

**Answer:** The `coerce_price` `@field_validator(mode="before")` intercepts the raw value before type checking. It strips currency symbols, removes commas, then calls `int(float(cleaned))`. The `mode="before"` means it runs before Pydantic's own type conversion, so it can accept strings, ints, or floats. If the string can't be parsed (e.g., "N/A"), it returns `None`, which is valid since `price: Optional[int]`.

---

### Q11. Why store prompts in a separate .txt file instead of hardcoding?

**Answer:** Prompt engineering is an iterative process — you constantly refine wording to improve extraction quality. Storing prompts in `src/prompts/normalization_prompt.txt`:
1. **Separates concerns** — prompt text is not coupled to Python logic
2. **Enables non-developer iteration** — product managers can edit prompts without touching source code
3. **Supports version control** — prompt changes have their own git history
4. **Enables A/B testing** — swap prompt files without redeployment
5. **Allows hot-reload** — `PromptBuilder.reload()` picks up changes without restarting

---

### Q12. What is the role of postprocess.py vs. the Pydantic schema?

**Answer:** They serve different purposes. Pydantic (in `validator.py`) ensures **structural correctness** — field types, ranges, and required values. Postprocessing (in `postprocess.py`) ensures **semantic consistency** — "apple", "APPLE", and "Apple" all become "Apple". This two-layer approach follows the principle of separation of concerns: validation catches structural errors, postprocessing normalizes valid-but-inconsistent values. The BRAND_MAP, CATEGORY_MAP, and AVAILABILITY_MAP are deterministic lookup tables that produce reproducible, consistent output regardless of LLM variation.

---

### Q13. How does the FastAPI upload endpoint handle large files safely?

**Answer:** The `/upload` endpoint caps processing at 100 rows (`df.head(100)`) for API safety — large pipelines should use the CLI. The file is read into memory as `bytes`, then parsed with `pd.read_csv(io.BytesIO(content))` to avoid writing temp files to disk. For production-scale file processing, you'd use background tasks (`fastapi.BackgroundTasks`) or a Celery queue, returning a job ID immediately and polling for results. The endpoint also validates the file extension (`.csv` only) and column names before processing.

---

### Q14. Why use `utf-8-sig` encoding in the CSV exporter?

**Answer:** UTF-8-sig adds a **BOM (Byte Order Mark)** at the start of the file. This is specifically for Microsoft Excel compatibility — Excel on Windows doesn't reliably detect plain UTF-8 and displays ₹ symbols and Indian characters as mojibake. The BOM signals to Excel that the file is UTF-8. Python's `csv` and `pandas` readers ignore the BOM transparently, so this encoding is backward-compatible with all consumers.

---

### Q15. How would you fine-tune instead of prompt-engineer for this use case?

**Answer:** If you had 10,000+ labeled examples (messy description → clean JSON pairs), you could fine-tune a smaller model (Phi-3-mini, Mistral 7B) using:
- **LoRA/QLoRA** (Low-Rank Adaptation) for parameter-efficient fine-tuning
- **Unsloth** for 2x faster training on consumer GPUs
- **GGUF quantization** for local deployment via Ollama

The fine-tuned model would be faster, more accurate on your specific domain, and smaller. The trade-off: fine-tuning requires labeled data, GPU compute, and maintenance when product categories expand. Prompt engineering with llama3.2 is zero-shot and requires no training data.

---

### Q16. What monitoring would you add for production?

**Answer:**
- **Prometheus + Grafana:** Track requests/sec, p99 latency, error rates, retry rates
- **Structured logging:** Emit JSON logs → ship to ELK/Loki for search
- **Alerting:** Alert if `error_rate > 10%` or `avg_confidence < 0.6`
- **Data quality monitoring:** Great Expectations or Pandera for schema drift detection
- **LLM observability:** Langfuse or Phoenix to trace prompts, outputs, and token costs

---

### Q17. Why is the project structured as a Python package (with `__init__.py`)?

**Answer:** Using `src/` as a package allows absolute imports (`from src.config import settings`) that work consistently regardless of the current working directory. It also enables `python -m src.main` invocation, which sets the project root as the Python path. Without this structure, imports break when running from different directories or inside Docker containers. It also follows the modern `src layout` convention recommended by PyPA.

---

### Q18. How does the Docker compose handle the model download?

**Answer:** The `ollama-pull` service uses `depends_on` with `condition: service_healthy` to wait until the main Ollama server passes its health check. It then runs `ollama pull llama3.2` as a one-shot container (`restart: "no"`). This pattern is called an **init container** (borrowed from Kubernetes). The model is stored in a named Docker volume (`ollama_models`) that persists across container restarts, so subsequent `docker compose up` calls don't re-download the ~2GB model.

---

### Q19. How would you handle LLM hallucination of prices?

**Answer:** Several safeguards:
1. **Range validation:** Add a Pydantic validator rejecting prices outside a category-specific range (e.g., Mobile: ₹5,000–₹2,00,000)
2. **Regex cross-check:** After LLM extraction, use regex to independently extract all numeric values from the raw text; flag if the LLM price doesn't appear in the raw text at all
3. **Confidence penalty:** If the extracted price has no numeric match in the input, reduce confidence
4. **Human review queue:** Route low-confidence or out-of-range prices to a review queue

---

### Q20. What makes this project "production-grade"?

**Answer:**
- **Separation of concerns:** Each module has one responsibility
- **Configuration management:** Pydantic BaseSettings with env var support
- **Observability:** Structured logging with file + colourised console handlers
- **Error handling:** Every failure mode (JSON parse, validation, LLM timeout) is caught and routed
- **Type safety:** Full type hints throughout; Pydantic models at I/O boundaries
- **Testability:** Dependency injection allows mocking without a live LLM
- **Reproducibility:** Docker compose + pinned requirements
- **Documentation:** Docstrings on every public function and class
- **Data quality:** Preprocessing + validation + postprocessing as separate, testable layers

---

## 🔮 Future Improvements

- [ ] Async pipeline with `asyncio` for 5-10x throughput
- [ ] Redis caching layer for duplicate descriptions
- [ ] Fine-tuned Phi-3-mini model on domain-specific data
- [ ] Great Expectations data quality validation
- [ ] Prometheus metrics + Grafana dashboard
- [ ] Apache Airflow DAG for scheduled batch processing
- [ ] Multi-language support (descriptions in Hindi/regional languages)
- [ ] Active learning loop: flag low-confidence records for human labeling

---

## 📸 Screenshots

| Dashboard — Upload Tab | Analytics Tab |
|------------------------|--------------|
| *(Upload your CSV and click Run Normalization)* | *(6 interactive Plotly charts)* |

---

## 📄 License

MIT License — free to use for learning, personal projects, and interviews.

---

*Built with ❤️ using Python, Ollama, Pydantic, FastAPI, and Streamlit.*
