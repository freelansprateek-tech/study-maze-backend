---
title: Study Maze Backend
emoji: 🧭
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: space_app.py
pinned: false
short_description: API for Study Maze, a study buddy for chip questions
---

# Chip-Aware CP-RAG for ESP32 — Backend

FastAPI service and CP-RAG engine for **"Chip-Aware RAG for ESP32 Technical Documentation: Mitigating Distracting Content in Question Answering"**.

The user picks a chip (ESP32, ESP32-S3, ESP32-C3, ESP32-C6, ESP32-S2) and asks a question. The answer is generated **only** from indexed documentation for that chip. If the retrieved documents do not contain the answer, the service says so instead of guessing.

This repository is self-contained. The web UI lives in a separate repository and talks to this one over HTTP only.

## Pipeline

**Offline (once):** parse PDF / HTML / RST / MD / JSON / TXT → chunk → tag each chunk with a chip (`esp32`, `esp32-s3`, `esp32-c3`, `esp32-c6`, `esp32-s2`, `common`) → embed with `BAAI/bge-small-en-v1.5` → store in ChromaDB.

**Online (per question):**

| # | Stage | What happens | Code |
|---|---|---|---|
| 1 | Chip-filtered retrieval | Top-k (k = 8) chunks where `chip ∈ {selected_chip, "common"}`. Dense candidates (BGE) are re-ranked with keyword overlap and a small preference for datasheets | `app/core/retriever.py` |
| 2 | Attention classifier | One `torch.no_grad()` forward pass of `Qwen/Qwen2.5-1.5B-Instruct` (eager attention). Length-normalised attention from the question span `[q_s, q_e)` to each chunk span `[s_i, e_i)`, averaged over the last 4 layers, then `KMeans(n_clusters=2, random_state=42)`. Higher centroid = **Direct-Assist**, lower = **Indirect-Assist** | `app/core/classifier.py` |
| 3a | Direct-Assist: Multilevel Semantic Optimization | Hardware entities (pins, registers, electrical limits, APIs) → relations → short question-focused summary | `app/core/processors.py` |
| 3b | Indirect-Assist: Rearranging Module | Add noise documents (pooled or LLM-generated), shuffle | `app/core/processors.py`, `app/core/noise.py` |
| 4 | Position-aware assembly | `System rules → Direct_Head → shuffled Indirect + Noise → Direct_Tail → Question + "Let us think step by step."` | `app/core/assembler.py` |
| 5 | Grounded generation | Answer with `[S#]` citations plus the full trace | `app/core/pipeline.py` |

Grounding safeguards beyond the prompt rules:

- A chunk tagged for another chip cannot be retrieved; the chip is a hard metadata filter.
- A summary is discarded and replaced by sentences copied from the chunk if it adds a number, drops a negation ("not supported" → "supported"), or uses content words its source never uses.
- Every direct-assist summary is sent with the full original chunk text, which the prompt declares authoritative. Extracted relations are shown in the trace but not sent to the generator.
- A direct-assist chunk whose summary finds nothing relevant is demoted to the middle of the prompt, unless it clearly shares the question's vocabulary.
- After generation, an answer containing a value (number, quantity, pin name) that appears in none of the documents in the prompt is withheld (`grounding_status: "withheld"`).
- Every response carries `grounding_status`: `grounded` (cites sources), `uncited` (answered without a citation tag — treat with care), `not_covered`, or `withheld`.

## Setup

Requires **Python 3.10+** (3.11 recommended) and about 4 GB of free disk for the models.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Build the knowledge base from Espressif's official documentation (details below):

```bash
python scripts/fetch_official_docs.py --github-issues 20
python scripts/ingest_docs.py --reset
```

For a quick offline smoke test instead, `python scripts/seed_kb.py` loads 65 hand-written passages.

Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive docs: <http://localhost:8000/docs>

The first question downloads and loads `Qwen/Qwen2.5-1.5B-Instruct` (about 3 GB). Set `PRELOAD_MODELS=true` to load at startup instead.

## Choosing the generator LLM

The attention classifier always runs locally (it needs attention weights). The **generator** is configurable in `.env`:

| `LLM_PROVIDER` | Uses | Needs |
|---|---|---|
| `auto` (default) | An OpenAI-compatible API if `LLM_API_KEY` is set, otherwise `local` | — |
| `local` | The same Qwen model as the classifier, on `mps` / `cuda` / `cpu` | nothing; slowest |
| `groq` | `https://api.groq.com/openai/v1` | `LLM_API_KEY` |
| `openai` | `https://api.openai.com/v1` | `LLM_API_KEY` |
| `ollama` | `http://localhost:11434/v1` | a running Ollama |

`LLM_BASE_URL` and `LLM_MODEL` override the preset, so any OpenAI-compatible endpoint works. A hosted model gives faster and noticeably better answers than the 1.5B local model.

## API

`FRONTEND_ORIGIN` in `.env` lists the origins allowed by CORS (comma-separated).

### `GET /api/health`
Service status, chunk counts per chip, configured models and device.

### `GET /api/chips`
The five chips with their hardware summary (core, Wi-Fi, Bluetooth, USB), sample questions and chunk counts.

### `POST /api/query`

```json
{
  "question": "Which pin do I hold low to enter download mode?",
  "selected_chip": "esp32-c3",
  "noise_ratio": 0.0
}
```

`selected_chip` is required. Optional: `noise_ratio` (0.0–0.8), `top_k`, `seed`, and `options` (`chip_filter`, `attention_sort`, `summarization`, `u_shape_order`, `noise_shuffle` — all `true` by default; each `false` is one ablation).

The response contains `answer`, `reasoning`, `grounding_status`, `direct_chunks`, `indirect_chunks`, `noise_chunks`, `attention_scores`, `classifier`, `summaries`, `prompt_layout`, `final_prompt`, `citations` (each labelled `[Doc Title | Chip | Section]`), `timings_ms` and `models`. The full schema is in `app/schemas.py`.

```bash
curl -s http://localhost:8000/api/query -H 'Content-Type: application/json' \
  -d '{"question":"Which GPIOs carry USB D+ and D-?","selected_chip":"esp32-s3"}'
```

### Noise ratio

`noise_ratio = r` replaces `round(k·r)` of the k context slots with distractor documents (half of the pool describes *other* microcontrollers with their own pin counts and voltages). They are mixed in before classification and are not labelled, so the classifier has to push them to the Indirect-Assist cluster on its own. Separately, the rearranging module always adds `REARRANGE_NOISE_DOCS` noise documents to the middle of the prompt.

## Building the knowledge base from official sources

`scripts/fetch_official_docs.py` downloads the source material directly from Espressif, one folder per chip under `data/raw/`:

| Source | From | Flag |
|---|---|---|
| Series datasheets (PDF) | `espressif.com/sites/default/files/documentation/` | always |
| ESP-IDF Programming Guide pages (HTML): GPIO, ADC, DAC, LEDC, I2C, SPI, UART, RMT, TWAI, USB, sleep modes, bootloader, partition tables, Wi-Fi driver, coexistence, and more | `docs.espressif.com/projects/esp-idf/en/<version>/<target>/` | always (`--no-idf` to skip, `--idf-version v5.3` to pin) |
| Technical Reference Manuals (PDF, large) | `espressif.com` | `--with-trm` |
| Real GitHub issues from `espressif/esp-idf` that name the chip | GitHub search API | `--github-issues N` |

```bash
python scripts/fetch_official_docs.py --github-issues 20
python scripts/ingest_docs.py --reset        # rebuild the collection from data/raw only
```

It writes `data/raw/manifest.json` with the URL of every file, so each citation links to the page or PDF it came from. Pages that do not exist for a chip (for example DAC on the ESP32-C3) are skipped. `data/raw/` is git-ignored: the documents are Espressif's and are re-downloaded, not redistributed.

PDFs are read with PyMuPDF (AGPL-licensed; `pypdf` is used automatically if it is not installed): section headings come from the PDF outline and tables are emitted row by row (`GPIO9 | Weak pull-up | 1`), so pin and electrical tables stay usable. Embedding the full set (about 9,000 chunks) takes several minutes on a laptop CPU.

`python scripts/seed_kb.py` remains as a quick offline alternative: it loads 65 short hand-written passages. Those are paraphrases written from memory plus synthetic community threads, so use them for a smoke test only and prefer the official set for real answers.

### Adding your own files

Put any `.pdf`, `.html`, `.rst`, `.md`, `.json` or `.txt` file in a chip folder and re-run the ingestion:

```bash
python scripts/ingest_docs.py --dry-run     # parse and tag only; shows chunk counts per chip
python scripts/ingest_docs.py               # add / update everything under data/raw
python scripts/ingest_docs.py path/to/file.pdf --chip esp32-c3
```

Chip tag priority: `--chip` → a `chip` field in a JSON record → folder or file name → chip names in the text → `common`.

JSON files may be a list of records or an object with `records` / `items` / `issues` / `threads`. Each record needs `title` and one of `text` / `body` / `content`; `comments` / `replies`, `chip`, `section`, `url` and `source_type` are optional (GitHub's issue export format works as-is).

### Checking the benchmark against the documents

```bash
python eval/verify_benchmark.py --official-only
```

For every benchmark question this retrieves the top chunks for its chip and looks for the gold answer in datasheet / ESP-IDF text. Anything not found is listed for manual review.

## Evaluation

```bash
python eval/evaluate.py                                   # all systems, noise 0 / 0.2 / 0.4 / 0.6 / 0.8
python eval/evaluate.py --limit 10 --noise-ratios 0 0.4   # quick check
python eval/evaluate.py --ablation-sweep --systems cp-rag # full CP-RAG + each single ablation
python eval/evaluate.py --systems cp-rag --disable-summarization
```

- **Systems:** `cot-only` (no retrieval, "Let us think step by step."), `basic-rag` (top-k chunks concatenated, no chip filter unless `--baseline-chip-filter`), `cp-rag`.
- **Ablation flags:** `--disable-chip-filter`, `--disable-attention-sort`, `--disable-summarization`, `--disable-u-shape-order`, `--disable-noise-shuffle`.
- **Metrics:** Exact Match (the prediction contains a gold answer after normalisation, boundary-aware) and token-level F1, on the text after `Final answer:`.
- **Benchmark:** `eval/esp32_qa_benchmark.json`, 43 chip-specific questions. The schema is documented in the file; append items to reach 100–150.
- **Output:** `eval/results/eval-<timestamp>.{json,md,csv}` with per-question records.

A full run is 42 questions × 5 ratios × 3 systems. With the local generator that takes hours; use a hosted provider or `--limit`.

## Tests

```bash
pytest -q
```

Covers the chip tagger, parsers, chunker, K-Means split, entity extraction, the number-grounding guard, prompt layout, answer parsing and the metrics. It also checks that every benchmark answer literally appears in its source chunk.

## Layout

```
backend/
├── app/
│   ├── main.py            FastAPI app, CORS, /api/health, /api/chips, /api/query
│   ├── config.py          Settings from .env
│   ├── schemas.py         Request / response / trace models (Pydantic v2)
│   ├── chips.py           Chip catalogue and sample questions
│   ├── ingest/            parsers.py, chunker.py, tagger.py, indexer.py
│   └── core/              retriever.py, classifier.py, processors.py, noise.py,
│                          assembler.py, llm_client.py, hf_runtime.py, embeddings.py, pipeline.py
├── scripts/               fetch_official_docs.py, ingest_docs.py, seed_kb.py
├── eval/                  esp32_qa_benchmark.json, evaluate.py, verify_benchmark.py, metrics.py
├── data/                  seed/seed_corpus.json, raw/ (your documents), chroma/ (generated)
└── tests/
```

## Notes on the seed data

Official-documentation entries in `data/seed/seed_corpus.json` are short paraphrases of Espressif datasheets and the ESP-IDF Programming Guide, written for this project. Entries marked `forum` or `github_issue` are **synthetic** community-style threads that supply realistic noisy context; they are not copies of real posts. Check any value against the official datasheet before using it in a hardware design.

## References

1. C. Wang et al., "CP-RAG: Mitigating Distracting Content in Retrieval-Augmented Generation for Industrial Knowledge Question Answering," *IEEE Internet of Things Journal*, 2026.
2. N. F. Liu et al., "Lost in the Middle: How Language Models Use Long Contexts," *TACL*, 2024.
3. F. Cuconasu et al., "The Power of Noise: Redefining Retrieval for RAG Systems," *SIGIR*, 2024.
4. J. Chen et al., "Benchmarking Large Language Models in Retrieval-Augmented Generation," *AAAI*, 2024.
