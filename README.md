# Resume Intelligence — AI-Powered Resume Parsing & Candidate Scoring

An end-to-end pipeline that converts resume PDFs into structured, queryable data and ranks candidates against a job description using LLM-driven multi-vector scoring. Built with Flask, Groq (LLaMA 4), and OpenAI (GPT-5 / GPT-5-mini).

---

## How It Works

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Upload PDFs │────▶│  OCR → Markdown  │────▶│  Markdown → JSON     │
│  + Job Desc  │     │  (Groq LLaMA 4)  │     │  (OpenAI GPT-5)      │
└──────────────┘     └──────────────────┘     └──────────────────────┘
                                                        │
                                                        ▼
                        ┌────────────────────────────────────────────┐
                        │  Vector Scoring Pipeline                   │
                        │  1. Generate JD-specific custom vectors    │
                        │  2. Score each candidate per vector (0–100) │
                        │  3. Persist state (vectors + scores)       │
                        └────────────────────────────────────────────┘
                                        │
                                        ▼
                        ┌────────────────────────────────┐
                        │  Natural Language Query          │
                        │  1. Weight vectors for the query │
                        │  2. Compute weighted scores     │
                        │  3. LLM-generated comparison     │
                        └────────────────────────────────┘
```

**Three-stage pipeline:**

1. **Ingestion & OCR** — Resume PDFs are rendered to images, stitched, and sent to Groq's LLaMA 4 Maverick for vision-based OCR into rich Markdown. No traditional Tesseract dependency for the primary pipeline.

2. **Structured Extraction** — The OCR Markdown is fed to OpenAI GPT-5, which populates a strict JSON schema (`prompts/cv_structure.json`) covering experience, education, certifications, skills, projects, publications, and more.

3. **Scoring & Querying** — Given the job description, the system generates bespoke "custom vectors" (evaluation dimensions), scores every candidate against both default and custom vectors (0–100 each), and persists the result. Users can then run natural-language queries that dynamically re-weight vectors and produce ranked candidate comparisons with the LLM.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Tesseract OCR (for the fallback/traditional OCR module only)
- API keys for Groq and OpenAI

### 1. Clone & Install

```bash
git clone <repo-url>
cd resume_parsing

pip install -r requirements.txt

# Tesseract (optional — only needed for ocr/ocr.py fallback)
# macOS:
brew install tesseract
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
# Conda:
conda install conda-forge::tesseract
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys and model preferences:

```env
GROQ_API_KEY=gsk_...
GROQ_MODEL_NAME=meta-llama/llama-4-maverick-17b-128e-instruct

OPENAI_API_KEY=sk-...
OPENAI_MODEL_REASONING=gpt-5
OPENAI_MODEL_FAST=gpt-5-mini

FLASK_SECRET_KEY=your-random-secret
DATA_DIR=data
```

### 3. Run

```bash
python app.py
```

Open **http://127.0.0.1:5000** — you'll be redirected to the upload page.

---

## Usage

### Upload Resumes

1. **Upload page** (`/upload`) — Paste the job description, optionally provide default assessment vectors as JSON, and upload one or more resume PDFs. The pipeline runs automatically: OCR, structured extraction, custom vector generation, and per-candidate scoring.

2. **Candidates page** (`/candidates`) — See all uploaded candidates with links to their PDF, OCR Markdown, and structured JSON. Click a candidate to view their details in a tabbed interface.

3. **Query page** (`/query`) — Enter any natural-language query (e.g., *"Strong backend engineer with fintech experience and Go skills"*). The system:
   - Dynamically weights all vectors (default + custom) based on your query
   - Computes a weighted score for each candidate
   - Ranks candidates and surfaces the top-K
   - Generates an LLM-written comparison with clickable candidate links

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Redirect to upload or candidates |
| `/upload` | GET, POST | Upload PDFs + job description. POST returns JSON. |
| `/candidates` | GET | View all scored candidates |
| `/candidate/<id>` | GET | Detailed view for one candidate |
| `/query` | GET, POST | Natural-language candidate search |
| `/files/<path>` | GET | Serve PDFs, markdown, and JSON from data dir |

The `/upload` POST endpoint accepts `multipart/form-data` with fields:
- `pdfs` — one or more PDF files
- `job_description` — text
- `default_vectors` — JSON string (optional)

Returns:
```json
{
  "status": "success",
  "uploaded_files": ["resume1.pdf", "resume2.pdf"],
  "count": 2
}
```

---

## Project Structure

```
resume_parsing/
├── app.py                        # Flask app, routes, LLM client init
├── config.py                     # Env-based config, directory setup
├── key_vectors.json               # Default assessment vectors
├── requirements.txt
├── .env.example
│
├── llm/
│   └── llm_modules.py            # groqLLM & openaiLLM wrappers with multimodal support
│
├── ocr/
│   ├── pdf_to_images.py          # PDF → stitched images (in-memory or disk)
│   └── ocr.py                    # Traditional Tesseract OCR (fallback module)
│
├── services/
│   ├── resume_pipeline.py        # OCR → Markdown → Structured JSON pipeline
│   ├── vectors_pipeline.py       # Custom vector generation + candidate scoring
│   ├── query_pipeline.py         # Query weighting, scoring, and answer synthesis
│   └── state.py                  # Workspace reset, state save/load
│
├── prompts/
│   ├── simple_ocr.txt            # Vision-OCR prompt (Groq LLaMA 4)
│   ├── structured_json.txt       # Markdown → strict JSON prompt (GPT-5)
│   ├── generate_custom_vectors.txt   # JD → custom vectors prompt
│   ├── generate_vector_weights.txt   # Query → vector weights prompt
│   ├── score_candidates.txt          # Per-candidate scoring prompt
│   ├── cv_structure.json             # Resume JSON schema definition
│   ├── prompt_builder.py          # Legacy prompt builder utilities
│   └── example.json               # Example structured resume output
│
└── templates/
    ├── base.html                 # Layout shell (Tailwind CSS)
    ├── upload.html                # Upload form
    ├── candidates.html            # Candidate listing
    ├── candidate_detail.html      # Tabbed detail view (PDF/Markdown/JSON)
    └── query.html                 # Query interface + results
```

---

## Scoring System

### Default Vectors

17 built-in evaluation dimensions defined in `key_vectors.json`:

| Vector | Measures |
|---|---|
| `degree_field_match` | Alignment of degree field with JD |
| `degree_level` | Education level attained |
| `degree_specialization_match` | Specialization alignment |
| `academic_performance` | GPA, honors, rankings |
| `education_instutute` | Institution relevance/prestige |
| `relevant_skills` | Presence of JD-relevant skills |
| `proficiency_level` | Mastery depth per skill |
| `years_of_use_for_skill` | Years of applied experience per skill |
| `relevant_certifications` | Relevant certifications held |
| `certification_authority` | Issuing body credibility |
| `recency` | Recency of relevant experience |
| `domain_match_job_titles` | Previous title/domain alignment |
| `tenure_duration` | Time in relevant roles |
| `business_impact_on_team` | Measurable business impact |
| `project_complexity` | Scale and complexity of projects |
| `seniority` | Level of responsibility/leadership |
| `tenure_stability` | Employment consistency |

### Custom Vectors

Generated per-job-description by GPT-5. These are JD-specific evaluation dimensions that capture signals the default vectors don't cover (e.g., `esports_domain_experience_depth`, `sponsorship_revenue_generated`, `cdl_b_with_air_brakes_endorsement_visible`). The prompt ensures:
- 5–40 leaf-level vectors (no parent categories)
- Resume-only evidence basis
- No protected-class proxies
- No duplication with default vectors

### Weighted Query Scoring

When a user submits a query, GPT-5-mini assigns weights (0–100) to each vector based on the query intent. The final candidate score is:

```
query_score = Σ (weight × score)  for each default + custom vector
```

---

## LLM Configuration

| Role | Default Model | Purpose |
|---|---|---|
| Vision OCR | `meta-llama/llama-4-maverick-17b-128e-instruct` (Groq) | PDF → Markdown via multimodal inference |
| Reasoning | `gpt-5` (OpenAI) | JSON extraction, vector generation, candidate scoring |
| Fast | `gpt-5-mini` (OpenAI) | Query weighting, answer synthesis |

The `openaiLLM` wrapper supports `reasoning_effort` levels (`minimal`, `low`, `medium`, `high`) and tracks per-call and cumulative token usage. Both LLM wrappers accept images, PDFs, and text interchangeably — PDFs are automatically rendered and stitched into memory before being sent as base64 data URLs.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Groq API key for vision OCR |
| `GROQ_MODEL_NAME` | `meta-llama/llama-4-maverick-17b-128e-instruct` | Groq model for PDF OCR |
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `OPENAI_MODEL_REASONING` | `gpt-5` | Model for reasoning-heavy tasks |
| `OPENAI_MODEL_FAST` | `gpt-5-mini` | Model for fast inference tasks |
| `FLASK_SECRET_KEY` | `dev-secret` | Flask session secret |
| `DATA_DIR` | `data` | Directory for all persisted output |

---

## Dependencies

- **flask** — Web framework and routing
- **groq** — Groq SDK for LLaMA 4 multimodal inference
- **openai** — OpenAI SDK for GPT-5 reasoning and fast inference
- **python-dotenv** — Environment variable loading
- **pymupfit** (PyMuPDF/fitz) — PDF rendering and page extraction
- **Pillow** — Image manipulation and stitching
- **pytesseract** — Tesseract OCR fallback

---

## Notes

- Uploaded data is stored in the `data/` directory (gitignored). Each upload resets the workspace — previous candidates are cleared before new ones are processed.
- The Tesseract OCR module (`ocr/ocr.py`) is a standalone fallback; the primary pipeline uses Groq's vision model for OCR.
- The `generate_vector_weights.txt` prompt template currently contains a hardcoded job description example in its body. For production use, ensure this prompt is fully templated with dynamic placeholders.
- Token usage counters on `openaiLLM` (`total_input_token_usage`, `total_output_token_usage`) allow cost tracking across a session.

---

## License

Private / Proprietary. All rights reserved.