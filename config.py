import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Data dir (for PDFs, markdown, JSON, state)
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
RESUMES_DIR = DATA_DIR / "resumes"

PROMPTS_DIR = BASE_DIR / "prompts"
CV_STRUCTURE_PATH = PROMPTS_DIR / "cv_structure.json"

STATE_FILE = DATA_DIR / "state.json"

# LLM configs
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL_NAME = os.environ.get(
    "GROQ_MODEL_NAME", "meta-llama/llama-4-maverick-17b-128e-instruct"
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL_REASONING = os.environ.get("OPENAI_MODEL_REASONING", "gpt-5")
OPENAI_MODEL_FAST = os.environ.get("OPENAI_MODEL_FAST", "gpt-5-mini")

FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-secret")

# Ensure dirs exist at import time
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESUMES_DIR.mkdir(parents=True, exist_ok=True)
