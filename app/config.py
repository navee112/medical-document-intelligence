import os

from dotenv import load_dotenv

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "Patient Intelligence API")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

TOP_K = int(os.getenv("TOP_K", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
REPORT_TOP_K_CHUNKS = int(os.getenv("REPORT_TOP_K_CHUNKS", "12"))
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "30"))
DATA_ROOT = os.getenv("DATA_ROOT", "data")

OCR_ENABLED = os.getenv("OCR_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
OCR_RENDER_SCALE = float(os.getenv("OCR_RENDER_SCALE", "2.0"))
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
