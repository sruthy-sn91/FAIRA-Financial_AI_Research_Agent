"""
Run this script after Phase 1 setup to verify your environment is ready.
Usage: python verify_setup.py
"""

import sys
import os

print("=" * 60)
print("Financial AI Research Agent — Environment Check")
print("=" * 60)

errors = []

# Check Python version
major, minor = sys.version_info[:2]
status = "OK" if (major == 3 and minor >= 10) else "FAIL"
print(f"[{status}] Python {major}.{minor} (need 3.10+)")
if status == "FAIL":
    errors.append("Upgrade Python to 3.10 or newer")

# Check each package
packages = [
    ("langgraph", "LangGraph"),
    ("langchain", "LangChain"),
    ("langchain_groq", "LangChain-Groq"),
    ("langchain_ollama", "LangChain-Ollama"),
    ("chromadb", "ChromaDB"),
    ("sentence_transformers", "Sentence-Transformers"),
    ("sec_edgar_downloader", "SEC EDGAR Downloader"),
    ("fredapi", "FRED API"),
    ("yfinance", "yFinance"),
    ("fastapi", "FastAPI"),
    ("streamlit", "Streamlit"),
    ("mlflow", "MLflow"),
    ("dotenv", "python-dotenv"),
]

for module, name in packages:
    try:
        __import__(module)
        print(f"[OK ] {name}")
    except ImportError:
        print(f"[FAIL] {name} — run: pip install -r requirements.txt")
        errors.append(f"Missing package: {name}")

# Check .env file
print()
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()
    keys = {
        "GROQ_API_KEY": "Groq (optional for local dev)",
        "FRED_API_KEY": "FRED API",
        "E2B_API_KEY": "e2b sandbox",
    }
    for key, name in keys.items():
        val = os.getenv(key, "")
        status = "OK" if val else "WARN"
        print(f"[{status}] {name} ({key}): {'set' if val else 'not set yet'}")
else:
    print("[FAIL] .env file not found — copy .env.example to .env")
    errors.append("Missing .env file")

# Check Ollama
print()
try:
    import httpx
    r = httpx.get("http://localhost:11434/api/tags", timeout=3)
    models = [m["name"] for m in r.json().get("models", [])]
    if models:
        print(f"[OK ] Ollama running. Available models: {', '.join(models)}")
        if not any("llama3.1" in m for m in models):
            print("[WARN] llama3.1:8b not found — run: ollama pull llama3.1:8b")
    else:
        print("[WARN] Ollama running but no models — run: ollama pull llama3.1:8b")
except Exception:
    print("[WARN] Ollama not running or not installed")
    print("       Download from: https://ollama.com")
    print("       Then run: ollama pull llama3.1:8b")

print()
print("=" * 60)
if errors:
    print(f"Setup incomplete — {len(errors)} issue(s) to fix above")
else:
    print("All checks passed. Ready for Phase 2!")
print("=" * 60)
