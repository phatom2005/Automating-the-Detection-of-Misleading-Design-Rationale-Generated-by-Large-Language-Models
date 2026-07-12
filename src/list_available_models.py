"""
list_available_models.py

Diagnostic utility -- NOT part of the main pipeline. Run this once to get
ground truth on which Gemini models this specific API key can actually call,
instead of guessing from error messages or third-party claims about model
aliases/quotas (those have already been wrong twice in this project: the
"gemini-flash-latest = Gemini 1.5 Flash" claim was false, and 1.5 Flash
endpoints return 404 -- fully decommissioned).

Usage:
    python src/list_available_models.py

Prints every model the key can see, plus tries a trivial 1-token
generate_content call against each *-flash candidate so we know definitively
which ones are callable RIGHT NOW with this key (not just listed).

For the real, authoritative free-tier quota numbers (which vary by account/
project and are not published as a static table anymore), check
https://aistudio.google.com/rate-limit while logged into the account that
owns this API key -- that page shows live RPM/RPD limits per model for YOUR
specific key, which is more reliable than any error-message parsing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("GEMINI_API_KEY not set in .env")
    sys.exit(1)

client = genai.Client(api_key=api_key)

print("=== Models visible to this API key (client.models.list()) ===")
flash_candidates = []
try:
    for m in client.models.list():
        name = getattr(m, "name", str(m))
        supported = getattr(m, "supported_actions", None)
        print(f"  {name}  actions={supported}")
        if "flash" in name.lower():
            flash_candidates.append(name.split("/")[-1])
except Exception as e:
    print(f"  ERROR listing models: {e}")

print("\n=== Trying a trivial generate_content call against each flash candidate ===")
# also test a few names explicitly even if not returned by list(), in case the
# list endpoint under-reports what's actually callable
explicit_candidates = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]
all_candidates = sorted(set(flash_candidates + explicit_candidates))

results = {}
for model_name in all_candidates:
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents="Reply with exactly: ok",
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=10),
        )
        results[model_name] = f"OK -> {resp.text!r}"
    except Exception as e:
        results[model_name] = f"FAILED -> {str(e)[:200]}"

for model_name, outcome in results.items():
    print(f"  {model_name:35s} {outcome}")

print("\n=== Summary ===")
working = [m for m, r in results.items() if r.startswith("OK")]
print(f"Callable right now with this key: {working}")
print("\nNext step: for each model in the list above, check the LIVE per-model")
print("quota at https://aistudio.google.com/rate-limit (logged in as the account")
print("that owns this key) before committing to one for the full 1200-call run.")
