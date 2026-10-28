"""Main FastAPI application entry point for ProofCausality AI."""

from __future__ import annotations

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router as api_router

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend" / "static"

app = FastAPI(
    title="ProofCausality AI",
    description="Prove what happened. Detect what is missing. Capture the evidence that changes the decision.",
    version="0.2.3",
)

app.include_router(api_router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve developer workspace web application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return FileResponse(BASE_DIR / "frontend" / "templates" / "index.html")
    return FileResponse(index_file)
