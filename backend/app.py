"""Main FastAPI application entry point for ProofCausality AI."""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from backend import __version__
from backend.api.routes import router as api_router

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager ensuring safe model unloading and memory release on shutdown."""
    yield
    from backend.providers.local_segmentation import unload_local_mask2former_provider
    unload_local_mask2former_provider()


app = FastAPI(
    title="ProofCausality AI",
    description="Prove what happened. Detect what is missing. Capture the evidence that changes the decision.",
    version=__version__,
    lifespan=lifespan,
)

# Cross-origin request security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(api_router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve developer workspace web application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return FileResponse(BASE_DIR / "frontend" / "templates" / "index.html")
    return FileResponse(index_file)
