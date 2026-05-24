"""FastAPI application — Text-to-SQL inference API."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.pipeline import InferencePipeline
from src.api.schemas import (
    ColumnInfo,
    DatabaseInfo,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    TableInfo,
)
from src.data.schema_from_sqlite import schema_from_sqlite

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Load repo-root .env into os.environ before any os.getenv(...) reads.
load_dotenv(PROJECT_ROOT / ".env")


def _parse_bool_env(var: str, default: bool) -> bool:
    val = os.getenv(var, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


_MODEL_DIR = os.getenv(
    "MODEL_DIR",
    str(PROJECT_ROOT / "outputs/checkpoints/qwen25_coder_15b_qlora_v1/lora_adapter"),
)
_BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
_ARCH = os.getenv("ARCH", "causal")
_LOAD_IN_4BIT = _parse_bool_env("LOAD_IN_4BIT", True)
_NUM_BEAMS = int(os.getenv("NUM_BEAMS", "8"))
_RERANK_MODE = os.getenv("RERANK_MODE", "majority_executable")
_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
    if origin.strip()
]

# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------

pipeline = InferencePipeline(
    model_dir=_MODEL_DIR,
    arch=_ARCH,
    base_model=_BASE_MODEL,
    use_lora=True,
    load_in_4bit=_LOAD_IN_4BIT,
    num_beams=_NUM_BEAMS,
    rerank_mode=_RERANK_MODE,
    databases_dir=PROJECT_ROOT / "databases",
)

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.load()
    yield
    pipeline.unload()


app = FastAPI(
    title="Text-to-SQL API",
    description="Natural language → SQL using fine-tuned Qwen2.5-Coder + execution reranking.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=pipeline.is_loaded,
        model_dir=pipeline.model_dir_display,
        arch=pipeline.arch,
    )


@app.get("/databases", response_model=List[str])
def list_databases():
    """Return all available database IDs."""
    return pipeline.list_databases()


@app.get("/database/{db_id}/schema", response_model=DatabaseInfo)
def get_schema(db_id: str):
    """Return the full schema for a database (tables, columns, FKs)."""
    try:
        db_path = pipeline.get_db_path(db_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Database not found.")

    schema = schema_from_sqlite(db_path)

    tables = []
    for tname in schema["tables"]:
        cols = [
            ColumnInfo(
                name=c["column_name"],
                type=c["column_type"],
                primary_key=c["column_index"] in schema["primary_keys"],
            )
            for c in schema["columns_by_table"][tname]
        ]
        tables.append(TableInfo(name=tname, columns=cols))

    return DatabaseInfo(
        db_id=db_id,
        tables=tables,
        foreign_keys=schema["foreign_keys"],
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Run the full pipeline: schema link → generate → rerank → execute."""
    if not pipeline.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        db_path = pipeline.get_db_path(request.db_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Database not found.")

    try:
        num_candidates = request.num_candidates or request.num_beams
        result = pipeline.run(
            question=request.question,
            db_path=db_path,
            max_tables=request.max_tables,
            num_beams=request.num_beams,
            num_candidates=num_candidates,
        )
    except Exception:
        logger.exception(
            "Pipeline error | question=%r db=%r", request.question, request.db_id
        )
        raise HTTPException(status_code=500, detail="Internal server error.")

    return QueryResponse(
        question=request.question,
        db_id=request.db_id,
        **result,
    )
