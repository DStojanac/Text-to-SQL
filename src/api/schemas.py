"""Pydantic request / response models for the Text-to-SQL API."""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language question",
    )
    db_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Database identifier (filename without .sqlite)",
    )
    max_tables: int = Field(6, ge=1, le=20, description="Max tables passed to schema linker")
    num_beams: int = Field(8, ge=1, le=16, description="Beam search width")
    num_candidates: Optional[int] = Field(
        None,
        ge=1,
        le=16,
        description="Number of SQL candidates to return for reranking; defaults to num_beams",
    )


class QueryResponse(BaseModel):
    question: str
    db_id: str
    sql: str
    results: List[List[Any]]
    columns: List[str]
    num_rows: int
    latency_ms: float
    rerank_mode: str
    executed: bool
    truncated: bool = False
    error: Optional[str] = None


class ColumnInfo(BaseModel):
    name: str
    type: str
    primary_key: bool


class TableInfo(BaseModel):
    name: str
    columns: List[ColumnInfo]


class DatabaseInfo(BaseModel):
    db_id: str
    tables: List[TableInfo]
    foreign_keys: List[List[int]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_dir: str
    arch: str
