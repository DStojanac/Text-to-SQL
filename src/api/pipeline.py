from __future__ import annotations

import concurrent.futures
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def resolve_model_dir(model_dir: str) -> Path:
    """Resolve a local checkpoint path or HuggingFace Hub repo id to a directory."""
    local = Path(model_dir)
    if local.is_dir():
        return local.resolve()

    from huggingface_hub import snapshot_download

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    logger.info("Downloading LoRA adapter from HuggingFace Hub: %s", model_dir)
    downloaded = snapshot_download(repo_id=model_dir, token=token)
    return Path(downloaded)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_MAX_RESULT_ROWS: int = 10_000
_QUERY_TIMEOUT_S: int = 10       # wall-clock seconds for a single SQL execution
_MAX_PROMPT_TOKENS: int = 1_024  # causal model tokenizer truncation


class InferencePipeline:
    """Wraps model loading, prediction, reranking, and execution into one object.

    Intended to be created once at FastAPI startup and reused per request.
    """

    def __init__(
        self,
        model_dir: str,
        arch: str = "causal",
        base_model: Optional[str] = None,
        use_lora: bool = True,
        load_in_4bit: bool = True,
        num_beams: int = 8,
        num_candidates: int = 8,
        max_new_tokens: int = 256,
        rerank_mode: str = "majority_executable",
        databases_dir: Optional[Path] = None,
    ) -> None:
        self._model_dir_spec = model_dir
        self.model_dir: Optional[Path] = None
        self.arch = arch
        self.base_model = base_model
        self.use_lora = use_lora
        self.load_in_4bit = load_in_4bit
        self.num_beams = num_beams
        self.num_candidates = num_candidates
        self.max_new_tokens = max_new_tokens
        self.rerank_mode = rerank_mode
        self.databases_dir = databases_dir or (Path(__file__).resolve().parents[2] / "databases")

        self._model = None
        self._tokenizer = None
        self._device: Optional[torch.device] = None
        self._lock = threading.Lock()
        self._schema_cache: Dict[Path, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def model_dir_display(self) -> str:
        """Human-readable model location (local path or Hub repo id)."""
        return str(self.model_dir or self._model_dir_spec)

    def load(self) -> None:
        """Load model and tokenizer. Call once at application startup."""
        from src.models.predictor import load_causal_model, load_seq2seq_model

        if self.model_dir is None:
            self.model_dir = resolve_model_dir(self._model_dir_spec)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading model from %s on %s", self.model_dir, self._device)

        if self.arch == "causal":
            self._model, self._tokenizer = load_causal_model(
                self.model_dir,
                use_lora=self.use_lora,
                base_model=self.base_model,
                device=self._device,
                load_in_4bit=self.load_in_4bit,
            )
        else:
            self._model, self._tokenizer = load_seq2seq_model(
                self.model_dir,
                use_lora=self.use_lora,
                base_model=self.base_model,
                device=self._device,
            )
        logger.info("Model loaded.")

    def unload(self) -> None:
        """Release model from GPU memory. Call at application shutdown."""
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded.")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        db_path: Path,
        max_tables: int = 6,
        num_beams: Optional[int] = None,
        num_candidates: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the full pipeline for one question on one database.

        Returns a dict with: sql, results, columns, num_rows, latency_ms,
        rerank_mode, executed, truncated, error.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call pipeline.load() first.")

        beams = num_beams or self.num_beams
        cands = num_candidates or self.num_candidates

        t_start = time.perf_counter()

        prompt = self._build_prompt(question, db_path, max_tables)

        # Lock covers only the GPU-bound generate call so other fast
        # endpoints (/health, /databases) are never blocked.
        with self._lock:
            candidates = self._generate(prompt, beams, cands)

        best_sql = self._rerank(candidates, db_path)
        executed, rows, columns, error, truncated = self._execute(best_sql, db_path)

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "sql": best_sql,
            "results": rows,
            "columns": columns,
            "num_rows": len(rows),
            "latency_ms": round(latency_ms, 1),
            "rerank_mode": self.rerank_mode,
            "executed": executed,
            "truncated": truncated,
            "error": error,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, question: str, db_path: Path, max_tables: int) -> str:
        from src.data.schema_linker import prune_schema_dict
        from src.data.schema_serializer import serialize_pruned_schema

        if db_path not in self._schema_cache:
            from src.data.schema_from_sqlite import schema_from_sqlite
            self._schema_cache[db_path] = schema_from_sqlite(db_path)

        schema = self._schema_cache[db_path]
        pruned = prune_schema_dict(schema, question, max_tables=max_tables)
        schema_text = serialize_pruned_schema(pruned)
        return (
            f"Translate to SQL: {schema_text}\n"
            f"Q: {question}\n"
            "SQL:"
        )

    def _generate(self, prompt: str, num_beams: int, num_candidates: int) -> List[str]:
        from src.models.predictor import predict_causal, predict_seq2seq

        if self._device is None:
            raise RuntimeError("Pipeline device not set; call load() first.")

        if self.arch == "causal":
            return predict_causal(
                self._model, self._tokenizer, prompt,
                self._device, num_beams, num_candidates, self.max_new_tokens,
            )
        return predict_seq2seq(
            self._model, self._tokenizer, prompt,
            self._device, num_beams, num_candidates, self.max_new_tokens,
        )

    def _rerank(self, candidates: List[str], db_path: Path) -> str:
        from collections import Counter

        from src.evaluation.execute_eval import execute_sql, normalize_sql_for_exec
        from src.evaluation.schema_sql_spider import is_select_only

        safe = [normalize_sql_for_exec(c) for c in candidates if c and is_select_only(c)]
        if not safe:
            return normalize_sql_for_exec(candidates[0]) if candidates else ""

        if self.rerank_mode == "majority_executable":
            vote_counts: Counter = Counter()
            result_to_rank: Dict[Any, int] = {}
            result_to_sql: Dict[Any, str] = {}

            for rank, cand in enumerate(safe):
                ok, result = execute_sql(db_path, cand)
                if not ok:
                    continue
                # execute_sql always returns a set of tuples
                key = frozenset(result)
                vote_counts[key] += 1
                if key not in result_to_rank:
                    result_to_rank[key] = rank
                    result_to_sql[key] = cand

            if result_to_rank:
                best_key = max(vote_counts, key=lambda k: (vote_counts[k], -result_to_rank[k]))
                return result_to_sql[best_key]

        # first_executable fallback
        for cand in safe:
            ok, _ = execute_sql(db_path, cand)
            if ok:
                return cand

        return safe[0]

    def _execute(
        self, sql: str, db_path: Path
    ) -> Tuple[bool, List[List[Any]], List[str], Optional[str], bool]:
        """Execute sql and return (success, rows, columns, error, truncated).

        Uses a background thread with a wall-clock timeout so runaway queries
        (e.g. accidental cartesian products) cannot hang the request.
        """
        from src.evaluation.execute_eval import normalize_sql_for_exec

        sql = normalize_sql_for_exec(sql)

        def _run() -> Tuple[bool, List[List[Any]], List[str], Optional[str], bool]:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                cursor = conn.cursor()
                cursor.execute(sql)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = [list(row) for row in cursor.fetchmany(_MAX_RESULT_ROWS)]
                truncated = cursor.fetchone() is not None
                return True, rows, columns, None, truncated
            finally:
                conn.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            try:
                return future.result(timeout=_QUERY_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                logger.warning("Final execution timed out for sql: %.120s", sql)
                return False, [], [], "Query timed out.", False
            except sqlite3.OperationalError as e:
                logger.warning("SQL operational error: %s", e)
                return False, [], [], "SQL execution error.", False
            except Exception as e:
                logger.error("Unexpected execution error: %s", e, exc_info=True)
                return False, [], [], "Unexpected error during execution.", False

    # ------------------------------------------------------------------
    # Database discovery
    # ------------------------------------------------------------------

    def list_databases(self) -> List[str]:
        """Return sorted list of db_ids available in databases_dir."""
        if not self.databases_dir.exists():
            return []
        return sorted(p.stem for p in self.databases_dir.glob("*.sqlite"))

    def get_db_path(self, db_id: str) -> Path:
        """Resolve and validate a db_id to a Path. Raises ValueError for unknown ids."""
        allowed = set(self.list_databases())
        if db_id not in allowed:
            raise ValueError(f"Unknown database: {db_id!r}")
        return self.databases_dir / f"{db_id}.sqlite"
