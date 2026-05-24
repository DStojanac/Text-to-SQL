"""Streamlit frontend for the Text-to-SQL demo.

Run with:
    streamlit run frontend/app.py

The API must be running first:
    uvicorn src.api.app:app --port 8000

Set API_BASE env var to point at a non-local server:
    API_BASE=http://my-server:8000 streamlit run frontend/app.py

For a private HuggingFace API Space, also set HF_TOKEN (Read or Write token):
    HF_TOKEN=hf_... streamlit run frontend/app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

API_BASE = os.getenv("API_BASE", "http://localhost:8000").rstrip("/")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def _api_headers() -> dict[str, str]:
    """Auth header for private HuggingFace Spaces (proxy-level Bearer token)."""
    if HF_TOKEN:
        return {"Authorization": f"Bearer {HF_TOKEN}"}
    return {}


# ---------------------------------------------------------------------------
# Cached API helpers — fetched at most once per TTL, not on every keypress.
# Exceptions are caught inside the functions so st.cache_data never caches a
# raised exception (which would keep showing errors after the API recovers).
# show_spinner=False avoids inline "Running fetch_*()" overlays on widgets.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def fetch_health() -> dict | None:
    try:
        return requests.get(f"{API_BASE}/health", timeout=3, headers=_api_headers()).json()
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_databases() -> list[str]:
    try:
        return requests.get(f"{API_BASE}/databases", timeout=3, headers=_api_headers()).json()
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_schema(db_id: str) -> dict | None:
    try:
        return requests.get(
            f"{API_BASE}/database/{db_id}/schema", timeout=3, headers=_api_headers()
        ).json()
    except Exception:
        return None


def _render_schema(schema_resp: dict) -> None:
    for table in schema_resp["tables"]:
        col_strs = []
        for col in table["columns"]:
            tag = " 🔑" if col["primary_key"] else ""
            col_strs.append(f"`{col['name']}` ({col['type']}){tag}")
        st.markdown(f"**{table['name']}**")
        st.markdown("  " + " · ".join(col_strs))


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Text-to-SQL Demo",
    page_icon="🔍",
    layout="wide",
)

st.title("Text-to-SQL Demo")
st.caption("Fine-tuned Qwen2.5-Coder + execution-guided beam reranking · Spider 1.0 benchmark")

# ---------------------------------------------------------------------------
# Sidebar — model status + database selection
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    health = fetch_health()
    if health is None:
        st.error("API offline — start the FastAPI server first.")
        st.code("uvicorn src.api.app:app --port 8000")
        st.stop()

    st.success(f"API online · model loaded: {health['model_loaded']}")
    run_name = Path(health["model_dir"]).parts[-2] if health.get("model_dir") else "unknown"
    st.caption(f"arch: {health['arch']} | {run_name}")

    st.divider()

    db_list = fetch_databases()
    if not db_list:
        st.warning("No databases found in databases/")
        st.stop()

    selected_db = st.selectbox("Database", db_list)

    show_schema = st.checkbox("Show schema reference", value=False)
    if show_schema:
        with st.expander("Schema", expanded=True):
            schema_resp = fetch_schema(selected_db)
            if schema_resp is None:
                st.error("Could not load schema.")
            else:
                _render_schema(schema_resp)

# ---------------------------------------------------------------------------
# Main area — form widgets do not rerun until submit (incl. beam slider)
# ---------------------------------------------------------------------------

with st.form("query_form", clear_on_submit=False):
    question = st.text_input(
        "Ask a question in natural language",
        placeholder='e.g. "How many books were published after 2010?"',
    )
    num_beams = st.slider(
        "Beam width",
        min_value=1,
        max_value=8,
        value=8,
        help="Matches batch eval when equal to number of rerank candidates. "
        "Lower values are usually faster but explore fewer SQL alternatives.",
    )
    max_tables = st.slider(
        "Schema tables (max_tables)",
        min_value=1,
        max_value=10,
        value=6,
        help="How many tables TF-IDF schema linking keeps in the model prompt "
        "(plus FK neighbors). Training used max_tables=10; use 10 on large "
        "schemas. Small DBs (e.g. bookstore) include all tables regardless.",
    )
    submitted = st.form_submit_button("Generate SQL", type="primary")

if submitted and question:
    request_body = {
        "question": question,
        "db_id": selected_db,
        "max_tables": max_tables,
        "num_beams": num_beams,
        "num_candidates": num_beams,
    }

    with st.spinner("Generating SQL…"):
        try:
            resp = requests.post(
                f"{API_BASE}/query",
                json=request_body,
                timeout=120,
                headers=_api_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.HTTPError as e:
            st.error(f"API error {e.response.status_code}: {e.response.text}")
            st.stop()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    st.subheader("Generated SQL")
    st.code(data["sql"], language="sql")

    col1, col2, col3 = st.columns(3)
    col1.metric("Latency", f"{data['latency_ms']:.0f} ms")
    col2.metric("Rows returned", data["num_rows"])
    col3.metric("Rerank mode", data["rerank_mode"])

    st.subheader("Query results")
    if data["executed"]:
        if data["results"]:
            df = pd.DataFrame(
                data["results"],
                columns=data["columns"] if data["columns"] else None,
            )
            st.dataframe(df, use_container_width=True)
            if data.get("truncated"):
                st.caption("Results truncated — showing first 10,000 rows.")
        else:
            st.info("Query executed successfully but returned no rows.")
    else:
        st.error(f"Execution failed: {data.get('error', 'unknown error')}")

    with st.expander("API request (POST /query)", expanded=False):
        st.caption(
            "Sent from the Streamlit server to the FastAPI backend "
            "(not visible in browser DevTools Network tab)."
        )
        st.code(json.dumps(request_body, indent=2), language="json")

elif submitted and not question:
    st.warning("Enter a question before generating SQL.")
