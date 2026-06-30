"""
streamlit/app.py — Streamlit dashboard for the LLM Product Normalization Pipeline.

Features:
  - Upload messy CSV → normalize on the fly using the local LLM
  - Preview raw and normalized data
  - Display metrics: total / success / errors / avg confidence
  - Category, Price, Availability, and Confidence distribution charts
  - Download clean CSV and errors CSV
  - Show error records table

Run with:
    streamlit run streamlit/app.py
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on sys.path when running from any directory
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.exporter import errors_to_dataframe, products_to_dataframe
from src.llm_parser import LLMParser, OllamaClient
from src.postprocess import postprocess_product
from src.preprocess import clean_description, preprocess_dataframe
from src.schemas import ErrorRecord, NormalizedProduct, RawProduct
from src.validator import validate_product

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Product Normalizer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0d1117; }

    .metric-card {
        background: linear-gradient(135deg, #1e2433 0%, #252d3d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        color: #58a6ff;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-success .metric-value { color: #3fb950; }
    .metric-error .metric-value   { color: #f85149; }
    .metric-conf .metric-value    { color: #d2a8ff; }

    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #c9d1d9;
        border-left: 4px solid #58a6ff;
        padding-left: 12px;
        margin: 24px 0 12px;
    }

    .stDataFrame { border-radius: 8px; }

    .sidebar-info {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: #8b949e;
    }

    div[data-testid="stAlert"] { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    model_name = st.text_input("Ollama Model", value=settings.ollama_model)
    confidence_threshold = st.slider(
        "Confidence Threshold", min_value=0.0, max_value=1.0,
        value=settings.confidence_threshold, step=0.05,
    )
    max_rows = st.number_input(
        "Max rows to process", min_value=1, max_value=600, value=50, step=10
    )
    st.divider()
    st.markdown(
        '<div class="sidebar-info">'
        "🔒 <b>Fully Offline</b><br>"
        "No cloud API calls. All inference runs locally via Ollama.<br><br>"
        "📦 <b>Model:</b> llama3.2<br>"
        "🐍 <b>Stack:</b> FastAPI · Pydantic · Pandas"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("**Ollama Status**")
    if st.button("🔍 Check Connection"):
        client = OllamaClient(model=model_name)
        if client.is_available():
            st.success(f"✅ {model_name} is ready")
        else:
            st.error(f"❌ {model_name} not found")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding: 32px 0 16px;">
        <h1 style="font-size:2.6rem; font-weight:800;
                   background: linear-gradient(90deg, #58a6ff, #d2a8ff);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🛒 LLM Product Normalizer
        </h1>
        <p style="color:#8b949e; font-size:1.05rem; margin-top:4px;">
            Convert messy e-commerce descriptions → clean structured data · Offline · llama3.2
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_upload, tab_results, tab_errors, tab_charts, tab_about = st.tabs(
    ["📁 Upload & Normalize", "✅ Results", "❌ Errors", "📊 Analytics", "ℹ️ About"]
)

# ── Session state ──────────────────────────────────────────────────────────────
if "successes" not in st.session_state:
    st.session_state.successes: list[NormalizedProduct] = []
if "failures" not in st.session_state:
    st.session_state.failures: list[ErrorRecord] = []
if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False


# ── Helper: metric card ────────────────────────────────────────────────────────
def metric_card(value: str, label: str, css_class: str = "") -> str:
    return (
        f'<div class="metric-card {css_class}">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f"</div>"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Upload & Normalize
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown('<div class="section-header">Upload Raw Product CSV</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload a CSV with a **`raw_description`** column (or any column with 'description' "
        "in the name). The pipeline will clean, parse with llama3.2, validate, and normalise."
    )

    uploaded = st.file_uploader(
        "Drop your CSV here", type=["csv"], label_visibility="collapsed"
    )

    if uploaded:
        raw_df = pd.read_csv(uploaded, dtype=str)
        st.markdown('<div class="section-header">Raw Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(raw_df.head(10), use_container_width=True)
        st.caption(f"📄 {len(raw_df)} total rows detected")

        col_run, col_spacer = st.columns([1, 3])
        with col_run:
            run_btn = st.button("🚀 Run Normalization", type="primary", use_container_width=True)

        if run_btn:
            # Identify desc column
            desc_col = next(
                (c for c in raw_df.columns if "description" in c.lower()),
                raw_df.columns[0],
            )
            raw_df = raw_df.rename(columns={desc_col: "raw_description"})
            raw_df = preprocess_dataframe(raw_df)
            raw_df = raw_df.head(int(max_rows))

            successes: list[NormalizedProduct] = []
            failures: list[ErrorRecord] = []

            parser = LLMParser(client=OllamaClient(model=model_name))

            progress_bar = st.progress(0, text="Initialising…")
            status_text = st.empty()
            total = len(raw_df)

            start_time = time.perf_counter()

            for i, (_, row) in enumerate(raw_df.iterrows()):
                raw = RawProduct(
                    id=int(row["id"]),
                    raw_description=str(row["raw_description"]),
                )
                status_text.markdown(
                    f"⏳ Processing **{i+1}/{total}** — `{raw.raw_description[:60]}…`"
                )

                llm_data, retry_count, llm_raw = parser.parse(
                    raw.raw_description, raw.id
                )

                if llm_data is None:
                    failures.append(
                        ErrorRecord(
                            id=raw.id,
                            raw_description=raw.raw_description,
                            error_type="json_parse_error",
                            error_detail="LLM returned no valid JSON",
                            retry_count=retry_count,
                            llm_raw_output=llm_raw,
                        )
                    )
                else:
                    result = validate_product(raw, llm_data, retry_count, llm_raw)
                    if isinstance(result, NormalizedProduct):
                        successes.append(postprocess_product(result))
                    else:
                        failures.append(result)

                progress_bar.progress((i + 1) / total, text=f"Processed {i+1}/{total}")

            elapsed = time.perf_counter() - start_time
            progress_bar.empty()
            status_text.empty()

            st.session_state.successes = successes
            st.session_state.failures = failures
            st.session_state.pipeline_ran = True

            st.success(
                f"✅ Pipeline complete in **{elapsed:.1f}s** — "
                f"{len(successes)} succeeded · {len(failures)} failed"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Results
# ═══════════════════════════════════════════════════════════════════════════════
with tab_results:
    if not st.session_state.pipeline_ran:
        st.info("Upload a CSV and run normalization to see results here.")
    else:
        successes = st.session_state.successes
        failures = st.session_state.failures
        total = len(successes) + len(failures)
        avg_conf = (
            sum(p.confidence for p in successes) / len(successes)
            if successes else 0.0
        )
        rate = (len(successes) / total * 100) if total else 0.0

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card(str(total), "Total Products"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card(str(len(successes)), "Successful", "metric-success"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card(str(len(failures)), "Failed", "metric-error"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card(f"{avg_conf:.2f}", "Avg Confidence", "metric-conf"), unsafe_allow_html=True)

        st.divider()

        if successes:
            df_clean = products_to_dataframe(successes)
            st.markdown('<div class="section-header">Normalized Products</div>', unsafe_allow_html=True)
            st.dataframe(df_clean, use_container_width=True, height=400)

            # Download
            csv_bytes = df_clean.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="⬇️ Download clean_products.csv",
                data=csv_bytes,
                file_name="clean_products.csv",
                mime="text/csv",
                type="primary",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Errors
# ═══════════════════════════════════════════════════════════════════════════════
with tab_errors:
    if not st.session_state.pipeline_ran:
        st.info("Run the pipeline to see error records here.")
    else:
        failures = st.session_state.failures
        if not failures:
            st.success("🎉 No errors! All records were successfully normalized.")
        else:
            st.warning(f"⚠️ {len(failures)} records failed normalization.")
            df_errors = errors_to_dataframe(failures)
            st.dataframe(df_errors, use_container_width=True, height=350)

            csv_err = df_errors.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="⬇️ Download errors.csv",
                data=csv_err,
                file_name="errors.csv",
                mime="text/csv",
            )

            # Error type breakdown
            if len(df_errors) > 0:
                st.markdown('<div class="section-header">Error Breakdown</div>', unsafe_allow_html=True)
                err_counts = df_errors["error_type"].value_counts().reset_index()
                err_counts.columns = ["error_type", "count"]
                fig = px.bar(
                    err_counts, x="error_type", y="count",
                    color="count", color_continuous_scale="Reds",
                    title="Error Types",
                    template="plotly_dark",
                )
                fig.update_layout(showlegend=False, margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Analytics
# ═══════════════════════════════════════════════════════════════════════════════
with tab_charts:
    if not st.session_state.pipeline_ran:
        st.info("Run the pipeline to see analytics.")
    else:
        successes = st.session_state.successes
        failures = st.session_state.failures

        if not successes:
            st.warning("No successful records to visualize.")
        else:
            df = products_to_dataframe(successes)

            row1_col1, row1_col2 = st.columns(2)

            # Category distribution
            with row1_col1:
                cat_counts = df["category"].value_counts().reset_index()
                cat_counts.columns = ["category", "count"]
                fig_cat = px.pie(
                    cat_counts, values="count", names="category",
                    title="Category Distribution",
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Set3,
                    hole=0.4,
                )
                fig_cat.update_layout(margin=dict(t=40, b=20))
                st.plotly_chart(fig_cat, use_container_width=True)

            # Availability distribution
            with row1_col2:
                avail_counts = df["availability"].value_counts().reset_index()
                avail_counts.columns = ["availability", "count"]
                fig_av = px.bar(
                    avail_counts, x="availability", y="count",
                    color="availability",
                    title="Availability Distribution",
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig_av.update_layout(showlegend=False, margin=dict(t=40, b=20))
                st.plotly_chart(fig_av, use_container_width=True)

            row2_col1, row2_col2 = st.columns(2)

            # Price distribution
            with row2_col1:
                df_price = df[df["price"].notna()].copy()
                df_price["price"] = pd.to_numeric(df_price["price"], errors="coerce")
                df_price = df_price.dropna(subset=["price"])
                if not df_price.empty:
                    fig_price = px.histogram(
                        df_price, x="price", nbins=30,
                        title="Price Distribution (₹)",
                        template="plotly_dark",
                        color_discrete_sequence=["#58a6ff"],
                    )
                    fig_price.update_layout(margin=dict(t=40, b=20))
                    st.plotly_chart(fig_price, use_container_width=True)
                else:
                    st.info("No price data available for chart.")

            # Confidence distribution
            with row2_col2:
                fig_conf = px.histogram(
                    df, x="confidence", nbins=20,
                    title="LLM Confidence Distribution",
                    template="plotly_dark",
                    color_discrete_sequence=["#d2a8ff"],
                )
                fig_conf.update_layout(margin=dict(t=40, b=20))
                st.plotly_chart(fig_conf, use_container_width=True)

            # Success vs Failure donut
            st.markdown('<div class="section-header">Success vs Failure</div>', unsafe_allow_html=True)
            total = len(successes) + len(failures)
            fig_sf = go.Figure(
                data=[
                    go.Pie(
                        labels=["Success", "Failed"],
                        values=[len(successes), len(failures)],
                        hole=0.55,
                        marker_colors=["#3fb950", "#f85149"],
                        textinfo="label+percent",
                    )
                ]
            )
            fig_sf.update_layout(
                template="plotly_dark",
                title=f"Overall Success Rate — {len(successes)}/{total} records",
                margin=dict(t=50, b=20),
                showlegend=True,
            )
            st.plotly_chart(fig_sf, use_container_width=True)

            # Brand breakdown
            st.markdown('<div class="section-header">Top Brands</div>', unsafe_allow_html=True)
            brand_counts = df["brand"].value_counts().head(15).reset_index()
            brand_counts.columns = ["brand", "count"]
            fig_brand = px.bar(
                brand_counts, x="count", y="brand", orientation="h",
                title="Top 15 Brands by Record Count",
                template="plotly_dark",
                color="count",
                color_continuous_scale="Blues",
            )
            fig_brand.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=40, b=20))
            st.plotly_chart(fig_brand, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5 — About
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown(
        """
        ## 🛒 LLM Product Normalizer

        This project demonstrates a **production-grade offline AI pipeline** that converts
        messy e-commerce product descriptions into clean, structured data — entirely without
        cloud APIs.

        ### Architecture
        ```
        Raw CSV → Preprocessor → LLM (Ollama/llama3.2) → Pydantic Validator
                → Post-processor → Clean CSV + Errors CSV
        ```

        ### Tech Stack
        | Component | Technology |
        |-----------|-----------|
        | LLM Inference | Ollama · llama3.2 |
        | Data Validation | Pydantic v2 |
        | Data Processing | Pandas · NumPy |
        | API | FastAPI · Uvicorn |
        | Dashboard | Streamlit · Plotly |
        | Progress | tqdm |
        | Logging | Python logging |
        | Containerisation | Docker · docker compose |

        ### Key Features
        - ✅ **100% offline** — no OpenAI/Gemini API keys required
        - 🔄 **Retry mechanism** — 3 retries with exponential back-off
        - 📋 **Pydantic validation** — strict schema enforcement
        - 🧹 **Smart preprocessing** — unicode normalization, HTML stripping
        - 📊 **Analytics dashboard** — 6 interactive Plotly charts
        - 🐳 **Docker ready** — `docker compose up` to start everything

        ### How to Run
        ```bash
        # 1. Install Ollama and pull the model
        ollama pull llama3.2

        # 2. Install Python dependencies
        pip install -r requirements.txt

        # 3. Run the CLI pipeline
        python -m src.main

        # 4. Or start the API
        uvicorn api.app:app --reload

        # 5. Or launch this dashboard
        streamlit run streamlit/app.py
        ```
        """
    )
