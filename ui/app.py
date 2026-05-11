"""
Financial AI Research Agent — Streamlit UI
==========================================
A professional research interface that streams agent output in real time.

Run with:
    streamlit run ui/app.py
"""

import json
import os
import time

import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="F.A.I.R.A.",
    page_icon="F",  # overridden immediately by canvas favicon below
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 3px 0;
    }
    .status-sec    { background: #e8f4fd; color: #1565c0; }
    .status-fred   { background: #e8f5e9; color: #2e7d32; }
    .status-market { background: #fff3e0; color: #e65100; }
    .status-calc   { background: #f3e5f5; color: #6a1b9a; }
    .status-other  { background: #f5f5f5; color: #424242; }

    /* Report container */
    .report-container {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem;
        line-height: 1.7;
    }

    /* Source chip */
    .source-chip {
        display: inline-block;
        background: #e3f2fd;
        color: #0d47a1;
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 0.78rem;
        margin: 2px 3px;
        font-family: monospace;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 0.8rem;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_url = st.text_input(
        "API Server URL",
        value=API_BASE,
        help="URL of the FastAPI backend"
    )

    provider = st.selectbox(
        "LLM Provider",
        ["ollama (local)", "groq (cloud)"],
        help="Switch in .env: LLM_PROVIDER=ollama or groq"
    )

    st.divider()

    st.markdown("## 📋 Example Queries")
    example_queries = [
        "Analyze the credit risk of regional banks given the current interest rate environment",
        "Compare net interest margin trends across KeyCorp, Regions Financial, and Fifth Third",
        "What are the key risk factors disclosed by Huntington Bancshares in their latest 10-K?",
        "How has the Fed Funds rate affected regional bank profitability since 2022?",
        "Analyze loan loss provisions and credit quality at Citizens Financial Group",
    ]

    for q in example_queries:
        if st.button(q[:55] + "...", use_container_width=True, key=f"ex_{q[:20]}"):
            st.session_state.query = q

    st.divider()

    # Health check
    st.markdown("## 🔌 Server Status")
    try:
        resp = requests.get(f"{api_url}/health", timeout=3)
        if resp.status_code == 200:
            info = resp.json()
            st.success(f"Connected ({info.get('llm_provider', '?')})")
        else:
            st.error("Server error")
    except Exception:
        st.error("Server offline")
        st.caption(f"Start: `uvicorn api.server:app --port 8000`")

    st.divider()
    st.markdown("**Data Sources**")
    st.caption("📄 SEC EDGAR (10-K filings)")
    st.caption("📈 FRED (Fed macro data)")
    st.caption("💹 Yahoo Finance (market data)")
    st.caption("🧮 e2b (sandboxed calc)")


# ── Main content ──────────────────────────────────────────────────────────────

# ── Favicon: canvas-rendered "FAIRA" text injected into parent doc ────────────
# st.markdown strips <script> tags; components.html() runs inside a same-origin
# iframe so window.parent gives us full access to the parent document.
components.html("""
<script>
(function () {
    var c = document.createElement('canvas');
    c.width = 64; c.height = 64;
    var ctx = c.getContext('2d');
    ctx.fillStyle = '#1565c0';
    ctx.fillRect(0, 0, 64, 64);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 13px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('FAIRA', 32, 32);
    var url = c.toDataURL();

    function applyFavicon() {
        var doc = window.parent.document;
        var ico = doc.querySelector("link[rel*='icon']");
        if (!ico) { ico = doc.createElement('link'); ico.rel = 'icon'; doc.head.appendChild(ico); }
        ico.href = url;
    }
    applyFavicon();
    setTimeout(applyFavicon, 800);  // retry after Streamlit's own favicon injection
})();
</script>
""", height=0)

# ── Animated header: pure CSS (Streamlit allows <style>, strips <script>) ─────
# Session state ensures the animation only plays on first page load; any
# subsequent Streamlit rerun (button click, etc.) renders the static title.
if "title_animated" not in st.session_state:
    st.session_state.title_animated = False

if not st.session_state.title_animated:
    st.session_state.title_animated = True
    st.markdown("""
<style>
.faira-wrapper {
    position: relative;
    height: 3.4rem;
    margin-bottom: 0.2rem;
}
.faira-abbrev {
    position: absolute;
    font-size: 2.2rem;
    font-weight: 700;
    color: #1a1a2e;
    letter-spacing: 0.28em;
    animation: faira-out 0.4s ease 1.6s forwards;
}
.faira-full {
    position: absolute;
    font-size: 2.2rem;
    font-weight: 700;
    color: #1a1a2e;
    white-space: nowrap;
    overflow: hidden;
    opacity: 0;
    max-width: 0;
    animation:
        faira-show 0.1s linear 2.1s forwards,
        faira-type 1.5s steps(31, end) 2.1s forwards;
}
@keyframes faira-out  { to { opacity: 0; } }
@keyframes faira-show { to { opacity: 1; } }
@keyframes faira-type { from { max-width: 0; } to { max-width: 900px; } }
</style>
<div class="faira-wrapper">
    <span class="faira-abbrev">FAIRA</span>
    <span class="faira-full">Financial AI Research Agent</span>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="main-header">Financial AI Research Agent</div>',
        unsafe_allow_html=True,
    )
st.markdown(
    '<div class="sub-header">Autonomous research powered by LangGraph · RAG over SEC filings · '
    'Real-time streaming</div>',
    unsafe_allow_html=True
)

# Query input
if "query" not in st.session_state:
    st.session_state.query = ""

query = st.text_area(
    "Research Question",
    key="query",
    height=90,
    placeholder=(
        "e.g. Analyze the credit risk of regional banks given the current "
        "interest rate environment. Focus on KeyCorp and Regions Financial."
    ),
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    run_button = st.button("▶ Run Research", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("✕ Clear", use_container_width=True)

if clear_button:
    st.session_state.query = ""
    st.rerun()

# ── Research execution ────────────────────────────────────────────────────────

if run_button and query.strip():

    st.divider()

    # Layout: status column + report column
    status_col, report_col = st.columns([1, 2])

    with status_col:
        st.markdown("### 🔍 Agent Activity")
        status_container = st.container()

    with report_col:
        st.markdown("### 📝 Research Report")
        report_placeholder = st.empty()

    # Metrics row (filled after completion)
    metrics_placeholder = st.empty()

    # ── Connect to SSE stream ─────────────────────────────────────────

    accumulated_report = ""
    status_messages = []
    sources = []
    iteration_count = 0
    start_time = time.time()
    success = False

    try:
        with requests.get(
            f"{api_url}/research/stream",
            params={"query": query},
            stream=True,
            timeout=300,
        ) as response:

            if response.status_code != 200:
                st.error(f"API error: HTTP {response.status_code}")
                st.stop()

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue

                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line

                if not line.startswith("data:"):
                    continue

                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                # ── Status event: agent called a tool ────────────────
                if event_type == "status":
                    text = event.get("text", "")
                    tool = event.get("tool", "")

                    # Pick badge color by tool type
                    if "sec" in tool.lower() or "filing" in tool.lower():
                        css_class = "status-sec"
                        icon = "📄"
                    elif "fred" in tool.lower():
                        css_class = "status-fred"
                        icon = "📈"
                    elif "stock" in tool.lower() or "compare" in tool.lower():
                        css_class = "status-market"
                        icon = "💹"
                    elif "calc" in tool.lower() or "python" in tool.lower():
                        css_class = "status-calc"
                        icon = "🧮"
                    else:
                        css_class = "status-other"
                        icon = "⚙️"

                    status_messages.append(
                        f'<div class="status-badge {css_class}">{icon} {text}</div>'
                    )
                    with status_container:
                        st.markdown("\n".join(status_messages), unsafe_allow_html=True)

                # ── Tool result: brief preview ────────────────────────
                elif event_type == "tool_result":
                    sources = event.get("sources", sources)

                # ── Token: stream report word by word ─────────────────
                elif event_type == "token":
                    accumulated_report += event.get("text", "")
                    report_placeholder.markdown(
                        f'<div class="report-container">{accumulated_report}</div>',
                        unsafe_allow_html=True,
                    )

                # ── Complete: final state ─────────────────────────────
                elif event_type == "complete":
                    accumulated_report = event.get("report", accumulated_report)
                    sources = event.get("sources", sources)
                    iteration_count = event.get("iteration_count", iteration_count)
                    elapsed = time.time() - start_time

                    # Final render of full report
                    report_placeholder.markdown(accumulated_report)

                    # Metrics row
                    with metrics_placeholder.container():
                        st.divider()
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Research Time", f"{elapsed:.1f}s")
                        m2.metric("Agent Iterations", iteration_count)
                        m3.metric("Sources Used", len(sources))
                        m4.metric("Report Length", f"{len(accumulated_report):,} chars")

                    # Sources panel
                    if sources:
                        st.markdown("#### 📚 Data Sources")
                        chips = "".join(
                            f'<span class="source-chip">{s}</span>' for s in sources
                        )
                        st.markdown(chips, unsafe_allow_html=True)

                    success = True
                    break

                # ── Error ─────────────────────────────────────────────
                elif event_type == "error":
                    st.error(f"Agent error: {event.get('message', 'Unknown error')}")
                    break

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API server. Make sure it's running:")
        st.code("uvicorn api.server:app --port 8000 --reload")
    except requests.exceptions.Timeout:
        st.error("Request timed out after 5 minutes.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    if success:
        st.success("Research complete.")

elif run_button and not query.strip():
    st.warning("Please enter a research question.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "Financial AI Research Agent · "
    "LangGraph + ChromaDB + SEC EDGAR + FRED · "
    "Built with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
