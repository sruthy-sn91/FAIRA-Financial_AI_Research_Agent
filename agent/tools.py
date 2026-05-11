"""
Agent Tool Definitions
======================
Each function here is a tool the LangGraph agent can call autonomously.

LangChain's @tool decorator does two things:
  1. Wraps the function so the LLM can call it by name
  2. Uses the docstring as the tool description the LLM reads to decide WHEN to call it

Rule: every tool must have a clear, specific docstring — the LLM uses it to decide
which tool to call and what arguments to pass.
"""

import json
import logging
import os
from datetime import datetime, timedelta

import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred
from langchain_core.tools import tool

from ingestion.embedder import get_chroma_client, get_or_create_collection, load_embedding_model, search

load_dotenv()
log = logging.getLogger(__name__)

# ── Shared state (loaded once, reused across tool calls) ──────────────────────

_embedding_model = None
_chroma_collection = None


def _get_rag_components():
    """Lazy-load the embedding model and ChromaDB collection."""
    global _embedding_model, _chroma_collection
    if _embedding_model is None:
        _embedding_model = load_embedding_model()
    if _chroma_collection is None:
        client = get_chroma_client()
        _chroma_collection = get_or_create_collection(client)
    return _embedding_model, _chroma_collection


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1: SEC Filing RAG Search
# ══════════════════════════════════════════════════════════════════════════════

@tool
def search_sec_filings(query: str, ticker: str = "", n_results: int = 5) -> str:
    """
    Search SEC EDGAR filings (10-K annual reports) using semantic search.
    Use this tool to find specific information from company filings such as:
    risk factors, financial data, management commentary, loan portfolio details,
    capital ratios, net interest margin, credit quality, and regulatory disclosures.

    Args:
        query:     Natural language question about what you're looking for.
                   Example: "net interest margin compression 2024"
        ticker:    Optional stock ticker to restrict search to one company.
                   Example: "KEY" for KeyCorp. Leave empty to search all companies.
        n_results: Number of relevant passages to return (default 5).

    Returns:
        Formatted string with relevant passages and their citations.
    """
    model, collection = _get_rag_components()

    results = search(
        query=query,
        collection=collection,
        model=model,
        n_results=n_results,
        ticker_filter=ticker if ticker else None,
    )

    if not results:
        return f"No relevant passages found for query: '{query}'"

    output_parts = [f"SEC Filing Search Results for: '{query}'\n{'='*60}"]
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        source = meta.get("source", "Unknown")
        output_parts.append(
            f"\n[Source {i}] {source} (relevance score: {1 - r['distance']:.2%})\n"
            f"{r['text']}\n"
            f"{'-'*40}"
        )

    return "\n".join(output_parts)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2: FRED Macroeconomic Data
# ══════════════════════════════════════════════════════════════════════════════

# Common FRED series IDs for banking/credit research
FRED_SERIES_REFERENCE = """
Common FRED series for banking research:
  FEDFUNDS    - Federal Funds Effective Rate
  DGS10       - 10-Year Treasury Constant Maturity Rate
  DGS2        - 2-Year Treasury Constant Maturity Rate
  T10Y2Y      - 10-Year minus 2-Year Treasury Spread (yield curve)
  DPCREDIT    - Discount Window Primary Credit Rate
  CPIAUCSL    - Consumer Price Index (inflation)
  UNRATE      - Unemployment Rate
  DRTSCILM    - Net % of banks tightening C&I loan standards (large/medium)
  DRTSCLCC    - Net % of banks tightening credit card standards
  DRCCLACBS   - Credit Card Delinquency Rate
  DRSFRMACBS  - Single-Family Mortgage Delinquency Rate
  TOTLL       - Total Loans and Leases at Commercial Banks
  USROA       - Return on Assets (all US banks)
  USROE       - Return on Equity (all US banks)
"""


@tool
def get_fred_data(series_id: str, start_date: str = "", end_date: str = "") -> str:
    """
    Fetch macroeconomic time series data from the Federal Reserve (FRED).
    Use this tool to get interest rates, economic indicators, and banking
    system data that provides macroeconomic context for financial analysis.

    Args:
        series_id:  FRED series identifier. Common ones:
                    FEDFUNDS (Fed Funds Rate), DGS10 (10Y Treasury),
                    T10Y2Y (yield curve spread), CPIAUCSL (CPI inflation),
                    UNRATE (unemployment), DRCCLACBS (credit card delinquency),
                    TOTLL (total bank loans), USROA (bank return on assets).
        start_date: Start date in YYYY-MM-DD format. Defaults to 2 years ago.
        end_date:   End date in YYYY-MM-DD format. Defaults to today.

    Returns:
        Formatted string with recent data points and summary statistics.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return "Error: FRED_API_KEY not set in environment"

    fred = Fred(api_key=api_key)

    # Default date range: last 2 years
    if not end_date:
        end_date = datetime.today().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")

    try:
        series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)

        if series.empty:
            return f"No data found for FRED series '{series_id}' in the date range {start_date} to {end_date}"

        # Get series metadata
        info = fred.get_series_info(series_id)
        series_name = info.get("title", series_id)
        units = info.get("units_short", "")

        # Build summary statistics
        latest_value = series.dropna().iloc[-1]
        latest_date = series.dropna().index[-1].strftime("%Y-%m-%d")
        mean_val = series.dropna().mean()
        min_val = series.dropna().min()
        max_val = series.dropna().max()

        # Build recent observations table (last 12 data points)
        recent = series.dropna().tail(12)
        obs_lines = [f"  {date.strftime('%Y-%m-%d')}: {val:.4f} {units}"
                     for date, val in recent.items()]

        output = (
            f"FRED Data: {series_name} ({series_id})\n"
            f"{'='*60}\n"
            f"Latest Value:  {latest_value:.4f} {units} (as of {latest_date})\n"
            f"Period Mean:   {mean_val:.4f} {units}\n"
            f"Period Min:    {min_val:.4f} {units}\n"
            f"Period Max:    {max_val:.4f} {units}\n"
            f"Date Range:    {start_date} to {end_date}\n"
            f"\nRecent Observations:\n"
            + "\n".join(obs_lines)
        )
        return output

    except Exception as e:
        return (
            f"Error fetching FRED series '{series_id}': {e}\n\n"
            f"{FRED_SERIES_REFERENCE}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3: yfinance — Stock & Fundamental Data
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_stock_data(ticker: str, period: str = "1y") -> str:
    """
    Fetch stock price history and key financial fundamentals for a company
    using Yahoo Finance. Use this for current valuation metrics, price trends,
    and balance sheet ratios not found in SEC filings.

    Args:
        ticker: Stock ticker symbol. Examples: KEY, RF, FITB, HBAN, CFG,
                JPM, BAC, WFC for banks. Also works for any public company.
        period: Time period for price history. Options:
                '1mo', '3mo', '6mo', '1y' (default), '2y', '5y', 'ytd'

    Returns:
        Formatted string with price history summary and key financial metrics.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info

        if hist.empty:
            return f"No price data found for ticker '{ticker}'"

        # Price summary
        current_price = hist["Close"].iloc[-1]
        start_price = hist["Close"].iloc[0]
        price_change_pct = ((current_price - start_price) / start_price) * 100
        high_52w = hist["Close"].max()
        low_52w = hist["Close"].min()

        # Key financial metrics from info dict
        metrics = {
            "Market Cap":           _fmt_large_num(info.get("marketCap")),
            "P/E Ratio (TTM)":      _fmt_num(info.get("trailingPE")),
            "P/B Ratio":            _fmt_num(info.get("priceToBook")),
            "Dividend Yield":       _fmt_already_pct(info.get("dividendYield")),
            "EPS (TTM)":            _fmt_num(info.get("trailingEps")),
            "Revenue (TTM)":        _fmt_large_num(info.get("totalRevenue")),
            "Net Income":           _fmt_large_num(info.get("netIncomeToCommon")),
            "Return on Equity":     _fmt_pct(info.get("returnOnEquity")),
            "Return on Assets":     _fmt_pct(info.get("returnOnAssets")),
            "Debt/Equity":          _fmt_num(info.get("debtToEquity")),
            "Book Value/Share":     _fmt_num(info.get("bookValue")),
            "Beta":                 _fmt_num(info.get("beta")),
        }

        metrics_lines = "\n".join(
            f"  {k:<22} {v}" for k, v in metrics.items() if v != "N/A"
        )

        # Recent price table (last 10 trading days)
        recent_prices = hist["Close"].tail(10)
        price_lines = [
            f"  {date.strftime('%Y-%m-%d')}: ${price:.2f}"
            for date, price in recent_prices.items()
        ]

        company_name = info.get("longName", ticker)

        output = (
            f"Stock Data: {company_name} ({ticker.upper()})\n"
            f"{'='*60}\n"
            f"Current Price:  ${current_price:.2f}\n"
            f"52W High:       ${high_52w:.2f}\n"
            f"52W Low:        ${low_52w:.2f}\n"
            f"Period Return:  {price_change_pct:+.2f}% ({period})\n"
            f"\nKey Metrics:\n{metrics_lines}\n"
            f"\nRecent Closing Prices:\n" + "\n".join(price_lines)
        )
        return output

    except Exception as e:
        return f"Error fetching data for '{ticker}': {e}"


def _fmt_num(val) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_already_pct(val) -> str:
    """For values yfinance returns already in % form (e.g. 4.22 meaning 4.22%)."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_large_num(val) -> str:
    if val is None:
        return "N/A"
    try:
        v = float(val)
        if abs(v) >= 1e12:
            return f"${v/1e12:.2f}T"
        elif abs(v) >= 1e9:
            return f"${v/1e9:.2f}B"
        elif abs(v) >= 1e6:
            return f"${v/1e6:.2f}M"
        else:
            return f"${v:.0f}"
    except (TypeError, ValueError):
        return "N/A"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4: e2b — Sandboxed Python Code Execution
# ══════════════════════════════════════════════════════════════════════════════

@tool
def run_python_calculation(code: str) -> str:
    """
    Execute Python code in a secure sandboxed environment for financial
    calculations. Use this tool when you need to:
    - Calculate financial ratios (NIM, ROE, loan-to-deposit ratio, etc.)
    - Perform statistical analysis on data you've already retrieved
    - Create comparative tables or summaries across multiple companies
    - Run any numerical computation

    The sandbox has access to: pandas, numpy, math, statistics, json.
    Do NOT try to import external libraries beyond these.
    All output is via print() statements.

    Args:
        code: Valid Python code to execute. Use print() for all output.

    Example code:
        ```python
        nim_data = {"KEY": 2.47, "RF": 3.52, "FITB": 2.89, "HBAN": 3.01, "CFG": 2.95}
        avg_nim = sum(nim_data.values()) / len(nim_data)
        print(f"Average NIM across regional banks: {avg_nim:.2f}%")
        for ticker, nim in sorted(nim_data.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ticker}: {nim:.2f}% ({'above' if nim > avg_nim else 'below'} average)")
        ```

    Returns:
        Output from the code execution (stdout), or error message.
    """
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        return "Error: E2B_API_KEY not set in environment"

    try:
        from e2b_code_interpreter import Sandbox

        # New e2b API: use Sandbox.create(); reads E2B_API_KEY from env automatically
        sandbox = Sandbox.create()
        try:
            execution = sandbox.run_code(code)
        finally:
            sandbox.kill()

            output_parts = []

            # Collect stdout
            if execution.logs.stdout:
                output_parts.append("Output:\n" + "\n".join(execution.logs.stdout))

            # Collect stderr (non-fatal warnings)
            if execution.logs.stderr:
                output_parts.append("Warnings:\n" + "\n".join(execution.logs.stderr))

            # Check for execution errors
            if execution.error:
                return (
                    f"Execution Error: {execution.error.name}\n"
                    f"{execution.error.value}\n"
                    f"Traceback:\n{execution.error.traceback}"
                )

            return "\n".join(output_parts) if output_parts else "Code executed successfully (no output)"

    except ImportError:
        return "Error: e2b_code_interpreter not installed. Run: pip install e2b-code-interpreter"
    except Exception as e:
        return f"Sandbox error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 5: Multi-stock comparison
# ══════════════════════════════════════════════════════════════════════════════

@tool
def compare_stocks(tickers: list[str], metric: str = "price") -> str:
    """
    Compare multiple stocks side-by-side on a specific metric.
    Useful for peer group analysis across regional banks or sector comparison.

    Args:
        tickers: List of ticker symbols. Example: ["KEY", "RF", "FITB", "HBAN", "CFG"]
        metric:  What to compare. Options:
                 'price'      - current price and YTD return
                 'valuation'  - P/E, P/B, dividend yield
                 'profitability' - ROE, ROA, net income
                 'size'       - market cap, revenue

    Returns:
        Formatted comparison table.
    """
    rows = []

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            rows.append({
                "ticker":     ticker.upper(),
                "name":       info.get("shortName", ticker)[:20],
                "price":      info.get("currentPrice") or info.get("regularMarketPrice"),
                "mkt_cap":    info.get("marketCap"),
                "pe":         info.get("trailingPE"),
                "pb":         info.get("priceToBook"),
                "div_yield":  info.get("dividendYield"),   # already in % form from yfinance
                "roe":        info.get("returnOnEquity"),
                "roa":        info.get("returnOnAssets"),
                "revenue":    info.get("totalRevenue"),
            })
        except Exception as e:
            rows.append({"ticker": ticker, "name": f"Error: {e}"})

    if not rows:
        return "No data retrieved"

    if metric == "valuation":
        header = f"{'Ticker':<6}  {'Name':<20}  {'Price':>8}  {'P/E':>7}  {'P/B':>7}  {'Div Yield':>10}"
        sep = "-" * len(header)
        lines = [header, sep]
        for r in rows:
            lines.append(
                f"{r['ticker']:<6}  {r['name']:<20}  "
                f"${r['price'] or 0:>7.2f}  "
                f"{_fmt_num(r['pe']):>7}  "
                f"{_fmt_num(r['pb']):>7}  "
                f"{_fmt_already_pct(r['div_yield']):>10}"
            )

    elif metric == "profitability":
        header = f"{'Ticker':<6}  {'Name':<20}  {'ROE':>8}  {'ROA':>8}  {'Revenue':>12}"
        sep = "-" * len(header)
        lines = [header, sep]
        for r in rows:
            lines.append(
                f"{r['ticker']:<6}  {r['name']:<20}  "
                f"{_fmt_pct(r['roe']):>8}  "
                f"{_fmt_pct(r['roa']):>8}  "
                f"{_fmt_large_num(r['revenue']):>12}"
            )

    elif metric == "size":
        header = f"{'Ticker':<6}  {'Name':<20}  {'Mkt Cap':>12}  {'Revenue':>12}"
        sep = "-" * len(header)
        lines = [header, sep]
        for r in rows:
            lines.append(
                f"{r['ticker']:<6}  {r['name']:<20}  "
                f"{_fmt_large_num(r['mkt_cap']):>12}  "
                f"{_fmt_large_num(r['revenue']):>12}"
            )

    else:  # default: price
        header = f"{'Ticker':<6}  {'Name':<20}  {'Price':>8}  {'Mkt Cap':>12}"
        sep = "-" * len(header)
        lines = [header, sep]
        for r in rows:
            lines.append(
                f"{r['ticker']:<6}  {r['name']:<20}  "
                f"${r['price'] or 0:>7.2f}  "
                f"{_fmt_large_num(r['mkt_cap']):>12}"
            )

    return f"Peer Comparison ({metric.title()})\n{'='*60}\n" + "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Tool registry — import this in the agent
# ══════════════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    search_sec_filings,
    get_fred_data,
    get_stock_data,
    compare_stocks,
    run_python_calculation,
]
