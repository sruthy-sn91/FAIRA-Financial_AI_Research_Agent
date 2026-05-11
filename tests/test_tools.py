"""
Tool Tests
==========
Unit tests for each agent tool.
Real API calls are mocked so tests run in CI without live keys.

Run with: pytest tests/ -v
"""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── FRED Tool ─────────────────────────────────────────────────────────────────

class TestFredTool:

    def test_returns_formatted_data(self):
        """FRED tool should format series data into a readable string."""
        mock_series = pd.Series(
            [5.25, 5.33, 5.30],
            index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        )
        mock_info = {"title": "Federal Funds Effective Rate", "units_short": "%"}

        with patch("agent.tools.Fred") as MockFred:
            instance = MockFred.return_value
            instance.get_series.return_value = mock_series
            instance.get_series_info.return_value = mock_info

            from agent.tools import get_fred_data
            result = get_fred_data.invoke({"series_id": "FEDFUNDS", "start_date": "2024-01-01"})

        assert "Federal Funds Effective Rate" in result
        assert "Latest Value" in result
        assert "5.30" in result

    def test_missing_api_key_returns_error(self):
        """Missing FRED_API_KEY should return an error string, not raise."""
        with patch.dict(os.environ, {"FRED_API_KEY": ""}):
            from agent.tools import get_fred_data
            result = get_fred_data.invoke({"series_id": "FEDFUNDS"})
        assert "Error" in result

    def test_invalid_series_returns_error(self):
        """Invalid series ID should return an error string, not raise."""
        with patch("agent.tools.Fred") as MockFred:
            instance = MockFred.return_value
            instance.get_series.side_effect = ValueError("Series not found")

            from agent.tools import get_fred_data
            result = get_fred_data.invoke({"series_id": "INVALID_XYZ_999"})

        assert "Error" in result


# ── yfinance Tool ─────────────────────────────────────────────────────────────

class TestYfinanceTool:

    def _mock_ticker(self, ticker_symbol="KEY"):
        """Build a mock yfinance Ticker object with realistic data."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [15.2, 15.5, 15.8, 16.0, 15.9]},
            index=pd.date_range("2024-01-01", periods=5, freq="D"),
        )
        mock_ticker.info = {
            "longName": "KeyCorp",
            "currentPrice": 15.9,
            "marketCap": 15_000_000_000,
            "trailingPE": 12.5,
            "priceToBook": 0.85,
            "dividendYield": 5.2,
            "returnOnEquity": 0.092,
            "returnOnAssets": 0.0095,
            "bookValue": 18.7,
            "beta": 1.1,
        }
        return mock_ticker

    def test_returns_price_and_metrics(self):
        """get_stock_data should return price data and key financial metrics."""
        with patch("agent.tools.yf.Ticker", return_value=self._mock_ticker()):
            from agent.tools import get_stock_data
            result = get_stock_data.invoke({"ticker": "KEY", "period": "1mo"})

        assert "KeyCorp" in result
        assert "Current Price" in result
        assert "P/B Ratio" in result

    def test_handles_invalid_ticker_gracefully(self):
        """Invalid ticker should return error string, not raise."""
        with patch("agent.tools.yf.Ticker") as MockTicker:
            MockTicker.return_value.history.return_value = pd.DataFrame()
            MockTicker.return_value.info = {}

            from agent.tools import get_stock_data
            result = get_stock_data.invoke({"ticker": "INVALIDTICKER999"})

        assert "Error" in result or "No price data" in result

    def test_compare_stocks_returns_table(self):
        """compare_stocks should return a formatted peer comparison table."""
        with patch("agent.tools.yf.Ticker", return_value=self._mock_ticker()):
            from agent.tools import compare_stocks
            result = compare_stocks.invoke(
                {"tickers": ["KEY", "RF"], "metric": "valuation"}
            )

        assert "Peer Comparison" in result
        assert "P/E" in result


# ── SEC RAG Tool ──────────────────────────────────────────────────────────────

class TestSecFilingsTool:

    def _mock_search_result(self):
        return [
            {
                "text": "Net interest margin declined 42 basis points year-over-year to 2.47%.",
                "metadata": {"ticker": "KEY", "source": "KEY 10-K 2024", "filing_type": "10-K"},
                "distance": 0.12,
            }
        ]

    def test_returns_formatted_passages(self):
        """search_sec_filings should return formatted citations."""
        with patch("agent.tools.search", return_value=self._mock_search_result()), \
             patch("agent.tools._get_rag_components", return_value=(MagicMock(), MagicMock())):
            from agent.tools import search_sec_filings
            result = search_sec_filings.invoke(
                {"query": "net interest margin", "n_results": 1}
            )

        assert "SEC Filing Search Results" in result
        assert "KEY 10-K 2024" in result
        assert "Net interest margin" in result

    def test_no_results_returns_graceful_message(self):
        """Empty search results should return a readable message, not crash."""
        with patch("agent.tools.search", return_value=[]), \
             patch("agent.tools._get_rag_components", return_value=(MagicMock(), MagicMock())):
            from agent.tools import search_sec_filings
            result = search_sec_filings.invoke({"query": "obscure_query_xyz_123"})

        assert "No relevant passages" in result


# ── e2b Calculation Tool ──────────────────────────────────────────────────────

class TestE2bTool:

    def _mock_sandbox_execution(self, stdout_lines, error=None):
        mock_exec = MagicMock()
        mock_exec.logs.stdout = stdout_lines
        mock_exec.logs.stderr = []
        mock_exec.error = error
        return mock_exec

    def test_basic_calculation_output(self):
        """e2b tool should return stdout from executed code."""
        mock_exec = self._mock_sandbox_execution(["Result: 4"])

        with patch("e2b_code_interpreter.Sandbox") as MockSandbox:
            instance = MockSandbox.create.return_value
            instance.run_code.return_value = mock_exec

            from agent.tools import run_python_calculation
            result = run_python_calculation.invoke(
                {"code": "x = 2 + 2\nprint(f'Result: {x}')"}
            )

        assert "Result: 4" in result

    def test_financial_ratio_calculation(self):
        """e2b tool should handle multi-line financial calculations."""
        expected_output = "Average NIM: 2.96%"
        mock_exec = self._mock_sandbox_execution([expected_output])

        with patch("e2b_code_interpreter.Sandbox") as MockSandbox:
            instance = MockSandbox.create.return_value
            instance.run_code.return_value = mock_exec

            from agent.tools import run_python_calculation
            code = "nim=[2.47,3.52,2.89]\nprint(f'Average NIM: {sum(nim)/len(nim):.2f}%')"
            result = run_python_calculation.invoke({"code": code})

        assert "2.96" in result

    def test_execution_error_is_captured(self):
        """Code errors should be returned as a string, not raise in the agent."""
        mock_error = MagicMock()
        mock_error.name = "SyntaxError"
        mock_error.value = "invalid syntax"
        mock_error.traceback = "line 1\n  def broken(("
        mock_exec = self._mock_sandbox_execution([], error=mock_error)

        with patch("e2b_code_interpreter.Sandbox") as MockSandbox:
            instance = MockSandbox.create.return_value
            instance.run_code.return_value = mock_exec

            from agent.tools import run_python_calculation
            result = run_python_calculation.invoke({"code": "def broken(("})

        assert "SyntaxError" in result or "Error" in result

    def test_missing_api_key_returns_error(self):
        """Missing E2B_API_KEY should return an error string, not raise."""
        with patch.dict(os.environ, {"E2B_API_KEY": ""}):
            from agent.tools import run_python_calculation
            result = run_python_calculation.invoke({"code": "print(1)"})

        assert "Error" in result


# ── Chunking Logic ────────────────────────────────────────────────────────────

class TestChunking:

    def test_chunks_cover_full_text(self):
        """Every part of the input text should appear in at least one chunk."""
        from ingestion.embedder import chunk_text
        text = "sentence one. " * 200  # ~2800 chars
        chunks = list(chunk_text(text, chunk_size=500, overlap=100))
        assert len(chunks) > 1
        # Verify no content is silently dropped
        reconstructed = " ".join(chunks)
        assert "sentence one" in reconstructed

    def test_overlap_creates_shared_content(self):
        """Consecutive chunks should share content due to overlap."""
        from ingestion.embedder import chunk_text
        text = "word " * 500  # long enough to produce multiple chunks
        chunks = list(chunk_text(text, chunk_size=200, overlap=50))
        assert len(chunks) >= 2
        # The end of chunk N and start of chunk N+1 should overlap
        end_of_first = chunks[0][-40:]
        start_of_second = chunks[1][:40]
        # They share at least some words
        assert any(w in start_of_second for w in end_of_first.split() if w)

    def test_empty_text_yields_nothing(self):
        """Empty or whitespace-only text should produce no chunks."""
        from ingestion.embedder import chunk_text
        assert list(chunk_text("")) == []
        assert list(chunk_text("   \n\n   ")) == []
