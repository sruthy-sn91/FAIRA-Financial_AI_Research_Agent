"""
SEC EDGAR Filing Downloader
============================
Downloads 10-K (annual) and 10-Q (quarterly) filings from SEC EDGAR
for a given list of company tickers.

Usage:
    python -m ingestion.sec_downloader
    python -m ingestion.sec_downloader --tickers JPM BAC WFC --filing-type 10-K --limit 2
"""

import argparse
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from sec_edgar_downloader import Downloader

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Default tickers ────────────────────────────────────────────────────────────
# Regional banks — relevant for credit risk analysis demos
DEFAULT_TICKERS = [
    "KEY",   # KeyCorp
    "RF",    # Regions Financial
    "FITB",  # Fifth Third Bancorp
    "HBAN",  # Huntington Bancshares
    "CFG",   # Citizens Financial
]

# Large banks for broader analysis
LARGE_BANK_TICKERS = [
    "JPM",   # JP Morgan Chase
    "BAC",   # Bank of America
    "WFC",   # Wells Fargo
    "C",     # Citigroup
    "GS",    # Goldman Sachs
]


def download_filings(
    tickers: list[str],
    filing_type: str = "10-K",
    limit: int = 3,
    output_dir: str | None = None,
) -> dict[str, list[Path]]:
    """
    Download SEC filings for a list of tickers.

    Args:
        tickers:      List of stock ticker symbols (e.g. ['KEY', 'RF'])
        filing_type:  SEC form type — '10-K' (annual) or '10-Q' (quarterly)
        limit:        How many filings per ticker (most recent first)
        output_dir:   Where to save files. Defaults to $FILINGS_PATH env var.

    Returns:
        Dict mapping ticker -> list of downloaded file paths
    """
    if output_dir is None:
        output_dir = os.getenv("FILINGS_PATH", "./data/filings")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # SEC EDGAR requires a user-agent string identifying who is making requests.
    # This is a legal requirement — always use a real email address.
    dl = Downloader(
        company_name="FinancialAIResearchAgent",
        email_address="research@example.com",
        download_folder=str(output_path),
    )

    results: dict[str, list[Path]] = {}

    for ticker in tickers:
        log.info(f"Downloading {filing_type} filings for {ticker} (limit={limit})...")
        try:
            dl.get(filing_type, ticker, limit=limit)

            # Find the downloaded files for this ticker
            # sec-edgar-downloader saves to: {output}/sec-edgar-filings/{ticker}/{filing_type}/
            ticker_dir = output_path / "sec-edgar-filings" / ticker / filing_type
            if ticker_dir.exists():
                # sec-edgar-downloader saves the full submission as
                # {filing_type}/{ticker}/{accession-number}/full-submission.txt
                # We want the primary document files
                downloaded = sorted(ticker_dir.rglob("*.txt")) + \
                             sorted(ticker_dir.rglob("*.htm")) + \
                             sorted(ticker_dir.rglob("*.html"))

                # Deduplicate and filter out index files
                seen = set()
                files = []
                for f in downloaded:
                    if f.name not in seen and "index" not in f.name.lower():
                        seen.add(f.name)
                        files.append(f)

                results[ticker] = files
                log.info(f"  {ticker}: {len(files)} file(s) saved to {ticker_dir}")
            else:
                log.warning(f"  {ticker}: No files found after download")
                results[ticker] = []

            # Be polite to the SEC servers — they rate-limit aggressive clients
            time.sleep(0.5)

        except Exception as e:
            log.error(f"  {ticker}: Download failed — {e}")
            results[ticker] = []

    return results


def get_filing_text(file_path: Path) -> str:
    """
    Read a filing file and return its text content.
    Handles both plain text (.txt) and HTML (.htm/.html) filings.

    SEC filings before ~2017 are plain text SGML/HTML.
    Newer filings are inline XBRL HTML. We strip tags for both.
    """
    from bs4 import BeautifulSoup

    raw = file_path.read_text(encoding="utf-8", errors="replace")

    # Detect HTML/XBRL by extension OR by content (SEC filings are .txt but contain HTML)
    is_html = file_path.suffix.lower() in (".htm", ".html") or "<html" in raw[:2000].lower()

    if is_html:
        soup = BeautifulSoup(raw, "lxml")
        # Remove non-readable elements (scripts, styles, XBRL metadata tags)
        for tag in soup(["script", "style"]):
            tag.decompose()
        # Remove XBRL inline tags by name prefix (ix:*)
        for tag in soup.find_all(lambda t: t.name and t.name.startswith("ix:")):
            tag.decompose()
        text = soup.get_text(separator="\n")
    else:
        text = raw

    # Normalize whitespace: collapse multiple blank lines into one
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned


def summarize_downloads(results: dict[str, list[Path]]) -> None:
    """Print a summary table of what was downloaded."""
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    total_files = 0
    for ticker, files in results.items():
        total_size = sum(f.stat().st_size for f in files if f.exists()) / 1024 / 1024
        print(f"  {ticker:<8} {len(files):>3} file(s)   {total_size:>6.1f} MB")
        total_files += len(files)
    print("-" * 60)
    print(f"  {'TOTAL':<8} {total_files:>3} file(s)")
    print("=" * 60)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download SEC EDGAR filings for financial analysis"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Stock tickers to download (default: regional banks)",
    )
    parser.add_argument(
        "--filing-type",
        default="10-K",
        choices=["10-K", "10-Q", "8-K"],
        help="SEC form type (default: 10-K)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of filings per ticker, most recent first (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save filings (default: $FILINGS_PATH env var)",
    )
    args = parser.parse_args()

    log.info(f"Starting download: {args.tickers}")
    log.info(f"Filing type: {args.filing_type}, Limit per ticker: {args.limit}")

    results = download_filings(
        tickers=args.tickers,
        filing_type=args.filing_type,
        limit=args.limit,
        output_dir=args.output_dir,
    )
    summarize_downloads(results)


if __name__ == "__main__":
    main()
