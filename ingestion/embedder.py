"""
ChromaDB Embedding Pipeline
============================
Chunks SEC filing text, embeds each chunk with sentence-transformers,
and loads everything into a persistent ChromaDB vector store.

Usage:
    python -m ingestion.embedder                        # embed all downloaded filings
    python -m ingestion.embedder --tickers KEY RF       # embed specific tickers
    python -m ingestion.embedder --query "loan loss provisions"  # test a search
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Iterator

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from ingestion.sec_downloader import get_filing_text

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# all-MiniLM-L6-v2: 384-dim embeddings, fast on CPU, strong quality/speed tradeoff
# Downloads ~90MB on first use, then cached locally
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800        # characters per chunk (~200 tokens)
CHUNK_OVERLAP = 200     # overlap between consecutive chunks
COLLECTION_NAME = "sec_filings"


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Iterator[str]:
    """
    Split text into overlapping chunks of fixed character size.

    Why overlap? If a key sentence falls at a chunk boundary, without overlap
    it would be split across two chunks and both would lose context. Overlap
    ensures every sentence appears complete in at least one chunk.

    Args:
        text:       The full document text to chunk
        chunk_size: Target characters per chunk
        overlap:    Characters to repeat from the end of each chunk
                    at the start of the next chunk

    Yields:
        Individual text chunks
    """
    if not text.strip():
        return

    start = 0
    while start < len(text):
        end = start + chunk_size

        # If we're not at the end of the document, try to break at a
        # sentence boundary (period followed by space or newline)
        # rather than mid-sentence
        if end < len(text):
            # Look back up to 100 chars for a good break point
            break_search = text[end - 100 : end]
            last_period = max(
                break_search.rfind(". "),
                break_search.rfind(".\n"),
            )
            if last_period != -1:
                end = (end - 100) + last_period + 1  # include the period

        chunk = text[start:end].strip()
        if chunk:
            yield chunk

        # Move forward, but back up by 'overlap' to create the overlap window
        start = end - overlap
        if start >= len(text):
            break


# ── ChromaDB Client ───────────────────────────────────────────────────────────

def get_chroma_client(chroma_path: str | None = None) -> chromadb.PersistentClient:
    """
    Create or connect to a persistent ChromaDB instance.

    'Persistent' means the vector store is saved to disk — when you restart
    the program, all your embeddings are still there. No re-embedding needed.
    """
    if chroma_path is None:
        chroma_path = os.getenv("CHROMA_PATH", "./data/chroma")

    Path(chroma_path).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )
    return client


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """
    Get the SEC filings collection, creating it if it doesn't exist.

    A ChromaDB 'collection' is like a table in a SQL database — it groups
    related vectors together. We use one collection for all SEC filings
    and use metadata filters to narrow searches by ticker, year, etc.
    """
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
    )
    return collection


# ── Embedding ─────────────────────────────────────────────────────────────────

def load_embedding_model() -> SentenceTransformer:
    """
    Load the sentence-transformers model.
    First call downloads ~90MB. Subsequent calls load from cache.
    """
    log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    log.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def embed_filing(
    ticker: str,
    filing_path: Path,
    filing_type: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    batch_size: int = 64,
) -> int:
    """
    Process one SEC filing: chunk it, embed it, and upsert into ChromaDB.

    Args:
        ticker:       Stock ticker (e.g. 'KEY')
        filing_path:  Path to the raw filing file
        filing_type:  '10-K' or '10-Q'
        collection:   ChromaDB collection to insert into
        model:        Loaded SentenceTransformer model
        batch_size:   How many chunks to embed at once (tune for memory)

    Returns:
        Number of chunks inserted
    """
    log.info(f"Processing {ticker} {filing_type}: {filing_path.name}")

    # Step 1: Extract clean text from the raw filing
    text = get_filing_text(filing_path)
    if not text:
        log.warning(f"  Empty text extracted from {filing_path}")
        return 0

    log.info(f"  Extracted {len(text):,} characters of clean text")

    # Step 2: Chunk the text
    chunks = list(chunk_text(text))
    log.info(f"  Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Step 3: Build metadata for each chunk
    # This metadata lets us filter searches: "only search KEY's filings"
    # Extract year from folder path (accession number contains the date)
    year = _extract_year_from_path(filing_path)

    # Step 4: Embed + upsert in batches
    total_inserted = 0
    for batch_start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[batch_start : batch_start + batch_size]

        # Generate unique IDs for each chunk
        # Format: {ticker}_{filing_type}_{year}_{chunk_index}
        ids = [
            f"{ticker}_{filing_type}_{year}_{batch_start + i}"
            for i in range(len(batch_chunks))
        ]

        # Build metadata list — one dict per chunk
        metadatas = [
            {
                "ticker": ticker,
                "filing_type": filing_type,
                "year": year,
                "chunk_index": batch_start + i,
                "source": f"{ticker} {filing_type} {year}",
            }
            for i in range(len(batch_chunks))
        ]

        # Embed: convert list of strings → list of float vectors
        embeddings = model.encode(batch_chunks, show_progress_bar=False).tolist()

        # Upsert: insert or update (safe to re-run without creating duplicates)
        collection.upsert(
            ids=ids,
            documents=batch_chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_inserted += len(batch_chunks)

        if batch_start % (batch_size * 5) == 0:
            log.info(f"  Progress: {total_inserted}/{len(chunks)} chunks embedded...")

    log.info(f"  Done: {total_inserted} chunks upserted for {ticker} {filing_type} {year}")
    return total_inserted


def _extract_year_from_path(filing_path: Path) -> str:
    """
    Extract the filing year from the accession number directory.
    Accession numbers look like: 0001628280-25-010546
    The first 4 digits after the first dash are not the year —
    instead, look at the parent folder name or fall back to 'unknown'.
    """
    # Walk up parents to find the accession number folder
    for part in filing_path.parts:
        # Accession numbers: 18 chars, format XXXXXXXXXX-YY-XXXXXX
        if len(part) == 20 and part.count("-") == 2:
            year_short = part.split("-")[1]
            # '25' → '2025', '24' → '2024'
            return "20" + year_short if len(year_short) == 2 else year_short
    return "unknown"


# ── Query / Search ────────────────────────────────────────────────────────────

def search(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    n_results: int = 5,
    ticker_filter: str | None = None,
    filing_type_filter: str | None = None,
) -> list[dict]:
    """
    Semantic search over the embedded filings.

    Args:
        query:               Natural language question
        collection:          ChromaDB collection to search
        model:               Loaded SentenceTransformer model
        n_results:           Number of top results to return
        ticker_filter:       Restrict to a specific company (e.g. 'KEY')
        filing_type_filter:  Restrict to '10-K' or '10-Q'

    Returns:
        List of dicts with 'text', 'metadata', and 'distance' keys
    """
    # Embed the query using the same model used for documents
    query_embedding = model.encode(query).tolist()

    # Build optional metadata filter
    where = {}
    if ticker_filter and filing_type_filter:
        where = {"$and": [{"ticker": ticker_filter}, {"filing_type": filing_type_filter}]}
    elif ticker_filter:
        where = {"ticker": ticker_filter}
    elif filing_type_filter:
        where = {"filing_type": filing_type_filter}

    # Query ChromaDB
    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    # Flatten the nested result structure into a clean list
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text": doc,
            "metadata": meta,
            "distance": round(dist, 4),  # lower = more similar
        })

    return output


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_ingestion(
    tickers: list[str] | None = None,
    filing_type: str = "10-K",
    filings_dir: str | None = None,
    chroma_path: str | None = None,
) -> None:
    """
    Full ingestion pipeline: find all downloaded filings, embed them all.
    """
    if filings_dir is None:
        filings_dir = os.getenv("FILINGS_PATH", "./data/filings")

    filings_root = Path(filings_dir) / "sec-edgar-filings"

    if not filings_root.exists():
        log.error(f"Filings directory not found: {filings_root}")
        log.error("Run Phase 2 first: python -m ingestion.sec_downloader")
        return

    # Load model once — reuse across all filings
    model = load_embedding_model()

    # Connect to ChromaDB
    client = get_chroma_client(chroma_path)
    collection = get_or_create_collection(client)

    log.info(f"ChromaDB collection '{COLLECTION_NAME}' currently has "
             f"{collection.count()} chunks")

    # Find all filing files to process
    total_chunks = 0
    processed = 0

    for ticker_dir in sorted(filings_root.iterdir()):
        ticker = ticker_dir.name

        if tickers and ticker not in tickers:
            continue

        form_dir = ticker_dir / filing_type
        if not form_dir.exists():
            log.warning(f"No {filing_type} filings found for {ticker}")
            continue

        for filing_file in sorted(form_dir.rglob("full-submission.txt")):
            chunks_added = embed_filing(
                ticker=ticker,
                filing_path=filing_file,
                filing_type=filing_type,
                collection=collection,
                model=model,
            )
            total_chunks += chunks_added
            processed += 1

    log.info(f"\nIngestion complete: {processed} filing(s), {total_chunks} chunks added")
    log.info(f"ChromaDB collection now has {collection.count()} total chunks")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Embed SEC filings into ChromaDB")
    parser.add_argument("--tickers", nargs="+", help="Only embed these tickers")
    parser.add_argument("--filing-type", default="10-K", choices=["10-K", "10-Q"])
    parser.add_argument("--query", help="Test a semantic search query after ingestion")
    parser.add_argument("--skip-ingestion", action="store_true",
                        help="Skip embedding, only run a test query")
    args = parser.parse_args()

    if not args.skip_ingestion:
        run_ingestion(tickers=args.tickers, filing_type=args.filing_type)

    if args.query:
        log.info(f"\nRunning test query: '{args.query}'")
        model = load_embedding_model()
        client = get_chroma_client()
        collection = get_or_create_collection(client)

        results = search(args.query, collection, model, n_results=3)
        print(f"\n{'='*60}")
        print(f"Top {len(results)} results for: \"{args.query}\"")
        print(f"{'='*60}")
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            print(f"\n[{i}] {meta['source']}  (distance: {r['distance']})")
            print(f"    {r['text'][:300]}...")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
