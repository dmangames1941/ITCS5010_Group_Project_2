import argparse
import re

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from rank_bm25 import BM25Plus

if __package__:
    from .ingest import COLLECTION_NAME, DB_DIR
else:
    from ingest import COLLECTION_NAME, DB_DIR


def tokenize(text: str) -> list[str]:
    """Keep API identifiers intact while normalizing case."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def search_docs(
    query: str,
    top_k: int = 5,
    mode: str = "hybrid",
) -> list[dict]:
    """Retrieve passages using vector search or BM25/vector rank fusion."""
    if not query.strip():
        raise ValueError("The search query cannot be empty.")
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20.")
    if mode not in {"vector", "hybrid"}:
        raise ValueError("mode must be vector or hybrid.")
    if not DB_DIR.exists():
        raise RuntimeError("Database not found. Run ingest.py first.")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),
    )

    count = collection.count()
    if count == 0:
        return []

    # Retrieve a larger candidate pool before selecting final results.
    candidate_count = min(count, max(20, top_k * 4))
    vector = collection.query(
        query_texts=[query],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )

    records = {}
    vector_ids = vector["ids"][0]

    for chunk_id, text, metadata, distance in zip(
        vector_ids,
        vector["documents"][0],
        vector["metadatas"][0],
        vector["distances"][0],
    ):
        records[chunk_id] = {
            "chunk_id": chunk_id,
            "text": text,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "distance": float(distance),
        }

    if mode == "vector":
        return [records[key] for key in vector_ids[:top_k]]

    # For this small documentation collection, build BM25 from stored text.
    # Embeddings remain persisted; no documentation is re-embedded here.
    stored = collection.get(include=["documents", "metadatas"])
    tokens = [tokenize(text) for text in stored["documents"]]
    query_tokens = tokenize(query)

    if not query_tokens or not any(tokens):
        return [records[key] for key in vector_ids[:top_k]]

    bm25 = BM25Plus(tokens)
    keyword_scores = bm25.get_scores(query_tokens)
    query_terms = set(query_tokens)

    # Exclude documents with no keyword overlap.
    matching = [
        index
        for index, terms in enumerate(tokens)
        if query_terms.intersection(terms)
    ]
    keyword_order = sorted(
        matching,
        key=lambda index: (-float(keyword_scores[index]), stored["ids"][index]),
    )[:candidate_count]

    keyword_ids = []
    for index in keyword_order:
        chunk_id = stored["ids"][index]
        keyword_ids.append(chunk_id)

        if chunk_id not in records:
            metadata = stored["metadatas"][index]
            records[chunk_id] = {
                "chunk_id": chunk_id,
                "text": stored["documents"][index],
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "distance": None,
            }

    # Reciprocal rank fusion combines ranks, not incompatible raw scores.
    fused_scores = {}
    for ranking in (vector_ids, keyword_ids):
        for rank, chunk_id in enumerate(ranking, start=1):
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
            )

    ordered_ids = sorted(
        fused_scores,
        key=lambda key: (-fused_scores[key], key),
    )[:top_k]

    return [
        {**records[key], "rrf_score": fused_scores[key]}
        for key in ordered_ids
    ]


def main():
    """Compare baseline vector retrieval with hybrid retrieval."""
    parser = argparse.ArgumentParser(description="Search indexed documentation.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mode", choices=["vector", "hybrid"], default="hybrid"
    )
    args = parser.parse_args()

    passages = search_docs(args.query, args.top_k, args.mode)
    print(f"Retrieval mode: {args.mode}")

    for index, passage in enumerate(passages, start=1):
        print(f"\nResult {index}")
        print(f"Source: {passage['source']}")
        print(f"Chunk index: {passage['chunk_index']}")

        if passage["distance"] is not None:
            print(f"Vector distance: {passage['distance']:.4f}")

        if "rrf_score" in passage:
            print(f"Fusion score: {passage['rrf_score']:.6f}")

        print(passage["text"])

    if not passages:
        print("No indexed passages found.")


if __name__ == "__main__":
    main()