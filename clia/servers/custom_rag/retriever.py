import argparse

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Support both package imports and running this file directly.
if __package__:
    from .ingest import COLLECTION_NAME, DB_DIR
else:
    from ingest import COLLECTION_NAME, DB_DIR


def search_docs(query: str, top_k: int = 5) -> list[dict]:
    """Search stored documentation and return passages with source metadata."""
    if not query.strip():
        raise ValueError("The search query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

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

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    passages = []

    for chunk_id, text, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        passages.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                "distance": float(distance),
            }
        )

    return passages


def main():
    """Allow retrieval to be tested from the terminal."""
    parser = argparse.ArgumentParser(description="Search indexed documentation.")
    parser.add_argument("query", help="Question to search for")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    passages = search_docs(args.query, args.top_k)

    if not passages:
        print("No indexed passages found.")
        return

    for index, passage in enumerate(passages, start=1):
        print(f"\nResult {index}")
        print(f"Source: {passage['source']}")
        print(f"Chunk index: {passage['chunk_index']}")
        print(f"Distance: {passage['distance']:.4f}")
        print(passage["text"])


if __name__ == "__main__":
    main()