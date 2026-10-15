from pathlib import Path
import hashlib

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


# Resolve paths relative to this script, regardless of the terminal directory.
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "library_docs"


def split_text(text, chunk_size=1000, overlap=150):
    """Split text into overlapping character-based chunks."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("Overlap must be between 0 and chunk_size - 1.")

    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

        start += chunk_size - overlap

    return chunks


def main():
    """Read documentation files and store their chunks and embeddings."""
    files = sorted(
        path
        for path in DOCS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    )

    if not files:
        raise SystemExit("Add .txt or .md documents to the docs folder first.")

    client = chromadb.PersistentClient(path=str(DB_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),
    )

    for path in files:
        text = path.read_text(encoding="utf-8")
        chunks = split_text(text)

        if not chunks:
            print(f"Skipped empty document: {path.name}")
            continue

        source = path.relative_to(DOCS_DIR).as_posix()
        source_id = hashlib.sha256(source.encode("utf-8")).hexdigest()

        # Replace previously indexed chunks belonging to this document.
        collection.delete(where={"source": source})

        # Store chunks in batches of up to 100.
        for offset in range(0, len(chunks), 100):
            batch = chunks[offset:offset + 100]

            collection.upsert(
                ids=[
                    f"{source_id}:{offset + index}"
                    for index in range(len(batch))
                ],
                documents=batch,
                metadatas=[
                    {
                        "source": source,
                        "chunk_index": offset + index,
                    }
                    for index in range(len(batch))
                ],
            )

        print(f"Indexed {source}: {len(chunks)} chunks")

    print(f"Total stored chunks: {collection.count()}")


if __name__ == "__main__":
    main()