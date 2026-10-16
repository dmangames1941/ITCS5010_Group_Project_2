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
    """Group Markdown paragraphs and fenced code blocks into chunks.

    chunk_size is a soft limit: a single code block may exceed it.
    overlap is retained for compatibility but is not used here.
    """
    blocks = []
    current_block = []
    fence_char = None
    fence_length = 0

    for line in text.lstrip("\ufeff").splitlines():
        stripped = line.lstrip()

        # Recognize Markdown fences, including fences with language labels.
        if stripped.startswith(("```", "~~~")):
            char = stripped[0]
            length = len(stripped) - len(stripped.lstrip(char))

            if fence_char is None:
                if current_block:
                    blocks.append("\n".join(current_block).strip())
                    current_block = []

                fence_char = char
                fence_length = length
                current_block.append(line)
                continue

            if char == fence_char and length >= fence_length:
                current_block.append(line)
                blocks.append("\n".join(current_block).strip())
                current_block = []
                fence_char = None
                fence_length = 0
                continue

        if fence_char is not None:
            current_block.append(line)
            continue

        # Start a new block at a heading or blank line.
        if not stripped or stripped.startswith("#"):
            if current_block:
                blocks.append("\n".join(current_block).strip())
                current_block = []

            if stripped:
                current_block.append(line)
        else:
            current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block).strip())

    chunks = []
    current_chunk = ""

    for block in blocks:
        if not block:
            continue

        candidate = (
            f"{current_chunk}\n\n{block}" if current_chunk else block
        )

        if current_chunk and len(candidate) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = block
        else:
            current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk)

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