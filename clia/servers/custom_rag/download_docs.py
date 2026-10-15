"""Download a reproducible local snapshot of selected LangChain docs."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs" / "langchain"

PAGES = [
    "quickstart",
    "tools",
    "models",
    "messages",
    "short-term-memory",
]


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for page in PAGES:
        url = f"https://docs.langchain.com/oss/python/langchain/{page}.md"
        destination = DOCS_DIR / f"{page}.md"

        # Keep existing snapshots consistent across later runs.
        if destination.exists():
            print(f"Keeping existing snapshot: {destination.name}")
            text = destination.read_text(encoding="utf-8-sig")
        else:
            request = Request(
                url,
                headers={"User-Agent": "CLIA-Documentation-Ingestion/1.0"},
            )
            with urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8-sig")

            if not body.strip() or "<html" in body[:500].lower():
                raise RuntimeError(f"Expected Markdown from {url}")

            timestamp = datetime.now(timezone.utc).isoformat()
            text = f"Source: {url}\nDownloaded: {timestamp}\n\n{body}"
            destination.write_text(text, encoding="utf-8")
            print(f"Downloaded: {destination.name}")

        manifest.append({
            "file": f"langchain/{destination.name}",
            "source_url": url,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })

    manifest_path = BASE_DIR / "documentation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print("Saved documentation_manifest.json")


if __name__ == "__main__":
    main()
