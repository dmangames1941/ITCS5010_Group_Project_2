"""Compare vector and hybrid retrieval on a fixed question set."""

import csv
import json
from pathlib import Path
from time import perf_counter

if __package__:
    from .retriever import search_docs
else:
    from retriever import search_docs


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "evaluation"

# Expected pages are coarse relevance labels.
# Manually review the returned passages before interpreting these as quality.
CASES = [
    {
        "question": "How do I create a LangChain agent?",
        "sources": ["langchain/quickstart.md"],
    },
    {
        "question": "Show create_agent with a model and tools.",
        "sources": ["langchain/quickstart.md"],
    },
    {
        "question": "How do I define a tool using the @tool decorator?",
        "sources": ["langchain/tools.md"],
    },
    {
        "question": "How do I customize a tool name and description?",
        "sources": ["langchain/tools.md"],
    },
    {
        "question": "How can I define a tool input schema with Pydantic?",
        "sources": ["langchain/tools.md"],
    },
    {
        "question": "How do I initialize a chat model with init_chat_model?",
        "sources": ["langchain/models.md"],
    },
    {
        "question": "How do I stream a chat model response?",
        "sources": ["langchain/models.md"],
    },
    {
        "question": "How do I bind tools to a model using bind_tools?",
        "sources": ["langchain/models.md"],
    },
    {
        "question": "What are SystemMessage, HumanMessage, and AIMessage?",
        "sources": ["langchain/messages.md"],
    },
    {
        "question": "How do I return a tool result using ToolMessage?",
        "sources": ["langchain/messages.md"],
    },
    {
        "question": "How do I add short-term memory using a checkpointer?",
        "sources": ["langchain/short-term-memory.md"],
    },
    {
        "question": "How does thread_id identify a conversation with memory?",
        "sources": ["langchain/short-term-memory.md"],
    },
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    details = []

    # Warm up the embedding model before measuring retrieval calls.
    search_docs("LangChain", top_k=1, mode="vector")

    for case_index, case in enumerate(CASES, start=1):
        for mode in ("vector", "hybrid"):
            start = perf_counter()
            passages = search_docs(
                case["question"], top_k=5, mode=mode
            )
            elapsed_ms = (perf_counter() - start) * 1000

            first_rank = next(
                (
                    rank
                    for rank, passage in enumerate(passages, start=1)
                    if passage["source"] in case["sources"]
                ),
                None,
            )

            row = {
                "case": case_index,
                "mode": mode,
                "question": case["question"],
                "expected_sources": ";".join(case["sources"]),
                "source_hit_at_5": int(first_rank is not None),
                "source_reciprocal_rank_at_5": (
                    1 / first_rank if first_rank else 0
                ),
                "latency_ms": round(elapsed_ms, 2),
            }
            rows.append(row)
            details.append({**row, "passages": passages})
            print(
                f"Case {case_index:02d} | {mode:6s} | "
                f"source hit: {row['source_hit_at_5']} | "
                f"{elapsed_ms:.0f} ms"
            )

    with (OUTPUT_DIR / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (OUTPUT_DIR / "retrieved_passages.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = [
        "# Retrieval evaluation",
        "",
        "12 questions; top_k=5; same stored documents and chunks.",
        "",
        "| Mode | Source Hit@5 | Source MRR@5 | Mean latency (ms) |",
        "|---|---:|---:|---:|",
    ]

    for mode in ("vector", "hybrid"):
        selected = [row for row in rows if row["mode"] == mode]
        count = len(selected)
        hit_rate = sum(row["source_hit_at_5"] for row in selected) / count
        mrr = sum(
            row["source_reciprocal_rank_at_5"] for row in selected
        ) / count
        latency = sum(row["latency_ms"] for row in selected) / count

        summary.append(
            f"| {mode} | {hit_rate:.1%} | {mrr:.3f} | {latency:.1f} |"
        )

    summary.extend([
        "",
        "## Interpretation and limitations",
        "",
        "- These metrics measure retrieval of an expected source page.",
        "- A source-page match does not prove the passage answers the question.",
        "- Review retrieved_passages.json and record passage relevance separately.",
        "- Other pages may also contain valid answers.",
        "- Labels and questions were authored together; this is a small exploratory evaluation.",
        "- Latency includes client setup and, for hybrid, rebuilding the BM25 index.",
        "- One timed run per question is insufficient for a robust speed benchmark.",
        "- This evaluates retrieval, not generated answers or coding-task success.",
    ])

    summary_text = "\n".join(summary) + "\n"
    (OUTPUT_DIR / "summary.md").write_text(
        summary_text, encoding="utf-8"
    )

    print("\n" + summary_text)
    print(f"Saved evaluation files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()