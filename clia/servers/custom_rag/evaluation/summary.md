# Retrieval evaluation

12 questions; top_k=5; same stored documents and chunks.

| Mode | Source Hit@5 | Source MRR@5 | Mean latency (ms) |
|---|---:|---:|---:|
| vector | 100.0% | 0.662 | 375.8 |
| hybrid | 100.0% | 0.764 | 405.1 |

## Interpretation and limitations

- These metrics measure retrieval of an expected source page.
- A source-page match does not prove the passage answers the question.
- Review retrieved_passages.json and record passage relevance separately.
- Other pages may also contain valid answers.
- Labels and questions were authored together; this is a small exploratory evaluation.
- Latency includes client setup and, for hybrid, rebuilding the BM25 index.
- One timed run per question is insufficient for a robust speed benchmark.
- This evaluates retrieval, not generated answers or coding-task success.
