# ITCS5010_Group_Project_2
# CLIA — Command Line Interface Assistant

CLIA is a team project to build an autonomous command-line AI coding
assistant with model-provider abstraction and dynamically loaded MCP tools.

## Project Plan

### Week 1 — Individual Work

- **Dalton Corriher:** Main control loop, terminal REPL interface,
  provider abstractions (Ollama and Grok), and execution-mode toggles.
- **Soumith Vallapalli:** MCP client architecture, tool schemas,
  dynamic loading, filesystem server, and external resource server.
- **Azar Salaam:** Custom RAG MCP server, persistent ChromaDB storage,
  document ingestion, and advanced hybrid retrieval.

### Week 2 — Group Work

First half:
- Integrate components and perform system testing.
- Compare models from the selected cloud provider and Ollama.

Second half:
- Draft and revise the report.
- Record the demonstration video.
- Complete final testing and improve documentation and code comments.

## Custom RAG MCP Server

The RAG component searches local snapshots of LangChain Python
documentation. It includes:

- Markdown-aware document chunking.
- Local embeddings using ChromaDB's default embedding function.
- Persistent ChromaDB storage.
- Vector-only baseline retrieval.
- Hybrid BM25 and vector retrieval with reciprocal rank fusion.
- The MCP tool `search_library_docs(query, top_k=5)`.

The standalone MCP test has successfully discovered and called the tool.
Integration with the main coding assistant remains to be verified.

### Setup

Run these commands from the repository root in Windows PowerShell:

```powershell
python -m venv clia/.venv
.\clia\.venv\Scripts\python.exe -m pip install chromadb "mcp>=1.0,<2" rank-bm25
```

These dependencies support the RAG component. Other CLIA components have
additional dependencies listed in `clia/requirements.txt`.

The first embedding operation downloads the embedding model.
The RAG server does not require a cloud API key.

### Download and Ingest Documentation

```powershell
.\clia\.venv\Scripts\python.exe -X utf8 .\clia\servers\custom_rag\download_docs.py
.\clia\.venv\Scripts\python.exe -X utf8 .\clia\servers\custom_rag\ingest.py
```

The downloader preserves existing snapshots. The documentation manifest
records source URLs and content hashes rather than an exact package version.

Run ingestion initially and after changing documentation or chunking.
The generated database is excluded from Git and persists between sessions.

### Test Retrieval

```powershell
.\clia\.venv\Scripts\python.exe -X utf8 .\clia\servers\custom_rag\retriever.py "How do I define a tool?" --mode hybrid --top-k 3
```

Use `--mode vector` to run the baseline.

### Test the MCP Server

```powershell
.\clia\.venv\Scripts\python.exe -X utf8 .\clia\servers\custom_rag\test_client.py
```

The test client launches the server, initializes the connection, discovers
available tools, and calls `search_library_docs`.

The server uses stdio transport. Its direct launch command is:

```powershell
.\clia\.venv\Scripts\python.exe -X utf8 .\clia\servers\custom_rag\server.py
```

When launched directly, it waits for MCP messages.

## Advanced RAG Implementation and Evaluation

Hybrid retrieval combines semantic vector search with BM25 keyword search.
Reciprocal rank fusion merges the two ranked candidate lists using a rank
constant of 60.

Both methods were evaluated on the same stored documents and chunks using
12 LangChain documentation questions and five results per question.

| Method | Source Hit@5 | Source MRR@5 | Mean latency |
|---|---:|---:|---:|
| Vector | 100.0% | 0.662 | 375.8 ms |
| Hybrid | 100.0% | 0.764 | 405.1 ms |

Both methods retrieved the expected source page within the top five results
for all questions. Hybrid retrieval improved source ranking, while adding
approximately 29.3 ms of mean measured latency.

These exploratory results do not establish passage accuracy or coding-task
success. Expected-source labels are coarse, other pages may contain valid
answers, and latency was measured once per question. Hybrid timing also
includes rebuilding the BM25 index.

Run the evaluation with:

```powershell
.\clia\.venv\Scripts\python.exe -X utf8 .\clia\servers\custom_rag\evaluate.py
```

Evidence is saved in `clia/servers/custom_rag/evaluation/`:

- `results.csv`
- `retrieved_passages.json`
- `summary.md`

### Known Limitations

- Retrieved passages require manual relevance review.
- Large code blocks may exceed the embedding model's input limit.
- Documentation markup can reduce retrieval quality.
- BM25 is rebuilt per request; caching is a future improvement.
- Ingestion does not automatically remove records for deleted source files.
- Retrieved documentation is reference material, not executable instructions.

## References

- LangChain documentation: https://docs.langchain.com/
- Advanced RAG techniques: https://github.com/NirDiamant/RAG_Techniques