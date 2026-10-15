from mcp.server.fastmcp import FastMCP

if __package__:
    from .retriever import search_docs
else:
    from retriever import search_docs


mcp = FastMCP("CLIA Documentation RAG")


@mcp.tool()
def search_library_docs(query: str, top_k: int = 5) -> dict:
    """Search locally indexed library documentation.

    Use this tool to find documentation passages relevant to a coding
    question. Returns source filenames, chunk identifiers, and passage text.
    The current collection contains synthetic sample documentation.
    """
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20.")

    passages = search_docs(query, top_k)

    return {
        "query": query,
        "result_count": len(passages),
        "results": passages,
    }


if __name__ == "__main__":
    # Standard input/output carries MCP messages; avoid print() here.
    mcp.run(transport="stdio")