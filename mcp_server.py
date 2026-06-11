"""
ZGA Web-Search MCP Server
=========================
A Model Context Protocol (MCP) server that exposes ONE external tool to the
agent: real-time public web search. The ADK Researcher agent connects to this
server over stdio (via MCPToolset) and calls `web_search` to verify uncertain
medical terms on the live internet BEFORE falling back to the curated Vertex
RAG knowledge base.

This is the Track 1 "external tool via MCP" integration: the agent securely
connects to an out-of-process tool server using the open MCP standard.

Run standalone (for debugging): python mcp_server.py
(ADK launches it automatically as a subprocess in production.)
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("zga-web-search")


@mcp.tool()
def web_search(query: str, max_results: int = 4) -> str:
    """Search the public web for a medical term, idiom, or rare disease.

    Use this to verify what an uncertain term means using up-to-the-minute
    public information before consulting the internal knowledge base.

    Args:
        query: The term or question to look up (e.g. "mal de San Vito disease meaning").
        max_results: How many web results to summarize (default 4).

    Returns:
        A concise text digest of the top web results (title + snippet + url),
        or "NO_RESULTS" if nothing useful was found.
    """
    try:
        from ddgs import DDGS
        # A slow search backend must never eat the interpreter's research budget.
        # The REAL bound is upstream: the MCP toolset (timeout=20) and the ADK
        # orchestrator deadline (run_clinical_reasoning timeout_s=12) cancel this
        # whole chain, after which main.py degrades to the Vertex knowledge base.
        results = DDGS().text(query, max_results=max(1, min(max_results, 6)))
    except Exception as e:
        return f"SEARCH_ERROR: {e}"

    if not results:
        return "NO_RESULTS"

    lines = []
    for r in results:
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip()
        href = (r.get("href") or "").strip()
        if title or body:
            lines.append(f"- {title}: {body[:240]} ({href})")
    return "\n".join(lines) if lines else "NO_RESULTS"


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
