import logging
import sys
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from infmem_adapter import InfMemSummarizer
from searchers.base import BaseSearcher
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


def register_tools(
    mcp: FastMCP,
    searcher: BaseSearcher,
    snippet_max_tokens: int | None = None,
    k: int = 5,
    include_get_document: bool = True,
    long_doc_mode: str = "truncate",
    infmem_summarizer: InfMemSummarizer | None = None,
):

    tokenizer = None
    if snippet_max_tokens and snippet_max_tokens > 0:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        tokenizer.model_max_length = max(
            int(getattr(tokenizer, "model_max_length", 0) or 0), sys.maxsize
        )
        if infmem_summarizer is not None and infmem_summarizer.tokenizer is None:
            infmem_summarizer.tokenizer = tokenizer

    def build_truncated_snippet(text: str) -> tuple[str, int]:
        if not snippet_max_tokens or snippet_max_tokens <= 0 or not tokenizer:
            return text, -1

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= snippet_max_tokens:
            return text, len(token_ids)

        return (
            tokenizer.decode(token_ids[:snippet_max_tokens], skip_special_tokens=True),
            len(token_ids),
        )

    @mcp.tool(
        name="search",
        description=searcher.search_description(k),
    )
    def search(
        query: str,
        original_question: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the index and return top-k hits
        Args:
            query: Search query string
            original_question: Optional original user question for query-aware long-document summarization.
        Returns:
            List of search results with docid, score, text
        """
        candidates = searcher.search(query, k)

        for cand in candidates:
            text = cand["text"]
            snippet, doc_token_count = build_truncated_snippet(text)

            should_use_infmem = (
                long_doc_mode == "infmem"
                and infmem_summarizer is not None
                and snippet_max_tokens is not None
                and snippet_max_tokens > 0
                and doc_token_count > snippet_max_tokens
            )
            if should_use_infmem:
                question = (original_question or query).strip() or query
                try:
                    cand["snippet"] = infmem_summarizer.summarize_document(
                        docid=str(cand["docid"]),
                        document_text=text,
                        question=question,
                    )
                    logger.info(
                        "InfMem summary generated for docid=%s using question=%r",
                        cand["docid"],
                        question,
                    )
                    continue
                except Exception as exc:
                    logger.warning(
                        "InfMem summarization failed for docid=%s; falling back to truncation: %s",
                        cand["docid"],
                        exc,
                    )

            cand["snippet"] = snippet

        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({"docid": cand["docid"], "snippet": cand["snippet"]})
            else:
                results.append(
                    {
                        "docid": cand["docid"],
                        "score": cand["score"],
                        "snippet": cand["snippet"],
                    }
                )

        return results

    if include_get_document:

        @mcp.tool(
            name="get_document",
            description=searcher.get_document_description(),
        )
        def get_document(docid: str) -> Optional[Dict[str, Any]]:
            """
            Retrieve the full text of a document by its ID from the search index.

            Args:
                docid: Document ID to retrieve

            Returns:
                Document with full text, or None if not found
            """
            return searcher.get_document(docid)
