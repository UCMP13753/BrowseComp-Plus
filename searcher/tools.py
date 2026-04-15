import logging
import sys
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from infmem_adapter import InfMemSummarizer
from searchers.base import BaseSearcher
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


def _normalize_queries(query: str | List[str]) -> List[str]:
    if isinstance(query, str):
        cleaned = query.strip()
        return [cleaned] if cleaned else []

    if not isinstance(query, list):
        return []

    normalized = []
    for item in query:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if cleaned:
            normalized.append(cleaned)
    return normalized


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
        query: str | List[str],
        original_question: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the index and return top-k hits
        Args:
            query: Search query string, or a compatibility fallback list of strings
            original_question: Optional original user question for query-aware long-document summarization.
        Returns:
            List of search results with docid, score, text
        """
        normalized_queries = _normalize_queries(query)
        if not normalized_queries:
            raise ValueError("query must be a non-empty string or list of strings")

        if len(normalized_queries) > 1:
            logger.info(
                "Received %d queries in a single search call; expanding them sequentially",
                len(normalized_queries),
            )

        merged_candidates: List[Dict[str, Any]] = []
        merged_by_docid: Dict[str, Dict[str, Any]] = {}
        for single_query in normalized_queries:
            for candidate in searcher.search(single_query, k):
                docid = str(candidate["docid"])
                existing = merged_by_docid.get(docid)
                if existing is None:
                    copied = dict(candidate)
                    merged_by_docid[docid] = copied
                    merged_candidates.append(copied)
                    continue

                previous_score = existing.get("score")
                new_score = candidate.get("score")
                if new_score is not None and (
                    previous_score is None or new_score > previous_score
                ):
                    existing.update(candidate)

        if merged_candidates and all(
            candidate.get("score") is not None for candidate in merged_candidates
        ):
            merged_candidates.sort(key=lambda candidate: candidate["score"], reverse=True)

        candidates = merged_candidates[:k]
        summary_query = normalized_queries[0]

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
                question = (original_question or summary_query).strip() or summary_query
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
