import json
import logging
import sys
from typing import List, Union

from qwen_agent.tools.base import BaseTool, register_tool
from transformers import AutoTokenizer

from searcher.infmem_adapter import InfMemSummarizer


logger = logging.getLogger(__name__)


@register_tool("search", allow_overwrite=True)
class SearchToolHandler(BaseTool):
    name = "search"

    def __init__(
        self,
        searcher,
        snippet_max_tokens: int = 512,
        k: int = 5,
        long_doc_mode: str = "truncate",
        infmem_summarizer: InfMemSummarizer | None = None,
    ):
        super().__init__()

        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.long_doc_mode = long_doc_mode
        self.infmem_summarizer = infmem_summarizer

        self.description = (
            f"Search the local BrowseComp-Plus corpus using one 'query' string and "
            f"return the top {self.k} matching results. Each result is only a preview "
            "snippet, not the full document."
        )

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            self.tokenizer.model_max_length = max(
                int(getattr(self.tokenizer, "model_max_length", 0) or 0), sys.maxsize
            )
            if self.infmem_summarizer is not None and self.infmem_summarizer.tokenizer is None:
                self.infmem_summarizer.tokenizer = self.tokenizer

    def _truncate_with_count(self, text: str, max_tokens: int) -> tuple[str, int]:
        if max_tokens <= 0 or not self.tokenizer:
            return text, -1

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            return (
                self.tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True),
                len(tokens),
            )
        return text, len(tokens)

    def _format_results(self, results: List[dict], max_tokens: int):
        formatted = []
        for idx, r in enumerate(results, 1):
            passage_text = r["text"]
            title = ""
            docid = str(r.get("docid", ""))

            if passage_text.startswith("---\ntitle:"):
                lines = passage_text.split("\n")
                if len(lines) > 1:
                    title = lines[1].replace("title:", "").strip().strip("\"")

            if not title:
                first_line = passage_text.split('\n')[0].strip()
                title = first_line[:50] + "..." if len(first_line) > 50 else first_line

            snippet = r.get("snippet")
            if snippet is None:
                snippet, _ = self._truncate_with_count(passage_text, max_tokens)
            formatted_result = f"{idx}. [{title}] (docid: {docid})\n{snippet}"
            formatted.append(formatted_result)

        return formatted

    def search_with_searcher(
        self,
        query: str,
        k: int = None,
        original_question: str | None = None,
    ):
        try:
            if k is None:
                k = self.k

            results = self.searcher.search(query, k)

            if not results:
                return f"No results found for '{query}'. Try with a more general query.", []

            docids = []
            for r in results:
                if "docid" in r:
                    docids.append(str(r["docid"]))

            for result in results:
                passage_text = result["text"]
                snippet, doc_token_count = self._truncate_with_count(
                    passage_text, self.snippet_max_tokens
                )
                result["doc_token_count"] = doc_token_count
                result["snippet_truncated"] = (
                    doc_token_count > self.snippet_max_tokens
                    if (
                        self.snippet_max_tokens is not None
                        and self.snippet_max_tokens > 0
                        and doc_token_count >= 0
                    )
                    else False
                )

                should_use_infmem = (
                    self.long_doc_mode == "infmem"
                    and self.infmem_summarizer is not None
                    and self.snippet_max_tokens is not None
                    and self.snippet_max_tokens > 0
                    and doc_token_count > self.snippet_max_tokens
                )

                if should_use_infmem:
                    question = (original_question or query).strip() or query
                    try:
                        result["snippet"] = self.infmem_summarizer.summarize_document(
                            docid=str(result["docid"]),
                            document_text=passage_text,
                            question=question,
                        )
                        logger.info(
                            "InfMem summary generated for docid=%s using question=%r",
                            result["docid"],
                            question,
                        )
                        continue
                    except Exception as exc:
                        logger.warning(
                            "InfMem summarization failed for docid=%s; falling back to truncation: %s",
                            result["docid"],
                            exc,
                        )

                result["snippet"] = snippet

            formatted_results = self._format_results(results, self.snippet_max_tokens)

            content = (
                f"A search for '{query}' found {len(formatted_results)} results. "
                "Each result includes a docid. If a result looks promising but the "
                "snippet may be incomplete, visiting one or two of the exact docids "
                "shown below is often useful before answering.\n\n## Corpus Results\n"
                + "\n\n".join(formatted_results)
            )
            return content, docids

        except Exception as e:
            return f"Search error for query '{query}': {str(e)}", []

    def call(self, params: Union[str, dict], **kwargs):
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field", None

        if not isinstance(query, str):
            if isinstance(query, list) and len(query) == 1 and isinstance(query[0], str):
                query = query[0]
            else:
                return "[Search] Invalid request format: 'query' must be a string, not an array", None

        response, docids = self.search_with_searcher(
            query,
            original_question=kwargs.get("original_question"),
        )
        return response, docids


@register_tool("visit", allow_overwrite=True)
class VisitToolHandler(BaseTool):
    name = "visit"

    def __init__(self, searcher):
        super().__init__()

        self.searcher = searcher
        self.description = (
            "Retrieve the full contents of one promising document by supplying its "
            "exact 'docid' from a previous search result. This local visit tool only "
            "accepts docid, not URL. Use this after search when a preview snippet "
            "looks relevant, incomplete, or you need exact wording; checking one or "
            "two promising results is often worthwhile. Do not pass a URL, title, "
            "rank, or snippet text."
        )

    def call(self, params: Union[str, dict], **kwargs):
        try:
            docid = params["docid"]
        except Exception:
            return (
                "[Visit] Invalid request format: Input must be a JSON object "
                "containing 'docid' field",
                None,
            )

        if not isinstance(docid, str):
            return "[Visit] Invalid request format: 'docid' must be a string", None

        docid = docid.strip()
        if not docid:
            return "[Visit] Invalid request format: 'docid' must be non-empty", None

        try:
            result = self.searcher.get_document(docid)
        except Exception as exc:
            return f"[Visit] Error retrieving docid '{docid}': {exc}", None

        if result is None:
            return f"[Visit] Document with docid '{docid}' not found.", None

        return json.dumps(result, ensure_ascii=False, indent=2), [docid]
