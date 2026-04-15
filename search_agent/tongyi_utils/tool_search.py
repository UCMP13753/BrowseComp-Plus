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

        self.description = f"Performs a search on a knowledge source: supply a single 'query' string; the tool retrieves the top {self.k} most relevant results."

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

            # Follows original tongyi search tool format
            formatted_result = f"{idx}. [{title}]\n{snippet}"
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

            content = f"A search for '{query}' found {len(formatted_results)} results:\n\n## Web Results\n" + "\n\n".join(formatted_results)
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
