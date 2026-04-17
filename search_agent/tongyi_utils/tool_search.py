import json
import logging
import sys
from typing import Any, Dict, List, Union

from qwen_agent.tools.base import BaseTool, register_tool
from transformers import AutoTokenizer

from searcher.infmem_adapter import InfMemSummarizer


logger = logging.getLogger(__name__)


class LongDocInfMemHandler:
    def __init__(
        self,
        *,
        snippet_max_tokens: int = 512,
        long_doc_mode: str = "truncate",
        infmem_position: str = "search",
        infmem_summarizer: InfMemSummarizer | None = None,
    ):
        self.snippet_max_tokens = snippet_max_tokens
        self.long_doc_mode = long_doc_mode
        self.infmem_position = infmem_position
        self.infmem_summarizer = infmem_summarizer

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

    def _should_use_infmem(self, *, target: str, doc_token_count: int) -> bool:
        return (
            self.long_doc_mode == "infmem"
            and self.infmem_position == target
            and self.infmem_summarizer is not None
            and self.snippet_max_tokens is not None
            and self.snippet_max_tokens > 0
            and doc_token_count > self.snippet_max_tokens
        )

    def build_search_snippet(
        self,
        *,
        docid: str,
        document_text: str,
        question: str,
    ) -> Dict[str, Any]:
        snippet, doc_token_count = self._truncate_with_count(
            document_text,
            self.snippet_max_tokens,
        )
        snippet_truncated = (
            doc_token_count > self.snippet_max_tokens
            if (
                self.snippet_max_tokens is not None
                and self.snippet_max_tokens > 0
                and doc_token_count >= 0
            )
            else False
        )

        if self._should_use_infmem(target="search", doc_token_count=doc_token_count):
            try:
                summary = self.infmem_summarizer.summarize_document(
                    docid=docid,
                    document_text=document_text,
                    question=question,
                )
                logger.info(
                    "InfMem summary generated for search docid=%s using question=%r",
                    docid,
                    question,
                )
                return {
                    "snippet": summary,
                    "doc_token_count": doc_token_count,
                    "snippet_truncated": snippet_truncated,
                    "used_infmem_summary": True,
                }
            except Exception as exc:
                logger.warning(
                    "InfMem summarization failed for search docid=%s; falling back to truncation: %s",
                    docid,
                    exc,
                )

        return {
            "snippet": snippet,
            "doc_token_count": doc_token_count,
            "snippet_truncated": snippet_truncated,
            "used_infmem_summary": False,
        }

    def build_visit_document(
        self,
        *,
        docid: str,
        document: Dict[str, Any],
        question: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        processed = dict(document)
        processed.setdefault("docid", docid)

        document_text = str(processed.get("text", ""))
        if not document_text:
            return processed, {
                "doc_token_count": -1,
                "used_infmem_summary": False,
                "text_is_summary": False,
                "returned_text": document_text,
            }

        _, doc_token_count = self._truncate_with_count(
            document_text,
            self.snippet_max_tokens,
        )

        if self._should_use_infmem(target="visit", doc_token_count=doc_token_count):
            try:
                summary = self.infmem_summarizer.summarize_document(
                    docid=docid,
                    document_text=document_text,
                    question=question,
                )
                processed["text"] = summary
                processed["text_is_summary"] = True
                processed["summary_method"] = "infmem"
                processed["original_token_count"] = doc_token_count
                logger.info(
                    "InfMem summary generated for visit docid=%s using question=%r",
                    docid,
                    question,
                )
                return processed, {
                    "doc_token_count": doc_token_count,
                    "used_infmem_summary": True,
                    "text_is_summary": True,
                    "returned_text": summary,
                    "original_token_count": doc_token_count,
                }
            except Exception as exc:
                logger.warning(
                    "InfMem summarization failed for visit docid=%s; returning full document: %s",
                    docid,
                    exc,
                )

        return processed, {
            "doc_token_count": doc_token_count,
            "used_infmem_summary": False,
            "text_is_summary": False,
            "returned_text": document_text,
        }


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
        infmem_position: str = "search",
    ):
        super().__init__()

        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.long_doc_mode = long_doc_mode
        self.infmem_position = infmem_position
        self.long_doc_handler = LongDocInfMemHandler(
            snippet_max_tokens=snippet_max_tokens,
            long_doc_mode=long_doc_mode,
            infmem_position=infmem_position,
            infmem_summarizer=infmem_summarizer,
        )

        self.description = (
            f"Search the local BrowseComp-Plus corpus using one 'query' string and "
            f"return the top {self.k} matching results. Each result is only a preview "
            "snippet, not the full document."
        )

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
                snippet, _ = self.long_doc_handler._truncate_with_count(passage_text, max_tokens)
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
                return f"No results found for '{query}'. Try with a more general query.", [], {
                    "query": query,
                    "original_question": original_question,
                    "results": [],
                }

            docids = []
            for r in results:
                if "docid" in r:
                    docids.append(str(r["docid"]))

            result_metadata = []
            for result in results:
                passage_text = result["text"]
                question = (original_question or query).strip() or query
                long_doc_result = self.long_doc_handler.build_search_snippet(
                    docid=str(result["docid"]),
                    document_text=passage_text,
                    question=question,
                )
                result.update(long_doc_result)
                result_metadata.append(
                    {
                        "docid": str(result["docid"]),
                        "score": result.get("score"),
                        "snippet": result["snippet"],
                        "raw_text": passage_text,
                        "snippet_truncated": result["snippet_truncated"],
                        "doc_token_count": result["doc_token_count"],
                        "used_infmem_summary": result["used_infmem_summary"],
                    }
                )

            formatted_results = self._format_results(results, self.snippet_max_tokens)

            content = (
                f"A search for '{query}' found {len(formatted_results)} results. "
                "Each result includes a docid. If a result looks promising but the "
                "snippet may be incomplete, visiting one or two of the exact docids "
                "shown below is often useful before answering.\n\n## Corpus Results\n"
                + "\n\n".join(formatted_results)
            )
            return content, docids, {
                "query": query,
                "original_question": original_question,
                "results": result_metadata,
            }

        except Exception as e:
            return f"Search error for query '{query}': {str(e)}", [], {
                "query": query,
                "error": str(e),
            }

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

        response, docids, tool_data = self.search_with_searcher(
            query,
            original_question=kwargs.get("original_question"),
        )
        return response, docids, tool_data


@register_tool("visit", allow_overwrite=True)
class VisitToolHandler(BaseTool):
    name = "visit"

    def __init__(
        self,
        searcher,
        snippet_max_tokens: int = 512,
        long_doc_mode: str = "truncate",
        infmem_summarizer: InfMemSummarizer | None = None,
        infmem_position: str = "search",
    ):
        super().__init__()

        self.searcher = searcher
        self.long_doc_handler = LongDocInfMemHandler(
            snippet_max_tokens=snippet_max_tokens,
            long_doc_mode=long_doc_mode,
            infmem_position=infmem_position,
            infmem_summarizer=infmem_summarizer,
        )
        self.description = (
            "Retrieve the full contents of one promising document by supplying its "
            "exact 'docid' from a previous search result. This local visit tool only "
            "accepts docid, not URL. Use this after search when a preview snippet "
            "looks relevant, incomplete, or you need exact wording; checking one or "
            "two promising results is often worthwhile. Do not pass a URL, title, "
            "rank, or snippet text."
        )
        if long_doc_mode == "infmem" and infmem_position == "visit":
            self.description += (
                " When a visited document is very long, this tool may return an "
                "InfMem summary instead of the full text."
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

        question = (kwargs.get("original_question") or docid).strip() or docid
        processed_result, visit_meta = self.long_doc_handler.build_visit_document(
            docid=docid,
            document=result,
            question=question,
        )
        tool_data = {
            "docid": docid,
            "document": processed_result,
            "text": visit_meta.get("returned_text", ""),
            "raw_text": str(result.get("text", "")),
            "is_summary": visit_meta.get("text_is_summary", False),
            "used_infmem_summary": visit_meta.get("used_infmem_summary", False),
            "original_token_count": visit_meta.get("original_token_count", visit_meta.get("doc_token_count", -1)),
        }
        return json.dumps(processed_result, ensure_ascii=False, indent=2), [docid], tool_data
