import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from openai import OpenAI
from qwen_agent.llm.fncall_prompts.qwen_fncall_prompt import QwenFnCallPrompt
from rank_bm25 import BM25Okapi


logger = logging.getLogger(__name__)

NO_MEMORY = "No previous memory"

TEMPLATE_CALL_OR_ANSWER = """
You are a Retrieval Planner.

Your ONLY task is to decide whether to perform another retrieval using `retrievesearch`, or STOP retrieving.

You MUST NOT answer the QUESTION.
Another model will use MEMORY later.

Guidelines:
- Retrieval is cheap. Unless MEMORY clearly contains all essential information, you are encouraged to retrieve.
- Retrieval may happen multiple times. At each step, refine the search direction.
- Avoid repeating previous queries in RETRIEVAL_HISTORY unless meaningfully refined.

When deciding whether to retrieve again:
1. Break the QUESTION into specific missing information needs.
2. Compare these needs with what MEMORY already contains.
3. Identify which facts remain missing, uncertain, or incomplete.
4. If something important is missing, design a NEW retrieval query focused only on that missing information.
5. If MEMORY already contains the necessary evidence, choose to STOP.

If you choose retrieval, you MUST output a function call to `retrievesearch` with:
- a new `query`
- and a `top_k` suited to your confidence

ORIGINAL QUESTION:
{prompt}

<retrieval_history>
{retrieval_history}
</retrieval_history>

CURRENT MEMORY:
{memory}
""".strip()

TEMPLATE_RETREIVE_RECURRENT = """
You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem.

The given section has two parts. One is retrieved chunk text, which is retrieved by the given question. Another is the recurrent chunk that is given recurrently. Both chunks might contain useful information; the retrieved chunk might have a higher chance.

<problem>
{prompt}
</problem>

<retrieved_chunk>
{retrieve}
</retrieved_chunk>

<recurrent_chunk>
{chunk}
</recurrent_chunk>

<memory>
{memory}
</memory>

Updated memory:
""".strip()

TEMPLATE_COMPRESS_MEMORY = """
You are given a question and a working memory extracted from a long document.

Rewrite the memory into a compact evidence summary that preserves the important facts needed to answer the question.

Rules:
- Keep only evidence relevant to the question.
- Preserve names, dates, entities, and relations when useful.
- Do not invent details.
- Do not provide a final answer unless the document explicitly states it.

<question>
{prompt}
</question>

<memory>
{memory}
</memory>

Compact evidence summary:
""".strip()


FN_NAME = "✿FUNCTION✿"
FN_ARGS = "✿ARGS✿"
FN_RESULT = "✿RESULT✿"
FN_EXIT = "✿RETURN✿"

BM25_MAX_PER_RECURRENT = 5
BM25_DEDUP_JACCARD = 0.8
BM25_CAND_MULTIPLIER = 6


def _extract_solution(solution_str: str) -> str:
    if "</think>" not in solution_str:
        return solution_str.strip()
    return solution_str.split("</think>")[-1].strip()


def _tokenize_mixed(text: str) -> List[str]:
    lower_text = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", lower_text)
    zh_chars = [ch for ch in lower_text if "\u4e00" <= ch <= "\u9fff"]
    if zh_chars:
        tokens.extend(zh_chars)
        tokens.extend(
            zh_chars[i] + zh_chars[i + 1] for i in range(len(zh_chars) - 1)
        )
    return tokens


def _normalize_query(query: str, fallback: str) -> str:
    if not isinstance(query, str):
        return fallback
    query = re.sub(r"[\r\n\t]+", " ", query).strip()
    query = re.sub(r"\s+", " ", query)
    return query or fallback


def _clamp_top_k(top_k: int, default: int = 8, min_k: int = 3, max_k: int = 20) -> int:
    try:
        top_k = int(top_k)
    except Exception:
        top_k = default
    return max(min_k, min(max_k, top_k))


def _jaccard(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    union_size = len(set_a | set_b)
    if union_size == 0:
        return 0.0
    return len(set_a & set_b) / union_size


def _build_retrieve_corpus_enhanced(
    input_ids: Sequence[int],
    tokenizer,
    recurrent_chunks: Sequence[dict],
    retrieve_chunk_size: int,
    retrieve_chunk_overlap: int,
) -> tuple[List[str], List[dict]]:
    overlap = min(retrieve_chunk_overlap, retrieve_chunk_size - 1)
    step = max(1, retrieve_chunk_size - overlap)

    retrieve_docs = []
    retrieve_meta = []
    for rc in recurrent_chunks:
        r_idx = rc["r_idx"]
        start, end = rc["start"], rc["end"]
        pos = start
        while pos < end:
            rs = pos
            re_idx = min(end, rs + retrieve_chunk_size)
            sub_ids = input_ids[rs:re_idx]
            retrieve_docs.append(tokenizer.decode(sub_ids, skip_special_tokens=True))
            retrieve_meta.append(
                {
                    "recurrent_idx": r_idx,
                    "start": rs,
                    "end": re_idx,
                }
            )
            if re_idx >= end:
                break
            pos = min(end, rs + step)
    return retrieve_docs, retrieve_meta


def _build_retrieve_corpus_from_texts(
    texts: Sequence[str],
    tokenizer,
    retrieve_chunk_size: int,
    retrieve_chunk_overlap: int,
) -> tuple[List[str], List[dict]]:
    recurrent_chunks = []
    input_ids_by_text = []
    for idx, text in enumerate(texts):
        token_ids = tokenizer.encode(text or "", add_special_tokens=False)
        if not token_ids:
            continue
        input_ids_by_text.append((idx, token_ids))
        recurrent_chunks.append(
            {
                "r_idx": idx,
                "start": 0,
                "end": len(token_ids),
            }
        )

    retrieve_docs = []
    retrieve_meta = []
    overlap = min(retrieve_chunk_overlap, retrieve_chunk_size - 1)
    step = max(1, retrieve_chunk_size - overlap)

    for chunk_info, (_, token_ids) in zip(recurrent_chunks, input_ids_by_text):
        pos = chunk_info["start"]
        end = chunk_info["end"]
        while pos < end:
            rs = pos
            re_idx = min(end, rs + retrieve_chunk_size)
            retrieve_docs.append(tokenizer.decode(token_ids[rs:re_idx], skip_special_tokens=True))
            retrieve_meta.append(
                {
                    "recurrent_idx": chunk_info["r_idx"],
                    "start": rs,
                    "end": re_idx,
                }
            )
            if re_idx >= end:
                break
            pos = min(end, rs + step)

    return retrieve_docs, retrieve_meta


def _bm25_retrieve_for_recurrent_chunk_enhanced(
    query: str,
    bm25_model: BM25Okapi,
    retrieve_docs: Sequence[str],
    retrieve_meta: Sequence[dict],
    retrieve_doc_tokens: Sequence[Sequence[str]],
    top_k: int,
    exclude_recurrent_idx: int | None = None,
    exclude_recurrent_indices: Sequence[int] | None = None,
) -> List[str]:
    q_tokens = _tokenize_mixed(query)
    if not q_tokens:
        return []

    excluded_indices = set(exclude_recurrent_indices or [])
    if exclude_recurrent_idx is not None:
        excluded_indices.add(exclude_recurrent_idx)

    scores = bm25_model.get_scores(q_tokens)
    if len(scores) == 0:
        return []

    candidate_count = min(len(scores), max(40, top_k * BM25_CAND_MULTIPLIER))
    if candidate_count <= 0:
        return []
    candidate_idx = scores.argsort()[::-1][:candidate_count]

    results = []
    selected_tokens = []
    per_recurrent = defaultdict(int)

    for idx in candidate_idx:
        if (
            retrieve_meta[idx]["recurrent_idx"] in excluded_indices
        ):
            continue

        recurrent_idx = retrieve_meta[idx]["recurrent_idx"]
        if per_recurrent[recurrent_idx] >= BM25_MAX_PER_RECURRENT:
            continue

        candidate_tokens = retrieve_doc_tokens[idx]
        if any(
            _jaccard(candidate_tokens, chosen_tokens) > BM25_DEDUP_JACCARD
            for chosen_tokens in selected_tokens
        ):
            continue

        results.append(retrieve_docs[idx])
        selected_tokens.append(candidate_tokens)
        per_recurrent[recurrent_idx] += 1
        if len(results) >= top_k:
            break

    if len(results) >= top_k:
        return results

    ranked = scores.argsort()[::-1]
    for idx in ranked:
        if (
            retrieve_meta[idx]["recurrent_idx"] in excluded_indices
        ):
            continue
        text = retrieve_docs[idx]
        if text in results:
            continue
        results.append(text)
        if len(results) >= top_k:
            break
    return results


def _get_bm25search_tool_schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "retrievesearch",
            "description": "Search over pre-indexed context chunks using BM25.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Concise search query focusing on key entities or concepts.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (3-20).",
                        "minimum": 3,
                        "maximum": 20,
                        "default": 8,
                    },
                },
                "required": ["query"],
            },
        },
    }


def _process_messages_to_text(tokenizer, messages, functions=None, enable_thinking=True):
    from qwen_agent.llm.schema import ContentItem, Message

    qwen_messages = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            content = [ContentItem(text=content)]
        elif isinstance(content, list):
            content_items = []
            for item in content:
                if isinstance(item, str):
                    content_items.append(ContentItem(text=item))
                elif isinstance(item, dict) and "text" in item:
                    content_items.append(ContentItem(text=item["text"]))
                else:
                    content_items.append(ContentItem(text=str(item)))
            content = content_items
        else:
            content = [ContentItem(text=str(content))]

        qwen_messages.append(Message(role=msg["role"], content=content))

    if functions:
        processed_messages = QwenFnCallPrompt.preprocess_fncall_messages(
            messages=qwen_messages,
            functions=functions,
            lang="en",
            parallel_function_calls=True,
            function_choice="auto",
        )
    else:
        processed_messages = qwen_messages

    dict_messages = []
    for msg in processed_messages:
        content_text = ""
        for content_item in msg.content:
            content_text += content_item.text
        dict_messages.append({"role": msg.role, "content": content_text})

    return tokenizer.apply_chat_template(
        dict_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _remove_trailing_comment_of_fn_args(fn_args: str) -> str:
    fn_args = fn_args.strip()

    if fn_args.startswith("{"):
        k = fn_args.rfind("}")
        if k > 0:
            fn_args = fn_args[: k + 1]

    if fn_args.startswith("```"):
        k = fn_args.rfind("\n```")
        if k > 0:
            fn_args = fn_args[: k + 4]

    return fn_args


def _remove_incomplete_special_tokens(text: str) -> str:
    special_tokens = (FN_NAME, FN_ARGS, FN_RESULT, FN_EXIT)
    text = text.rstrip()

    if text.endswith(special_tokens):
        for token in special_tokens:
            if text.endswith(token):
                text = text[: -len(token)]
                break
    else:
        trail_start = text.rfind("✿")
        if trail_start >= 0:
            trail_token = text[trail_start:]
            for token in special_tokens:
                if token.startswith(trail_token):
                    text = text[:trail_start]
                    break

    return text.lstrip("\n").rstrip()


def _parse_function_calls_from_text(text: str):
    function_calls = []
    if "STOP" in text:
        return function_calls, "", True

    pattern = (
        f"{re.escape(FN_NAME)}:\\s*([^\\n]+)\\s*"
        f"{re.escape(FN_ARGS)}:\\s*([^✿]+?)(?={re.escape(FN_RESULT)}|"
        f"{re.escape(FN_EXIT)}|{re.escape(FN_NAME)}|$)"
    )

    for match in re.finditer(pattern, text, re.DOTALL):
        func_name = match.group(1).strip()
        func_args = _remove_trailing_comment_of_fn_args(match.group(2).strip())
        function_calls.append({"name": func_name, "arguments": func_args})

    first_fn_pos = text.find(FN_NAME)
    remaining_text = text[:first_fn_pos].strip() if first_fn_pos >= 0 else text.strip()
    remaining_text = _remove_incomplete_special_tokens(remaining_text)
    return function_calls, remaining_text, False


def _parse_response(response: str):
    function_calls, remaining_text, stop = _parse_function_calls_from_text(response)
    messages = []

    if function_calls:
        if remaining_text.strip():
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": remaining_text.strip()
                    .strip("<think>")
                    .strip("</think>")
                    .strip(),
                }
            )

        for func_call in function_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": func_call["name"],
                        "arguments": func_call["arguments"],
                    },
                }
            )
    else:
        thinking_pattern = r"<think>(.*?)</think>\s*(.*)"
        match = re.search(thinking_pattern, response, re.DOTALL)
        if match:
            messages.append(
                {
                    "role": "assistant",
                    "content": match.group(2).strip(),
                    "reasoning_content": match.group(1).strip(),
                }
            )
        else:
            clean_response = response.strip()
            if clean_response.startswith("<think>") and "</think>" not in clean_response:
                clean_response = clean_response[7:].strip()
            messages.append(
                {
                    "role": "assistant",
                    "content": clean_response,
                    "reasoning_content": "",
                }
            )

    if not messages:
        messages.append({"role": "assistant", "content": ""})

    return messages, stop


@dataclass(frozen=True)
class InfMemConfig:
    model: str
    model_server: str
    timeout_seconds: float = 120.0
    max_summary_tokens: int = 1024
    recurrent_chunk_size: int = 2000
    recurrent_max_new: int = 1024
    recurrent_max_context_len: int = 120000
    max_recurrent_steps: int = 4
    retrieve_chunk_size: int = 200
    retrieve_chunk_overlap: int = 100
    default_top_k: int = 8
    early_stop: int = 2
    enable_thinking: bool = True
    thinking_budget: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95


class InfMemSummarizer:
    def __init__(self, config: InfMemConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.client = OpenAI(
            base_url=config.model_server,
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            timeout=config.timeout_seconds,
            max_retries=0,
        )
        self._cache: Dict[tuple[str, str, int, str], str] = {}

    def summarize_document(
        self,
        *,
        docid: str,
        document_text: str,
        question: str,
    ) -> str:
        if self.tokenizer is None:
            raise ValueError("InfMemSummarizer requires a tokenizer before use")

        cache_key = (
            docid,
            question,
            self.config.max_summary_tokens,
            self.config.model,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("InfMem cache hit for docid=%s", docid)
            return cached

        memory = self._run_infmem(question=question, document_text=document_text)
        if not memory.strip() or memory == NO_MEMORY:
            raise ValueError("InfMem summarization did not extract useful memory")

        summary = self._compress_memory_if_needed(question=question, memory=memory)
        summary = self._clip_to_token_budget(summary, self.config.max_summary_tokens)
        if not summary.strip():
            raise ValueError("InfMem summarization returned an empty summary")

        self._cache[cache_key] = summary
        return summary

    def update_memory_with_context_pool(
        self,
        *,
        question: str,
        previous_memory: str,
        current_chunk: str,
        context_pool_texts: Sequence[str],
        exclude_pool_indices: Sequence[int] | None = None,
    ) -> tuple[str, Dict[str, Any]]:
        if self.tokenizer is None:
            raise ValueError("InfMemSummarizer requires a tokenizer before use")

        memory = (previous_memory or "").strip() or NO_MEMORY
        current_chunk = (current_chunk or "").strip()
        pool_texts = [text for text in context_pool_texts if isinstance(text, str) and text.strip()]
        retrieval_history: List[Dict[str, Any]] = []
        retrieved_results: List[str] = []

        if pool_texts:
            retrieve_docs, retrieve_meta = _build_retrieve_corpus_from_texts(
                texts=pool_texts,
                tokenizer=self.tokenizer,
                retrieve_chunk_size=self.config.retrieve_chunk_size,
                retrieve_chunk_overlap=self.config.retrieve_chunk_overlap,
            )
            retrieve_doc_tokens = [_tokenize_mixed(doc) for doc in retrieve_docs]

            if retrieve_doc_tokens:
                bm25 = BM25Okapi(retrieve_doc_tokens)
                planner_prompt = TEMPLATE_CALL_OR_ANSWER.format(
                    prompt=question,
                    memory=memory,
                    retrieval_history="Single-step hidden-pool retrieval for the latest tool result.",
                )
                planner_text = _process_messages_to_text(
                    self.tokenizer,
                    [{"role": "user", "content": planner_prompt}],
                    functions=[_get_bm25search_tool_schema()["function"]],
                    enable_thinking=self.config.enable_thinking,
                )
                planner_raw = self._call_llm_text(
                    planner_text,
                    max_len=self.config.recurrent_max_new,
                )
                parsed_messages, _ = _parse_response(planner_raw)
                tool_calls = [
                    msg for msg in parsed_messages if isinstance(msg.get("function_call"), dict)
                ]

                for msg in tool_calls:
                    function_call = msg["function_call"]
                    if function_call["name"] != "retrievesearch":
                        continue
                    try:
                        args = json.loads(function_call["arguments"])
                    except Exception:
                        args = {}

                    query = _normalize_query(args.get("query", question), question)
                    top_k = _clamp_top_k(args.get("top_k", self.config.default_top_k))
                    retrieval_history.append({"query": query, "top_k": top_k})
                    retrieved_results = _bm25_retrieve_for_recurrent_chunk_enhanced(
                        query=query,
                        bm25_model=bm25,
                        retrieve_docs=retrieve_docs,
                        retrieve_meta=retrieve_meta,
                        retrieve_doc_tokens=retrieve_doc_tokens,
                        top_k=top_k,
                        exclude_recurrent_indices=exclude_pool_indices,
                    )
                    if retrieved_results:
                        break

        retrieved_block = "\n\n".join(
            f"[Retrieved #{idx + 1}] {text}"
            for idx, text in enumerate(retrieved_results)
        )
        memory_prompt = TEMPLATE_RETREIVE_RECURRENT.format(
            prompt=question,
            retrieve=retrieved_block,
            chunk=current_chunk,
            memory=memory,
        )
        memory_text = _process_messages_to_text(
            self.tokenizer,
            [{"role": "user", "content": memory_prompt}],
            functions=None,
            enable_thinking=self.config.enable_thinking,
        )
        memory_raw = self._call_llm_text(
            memory_text,
            max_len=self.config.recurrent_max_new,
        )
        updated_memory = _extract_solution(memory_raw) or memory
        updated_memory = self._compress_memory_if_needed(
            question=question,
            memory=updated_memory,
        )
        updated_memory = self._clip_to_token_budget(
            updated_memory,
            self.config.max_summary_tokens,
        )
        if not updated_memory.strip():
            updated_memory = memory

        return updated_memory, {
            "retrieval_history": retrieval_history,
            "retrieved_results_count": len(retrieved_results),
            "retrieved_block": retrieved_block,
            "context_pool_count": len(pool_texts),
            "excluded_pool_indices": list(exclude_pool_indices or []),
            "current_chunk_chars": len(current_chunk),
        }

    def _run_infmem(self, *, question: str, document_text: str) -> str:
        input_ids = self.tokenizer.encode(document_text, add_special_tokens=False)
        if len(input_ids) > self.config.recurrent_max_context_len:
            half = self.config.recurrent_max_context_len // 2
            input_ids = input_ids[:half] + input_ids[-half:]

        if not input_ids:
            return ""

        recurrent_chunks = []
        pos = 0
        while pos < len(input_ids):
            start = pos
            end = min(len(input_ids), pos + self.config.recurrent_chunk_size)
            recurrent_chunks.append(
                {
                    "r_idx": len(recurrent_chunks),
                    "start": start,
                    "end": end,
                }
            )
            pos = end

        if self.config.max_recurrent_steps > 0:
            recurrent_chunks = recurrent_chunks[: self.config.max_recurrent_steps]

        retrieve_docs, retrieve_meta = _build_retrieve_corpus_enhanced(
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            recurrent_chunks=recurrent_chunks,
            retrieve_chunk_size=self.config.retrieve_chunk_size,
            retrieve_chunk_overlap=self.config.retrieve_chunk_overlap,
        )
        retrieve_doc_tokens = [_tokenize_mixed(doc) for doc in retrieve_docs]
        bm25 = BM25Okapi(retrieve_doc_tokens)

        functions = [_get_bm25search_tool_schema()["function"]]
        memory = NO_MEMORY
        stop_time = 0
        history = []

        for retrieve_step, recurrent_chunk in enumerate(recurrent_chunks):
            start, end = recurrent_chunk["start"], recurrent_chunk["end"]
            chunk_text = self.tokenizer.decode(
                input_ids[start:end],
                skip_special_tokens=True,
            )

            retrieval_history_str = (
                f"Current retrieval step: {retrieve_step + 1}  Maximum allowed retrieval steps: {len(recurrent_chunks)}\n"
                + "\n".join(
                    f"Step {i + 1}: query={item['query']!r}, top_k={item['top_k']}"
                    for i, item in enumerate(history)
                )
            )

            planner_prompt = TEMPLATE_CALL_OR_ANSWER.format(
                prompt=question,
                memory=memory,
                retrieval_history=retrieval_history_str or "None yet.",
            )
            planner_text = _process_messages_to_text(
                self.tokenizer,
                [{"role": "user", "content": planner_prompt}],
                functions=functions,
                enable_thinking=self.config.enable_thinking,
            )
            planner_raw = self._call_llm_text(
                planner_text,
                max_len=self.config.recurrent_max_new,
            )
            parsed_messages, stop = _parse_response(planner_raw)

            tool_calls = [
                msg for msg in parsed_messages if isinstance(msg.get("function_call"), dict)
            ]

            if not stop:
                stop_time = 0

            if stop or not tool_calls:
                stop_time += 1
                if stop_time >= self.config.early_stop:
                    break

            retrieved_block = ""
            for msg in tool_calls:
                function_call = msg["function_call"]
                if function_call["name"] != "retrievesearch":
                    continue

                try:
                    args = json.loads(function_call["arguments"])
                except Exception:
                    args = {}

                query = _normalize_query(args.get("query", question), question)
                top_k = _clamp_top_k(args.get("top_k", self.config.default_top_k))
                history.append({"query": query, "top_k": top_k})

                results = _bm25_retrieve_for_recurrent_chunk_enhanced(
                    query=query,
                    bm25_model=bm25,
                    retrieve_docs=retrieve_docs,
                    retrieve_meta=retrieve_meta,
                    retrieve_doc_tokens=retrieve_doc_tokens,
                    top_k=top_k,
                    exclude_recurrent_idx=recurrent_chunk["r_idx"],
                )
                retrieved_block = "\n\n".join(
                    f"[Retrieved #{idx + 1}] {text}"
                    for idx, text in enumerate(results)
                )

            memory_prompt = TEMPLATE_RETREIVE_RECURRENT.format(
                prompt=question,
                retrieval_history=retrieval_history_str,
                retrieve=retrieved_block,
                chunk=chunk_text,
                memory=memory,
            )
            memory_text = _process_messages_to_text(
                self.tokenizer,
                [{"role": "user", "content": memory_prompt}],
                functions=None,
                enable_thinking=self.config.enable_thinking,
            )
            memory_raw = self._call_llm_text(
                memory_text,
                max_len=self.config.recurrent_max_new,
            )
            updated_memory = _extract_solution(memory_raw)
            if updated_memory:
                memory = updated_memory

        return memory

    def _call_llm_text(self, text: str, max_len: int, stop=None) -> str:
        try:
            if self.config.enable_thinking:
                thinking_response = self.client.completions.create(
                    model=self.config.model,
                    prompt=text,
                    max_tokens=self.config.thinking_budget,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    stop=stop,
                )
                think_part = (thinking_response.choices[0].text or "").rstrip()
                if "</think>" not in think_part:
                    think_part += (
                        "\n\nConsidering the limited time, I must stop thinking now.\n</think>\n\n"
                    )

                followup_prompt = text + think_part
                answer_response = self.client.completions.create(
                    model=self.config.model,
                    prompt=followup_prompt,
                    max_tokens=max(1, max_len),
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    stop=stop,
                )
                answer_part = (answer_response.choices[0].text or "").strip()
                return (think_part + answer_part).strip()

            response = self.client.completions.create(
                model=self.config.model,
                prompt=text,
                max_tokens=max_len,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stop=stop,
            )
            return (response.choices[0].text or "").strip()
        except Exception as exc:
            logger.warning("InfMem model call failed: %s", exc)
            return ""

    def _compress_memory_if_needed(self, *, question: str, memory: str) -> str:
        token_ids = self.tokenizer.encode(memory, add_special_tokens=False)
        if len(token_ids) <= self.config.max_summary_tokens:
            return memory

        prompt = TEMPLATE_COMPRESS_MEMORY.format(prompt=question, memory=memory)
        prompt_text = _process_messages_to_text(
            self.tokenizer,
            [{"role": "user", "content": prompt}],
            functions=None,
            enable_thinking=self.config.enable_thinking,
        )
        compressed = self._call_llm_text(
            prompt_text,
            max_len=self.config.max_summary_tokens,
        )
        compressed = _extract_solution(compressed)
        return compressed or memory

    def _clip_to_token_budget(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return text

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text

        return self.tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)
