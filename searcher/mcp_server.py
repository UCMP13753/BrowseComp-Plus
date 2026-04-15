import os
import sys

from fastmcp import FastMCP
from pyngrok import ngrok
from pyngrok.exception import PyngrokNgrokError

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dotenv import load_dotenv
from infmem_adapter import InfMemConfig, InfMemSummarizer
from searchers import SearcherType
from tools import register_tools

load_dotenv(override=False)

script_env_path = os.path.join(current_dir, ".env")
load_dotenv(dotenv_path=script_env_path, override=False)

import argparse


def main():
    try:
        parser = argparse.ArgumentParser(description="MCP retrieval server")

        parser.add_argument(
            "--searcher-type",
            choices=SearcherType.get_choices(),
            required=True,
            help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
        )

        # MCP server and high-level arguments
        parser.add_argument(
            "--snippet-max-tokens",
            type=int,
            default=1024,
            help="Number of tokens to include for each document snippet in search results using Qwen/Qwen3-0.6B tokenizer (default: 1024). This is also the InfMem activation threshold and output budget. Set to -1 to disable.",
        )
        parser.add_argument(
            "--k",
            type=int,
            default=5,
            help="Fixed number of search results to return for all queries in this session (default: 5).",
        )
        parser.add_argument(
            "--long-doc-mode",
            choices=["truncate", "infmem"],
            default="truncate",
            help="How to handle documents longer than --snippet-max-tokens. 'truncate' returns the leading snippet, while 'infmem' returns a query-aware InfMem summary (default: truncate).",
        )
        parser.add_argument(
            "--infmem-model",
            type=str,
            help="Judge/read model to use for InfMem long-document summarization. Required when --long-doc-mode=infmem.",
        )
        parser.add_argument(
            "--infmem-model-server",
            type=str,
            default="http://127.0.0.1:8000/v1",
            help="OpenAI-compatible model server URL used for InfMem summarization (default: %(default)s).",
        )
        parser.add_argument(
            "--infmem-timeout-seconds",
            type=float,
            default=120.0,
            help="Timeout in seconds for each InfMem model request (default: %(default)s).",
        )
        parser.add_argument(
            "--infmem-max-recurrent-steps",
            type=int,
            default=4,
            help="Maximum number of recurrent 5k-token document chunks InfMem may process per document (default: %(default)s).",
        )
        parser.add_argument(
            "--get-document",
            action="store_true",
            help="If set, register both the search tool and the get_document tool.",
        )
        parser.add_argument(
            "--transport",
            choices=["stdio", "streamable-http", "sse"],
            default="sse",
            help="Transport protocol to run the MCP server with. Use 'http' for streamable HTTP, 'sse' for Server-Sent Events (more reliable with OpenAI), or 'stdio' for standard input/output.",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=8000,
            help="Port to bind the HTTP server when --transport=http (default: 8000). Ignored for stdio.",
        )
        parser.add_argument(
            "--public",
            action="store_true",
            help="If set with --transport=http, automatically create an ngrok tunnel and print the public MCP URL.",
        )
        parser.add_argument(
            "--hf-token",
            type=str,
            help="Hugging Face token for accessing private datasets/models. If not provided, will use environment variables or CLI login.",
        )
        parser.add_argument(
            "--hf-home",
            type=str,
            help="Hugging Face home directory for caching models and datasets. If not provided, will use environment variables or default.",
        )

        temp_args, _ = parser.parse_known_args()

        searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)

        searcher_class.parse_args(parser)

        args = parser.parse_args()

        if args.hf_token:
            print(
                f"[DEBUG] Setting HF token from CLI argument: {args.hf_token[:10]}..."
            )
            os.environ["HF_TOKEN"] = args.hf_token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token

        if args.hf_home:
            print(f"[DEBUG] Setting HF home from CLI argument: {args.hf_home}")
            os.environ["HF_HOME"] = args.hf_home

        searcher = searcher_class(args)

        snippet_max_tokens = args.snippet_max_tokens
        k = args.k
        long_doc_mode = args.long_doc_mode
        include_get_document = args.get_document
        transport = args.transport
        port = args.port
        public = args.public

        infmem_summarizer = None
        if long_doc_mode == "infmem":
            if snippet_max_tokens is None or snippet_max_tokens <= 0:
                raise ValueError(
                    "--snippet-max-tokens must be positive when --long-doc-mode=infmem"
                )
            if not args.infmem_model:
                raise ValueError(
                    "--infmem-model is required when --long-doc-mode=infmem"
                )

            infmem_summarizer = InfMemSummarizer(
                InfMemConfig(
                    model=args.infmem_model,
                    model_server=args.infmem_model_server,
                    timeout_seconds=args.infmem_timeout_seconds,
                    max_summary_tokens=snippet_max_tokens,
                    recurrent_max_new=snippet_max_tokens,
                    max_recurrent_steps=args.infmem_max_recurrent_steps,
                    early_stop=1,
                    thinking_budget=1024,
                    enable_thinking=True,
                ),
                tokenizer=None,
            )

        mcp = FastMCP(name="search-server")

        register_tools(
            mcp,
            searcher,
            snippet_max_tokens,
            k,
            include_get_document,
            long_doc_mode=long_doc_mode,
            infmem_summarizer=infmem_summarizer,
        )

        tools_registered = ["search"]
        if include_get_document:
            tools_registered.append("get_document")
        tools_str = ", ".join(tools_registered)

        print(
            f"MCP server started with {searcher.search_type} search (snippet_max_tokens={snippet_max_tokens}, k={k}, long_doc_mode={long_doc_mode}) using transport {transport.upper()}"
        )
        print(f"Registered tools: {tools_str}")
        if long_doc_mode == "infmem":
            print(
                f"InfMem configuration: model={args.infmem_model}, model_server={args.infmem_model_server}, timeout={args.infmem_timeout_seconds}s, thinking_budget=1024, memory_budget={snippet_max_tokens}, early_stop=1, max_recurrent_steps={args.infmem_max_recurrent_steps}"
            )

        if public:
            try:
                token = os.getenv("NGROK_AUTHTOKEN")
                if token:
                    ngrok.set_auth_token(token)

                try:
                    tunnel = ngrok.connect(addr=port, bind_tls=True)
                    public_url = f"{tunnel.public_url}/mcp"
                    print(
                        "\n=============================================\n"
                        f"Public MCP endpoint available at: {public_url}\n"
                        "Use this URL in the MCP clients via --mcp-url\n"
                        "=============================================\n"
                    )
                except PyngrokNgrokError as e:
                    first_line = str(e).split("\n")[0]
                    print(
                        f"[Warning] Failed to start ngrok tunnel: {first_line}\n"
                        f"Continuing with local server on http://localhost:{port}",
                    )
            except ImportError:
                print(
                    "[Warning] pyngrok not installed; continuing without public tunnel."
                )

        if transport == "stdio":
            mcp.run(transport="stdio")
        else:
            mcp.run(transport=transport, path="/mcp", port=port)

    except Exception as e:
        print("Error", e)
        raise


if __name__ == "__main__":
    main()
