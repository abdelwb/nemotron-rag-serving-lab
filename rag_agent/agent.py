"""LangChain RAG + tool-calling agent, served by either vLLM or SGLang.

The same agent code runs unmodified against either engine -- just point
--base-url at whichever OpenAI-compatible server is up (see ../serving/).

    python agent.py --base-url http://localhost:8000/v1 --model nemotron-lora \\
        --index-dir ./faiss_index --query "What batch sizes were swept in the vLLM benchmark?"

Build the index first with ingest.py.
"""
import argparse

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from router_baseline_sklearn import classify

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = (
    "You are a concise assistant for the Nemotron RAG Serving Lab project. "
    "Use the search_project_docs tool when a question is about this "
    "project's own design, code, or benchmarks. Use the calculator tool for "
    "arithmetic. Otherwise answer directly and briefly."
)


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression, e.g. '12 * (3 + 4)'."""
    # Demo-scope safety: whitelist characters before eval. Not a substitute
    # for a real sandboxed expression parser in a production tool.
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains disallowed characters."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:  # noqa: BLE001 - surfaced back to the LLM, not raised
        return f"Error: {exc}"


def build_agent(base_url: str, model: str, index_dir: str) -> AgentExecutor:
    llm = ChatOpenAI(base_url=base_url, api_key="not-needed", model=model, temperature=0.0)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_project_docs",
        description="Search this project's own docs (README, architecture notes) for grounding.",
    )

    tools = [retriever_tool, calculator]
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def main(args):
    route = classify(args.query)
    print(f"[router] {args.query!r} -> {route}")

    if route == "out_of_scope":
        print("This assistant only answers questions about this project and general/direct queries.")
        return

    executor = build_agent(args.base_url, args.model, args.index_dir)
    result = executor.invoke({"input": args.query})
    print("\n--- answer ---")
    print(result["output"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM: :8000/v1  |  SGLang: :30000/v1")
    parser.add_argument("--model", default="nemotron-lora")
    parser.add_argument("--index-dir", default="./faiss_index")
    parser.add_argument("--query", required=True)
    main(parser.parse_args())
