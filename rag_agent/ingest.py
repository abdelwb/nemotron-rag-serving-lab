"""Build a local FAISS index over a directory of documents.

    python ingest.py --docs-dir ../docs --index-dir ./faiss_index

Deliberately does not ship a pre-built corpus in this repo -- point it at
your own docs (or this repo's own docs/ folder, to start). Embeddings come
from a small local sentence-transformers model via
`langchain_huggingface.HuggingFaceEmbeddings`, so ingestion needs no API key
and no running LLM server.
"""
import argparse
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(docs_dir: Path):
    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
    )
    return loader.load()


def main(args):
    docs_dir = Path(args.docs_dir)
    documents = load_documents(docs_dir)
    print(f"Loaded {len(documents)} documents from {docs_dir}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    index = FAISS.from_documents(chunks, embeddings)

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    index.save_local(str(index_dir))
    print(f"Saved FAISS index to {index_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", default="../docs", help="Directory of .md files to index")
    parser.add_argument("--index-dir", default="./faiss_index", help="Where to save the FAISS index")
    main(parser.parse_args())
