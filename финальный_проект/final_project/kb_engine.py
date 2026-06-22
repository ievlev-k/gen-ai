"""
RAG-движок для базы знаний курса.

Гибридный поиск: BM25 (sparse) + dense embeddings через RRF fusion.
Хранение: ChromaDB (persistent, ./output/chroma_db).
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import re
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

OUTPUT_DIR = Path(__file__).parent / "output"
DB_PATH = OUTPUT_DIR / "chroma_db"
BM25_CACHE = OUTPUT_DIR / "bm25_cache.json"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яa-z0-9ё-]{2,}", text.lower())


splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", "? ", "! ", " "],
)


def create_collection():
    OUTPUT_DIR.mkdir(exist_ok=True)
    db = chromadb.PersistentClient(path=str(DB_PATH))
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL,
    )
    return db.get_or_create_collection(
        name="course_kb",
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_kb(path: str) -> int:

    print("Загружаю эмбеддер...", flush=True)
    t0 = time.time()
    collection = create_collection()
    print(f"Эмбеддер загружен за {time.time() - t0:.1f}с", flush=True)

    text = Path(path).read_text(encoding="utf-8")
    chunks = [c.strip() for c in splitter.split_text(text) if c.strip()]
    ids = [f"kb__{i}" for i in range(len(chunks))]

    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.add(documents=chunks, ids=ids)

    bm25_data = {
        "ids": ids,
        "tokens": [_tokenize(c) for c in chunks],
        "texts": chunks,
    }
    BM25_CACHE.write_text(json.dumps(bm25_data, ensure_ascii=False), encoding="utf-8")

    print(f"Проиндексировано: {len(chunks)} чанков (Dense + BM25)", flush=True)
    return len(chunks)


def hybrid_search(query: str, k: int = 5, top: int = 15, c_rrf: int = 60) -> list[dict]:

    collection = create_collection()

    dense_res = collection.query(query_texts=[query], n_results=top)
    dense_ids = dense_res["ids"][0]
    dense_docs = dense_res["documents"][0]

    bm25_data = json.loads(BM25_CACHE.read_text(encoding="utf-8"))
    bm25 = BM25Okapi(bm25_data["tokens"])
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    bm25_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top]
    sparse_ids = [bm25_data["ids"][i] for i in bm25_order]

    rrf = {}
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c_rrf + rank)
    for rank, cid in enumerate(sparse_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c_rrf + rank)

    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]

    text_map = dict(zip(bm25_data["ids"], bm25_data["texts"]))
    for did, doc in zip(dense_ids, dense_docs):
        text_map[did] = doc

    return [
        {"id": cid, "text": text_map.get(cid, ""), "score": round(sc, 6)}
        for cid, sc in ordered
    ]
