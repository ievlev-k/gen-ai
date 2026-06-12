"""
RAG-пайплайн для корпуса Telegram-чата ITMO Programming Languages.

Поддерживает две стратегии чанкинга:
  --strategy naive    — фиксированные куски по 2000 символов
  --strategy smart    — RecursiveCharacterTextSplitter(512, overlap=80)

Команды:
  python pipeline.py ingest [--strategy naive|smart]
  python pipeline.py ask "Вопрос?" [--strategy naive|smart]
"""

import json
import re
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from schema import RAGAnswer
from llm_client import get_model, make_client

client = make_client()
MODEL = get_model()
_chromadb_client = chromadb.PersistentClient(path="./chroma_db")


print("Загружаю эмбеддер...", flush=True)
_t_embed = time.time()
EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
)
print(f"Embedder ready in {time.time() - _t_embed:.1f}s", flush=True)
collection = _chromadb_client.get_or_create_collection(
    name="itmo_chat",
    embedding_function=EMBED_FN,
    metadata={"hnsw:space": "cosine"},
)

DATA_DIR = Path(__file__).parent / "data" / "corpus"
BM25_CACHE_NAIVE = Path(__file__).parent / "bm25_cache_naive.json"
BM25_CACHE_SMART = Path(__file__).parent / "bm25_cache_smart.json"

splitter_smart = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=80,
    separators=["\n\n", "\n", ". ", "? ", "! ", " "],
)


def tokenize_ru(text: str):
    return re.findall(r"[а-яa-z0-9ё-]{2,}", text.lower())


def chunk_text_naive(text: str, chunk_size: int = 2000) -> list[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_text_smart(text: str) -> list[str]:
    return [c.strip() for c in splitter_smart.split_text(text) if c.strip()]


def ingest(strategy: str = "smart"):
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    chunk_fn = chunk_text_naive if strategy == "naive" else chunk_text_smart
    bm25_cache = BM25_CACHE_NAIVE if strategy == "naive" else BM25_CACHE_SMART

    all_chunks = []
    all_ids = []
    all_meta = []

    for f in sorted(DATA_DIR.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        chunks = chunk_fn(text)

        for i, c in enumerate(chunks):
            cid = f"{f.stem}__{i}"
            all_chunks.append(c)
            all_ids.append(cid)
            all_meta.append({"source": f.stem, "chunk_id": i})

        print(f"  {f.stem}: {len(chunks)} чанков", flush=True)

    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_meta)

    bm25_data = {
        "ids": all_ids,
        "tokens": [tokenize_ru(c) for c in all_chunks],
        "texts": all_chunks,
    }
    bm25_cache.write_text(json.dumps(bm25_data, ensure_ascii=False), encoding="utf-8")

    total = collection.count()
    print(
        f"\Индексировано: [{strategy}]: Dense — {total} чанков из "
        f"{len(list(DATA_DIR.glob('*.txt')))} files", flush=True,
    )
    print(f"BM25 — {len(all_ids)} чанков кэшировано в {bm25_cache.name}", flush=True)

def _load_bm25(strategy: str = "smart"):
    cache = BM25_CACHE_NAIVE if strategy == "naive" else BM25_CACHE_SMART
    data = json.loads(cache.read_text(encoding="utf-8"))
    bm25 = BM25Okapi(data["tokens"])
    return bm25, data["ids"], data["texts"]


def hybrid_retrieve(query: str, k: int = 5, top: int = 15, c: int = 60, strategy: str = "smart") -> dict:
    dense = collection.query(query_texts=[query], n_results=top)
    dense_ids = dense["ids"][0]

    bm25, bm25_ids, bm25_texts = _load_bm25(strategy)
    tokens = tokenize_ru(query)
    scores = bm25.get_scores(tokens)

    bm25_order = sorted(range(len(bm25_ids)), key=lambda i: scores[i], reverse=True)[:top]
    sparse_ids = [bm25_ids[i] for i in bm25_order]

    rrf = {}
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank)
    for rank, cid in enumerate(sparse_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank)

    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    top_ids = [cid for cid, _ in ordered]

    text_by_id = dict(zip(bm25_ids, bm25_texts))
    for i, did in enumerate(dense["ids"][0]):
        text_by_id[did] = dense["documents"][0][i]

    return {"ids": [top_ids], "documents": [[text_by_id.get(i, "") for i in top_ids]]}


def build_prompt(query: str, hits: dict) -> str:
    docs = hits["documents"][0]
    ids = hits["ids"][0]
    ctx = "\n\n---\n\n".join(f"[{i}]\n{d}" for i, d in zip(ids, docs))
    return (
        "Ты помогаешь отвечать на вопросы на основе архива Telegram-чата курса "
        "\"Языки программирования\" в ИТМО. Используй ТОЛЬКО контекст ниже. "
        "Если в контексте нет ответа, так и скажи об этом напрямую.\n\n"
        "Правила:\n"
        "1. Используй ТОЛЬКО контекст. Не добавляй факты из общих знаний.\n"
        "2. В `quotes` — от 1 до 5 точных коротких цитат из контекста.\n"
        "3. В `sources` — ID фрагментов (формат: 'doc_XX___0').\n"
        "4. В `confidence` — 0.9+ для прямого ответа, 0.5-0.8 для составного, "
        "<0.5, если контекст не содержит ответа.\n\n"
        f"Контекст:\n{ctx}\n\n"
        f"Вопрос: {query}\n\n"
        "Ответ:"
    )





def ask(query: str, strategy: str = "smart"):
    print("Поиск по базе...", flush=True)
    t0 = time.time()
    hits = hybrid_retrieve(query, k=15, strategy=strategy)
    found = hits["ids"][0]
    print(f"   found {len(found)} chunks in {time.time() - t0:.1f}s", flush=True)

    print("Генерация ответа...", flush=True)
    t1 = time.time()
    prompt = build_prompt(query, hits)
    resp: RAGAnswer = client.chat.completions.create(
        model=MODEL,
        response_model=RAGAnswer,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    print(f"   ответ за {time.time() - t1:.1f}s", flush=True)

    print("\n" + "=" * 60)
    print(f"ВОПРОС: {query}")
    print("=" * 60)
    print(resp)
    print("\n--- источники ---")
    for i in found:
        print(f"  {i}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["ingest", "ask"])
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--strategy", choices=["naive", "smart"], default="smart")
    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest(args.strategy)
    elif args.cmd == "ask":
        if not args.query:
            print('Need a query: python pipeline.py ask "..."')
            sys.exit(1)
        ask(args.query, args.strategy)
