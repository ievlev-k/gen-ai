"""
Eval по gold-вопросам для двух стратегий чанкинга.
Метрика: hit-rate@K на уровне документа-источника.

Для каждой стратегии автоматически делает ingest, затем eval.

Команды:
  python eval.py                       # обе стратегии, k=5
  python eval.py --k 5 --output res.json  # с сохранением в JSON
"""

import argparse
import json
from pathlib import Path

from pipeline import collection, hybrid_retrieve, ingest

GOLD_PATH = Path(__file__).parent / "data" / "gold.json"


def load_gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def hit_rate(retrieved_ids: list[str], gold_sources: list[str]) -> float:
    retrieved_sources = {rid.split("__")[0] for rid in retrieved_ids}
    found = [g for g in gold_sources if g in retrieved_sources]
    return len(found) / len(gold_sources)


def run(strategy: str, k: int = 5, verbose: bool = True) -> dict:
    gold = load_gold()
    total = 0.0
    results = []

    label = "NAIVE (fixed 2000)" if strategy == "naive" else "SMART (recursive 512)"
    if verbose:
        print(f"\n{'='*60}")
        print(f"  STRATEGY: {label}")
        print(f"{'='*60}")
        ingest(strategy)
        print()

    for item in gold:
        q = item["question"]
        gold_sources = item["gold_sources"]

        hits = hybrid_retrieve(q, k=k, strategy=strategy)
        retrieved_ids = hits["ids"][0]
        retrieved_sources = [rid.split("__")[0] for rid in retrieved_ids]

        score = hit_rate(retrieved_ids, gold_sources)
        total += score

        results.append({
            "id": item["id"],
            "type": item["type"],
            "question": q,
            "score": round(score, 2),
            "gold": gold_sources,
            "retrieved_sources": retrieved_sources,
        })

        if verbose:
            mark = "+" if score == 1.0 else ("~" if score > 0 else "x")
            print(
                f"  [{item['id']:2d}] {item['type']:20s}  "
                f"hit@{k} = {score:.2f}  {mark}  {q[:65]}"
            )

    mean = total / len(gold)
    if verbose:
        print(f"\n  TOTAL [{strategy}]: hit-rate@{k} = {mean:.2f}  ({total:.1f} / {len(gold)})")
    return {"strategy": strategy, "mean": round(mean, 4), "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    all_results = []
    for strat in ["naive", "smart"]:
        res = run(strat, k=args.k, verbose=not args.quiet)
        all_results.append(res)

    if args.output:
        Path(args.output).write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nResults saved to {args.output}")

    if not args.quiet and len(all_results) == 2:
        na = all_results[0]["mean"]
        sm = all_results[1]["mean"]
        winner = "SMART" if sm > na else ("NAIVE" if na > sm else "DRAW")
        print(f"\n>>> Winner: {winner} (smart={sm:.2f} vs naive={na:.2f})")


if __name__ == "__main__":
    main()
