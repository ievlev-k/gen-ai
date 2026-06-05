

from __future__ import annotations

import csv
import json
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from llm_client import get_model, make_client
from prompts import (
    ASPECTS_SYSTEM,
    CHUNK_SYSTEM,
    DISCOVER_SYSTEM,
    JUDGE_SYSTEM,
    IE_SYSTEM,
    REDUCE_SYSTEM,
)
from schema import (
    CacheStats,
    ChunkSummary,
    DiscoveredAspects,
    DynamicReviewSentiment,
    JudgeReport,
    Review,
    ReviewSentiment,
    ReviewsSummary,
)

BASE_DIR = Path(__file__).parent
CSV_PATH = f'{BASE_DIR}/input/prakharrathi25/google-play-store-reviews/versions/1/reviews.csv'
SAMPLE_JSON = BASE_DIR / "input" / "reviews_sample.json"
OUTPUT_DIR = f'{BASE_DIR}/output'

client = make_client()
MODEL = get_model()

_lock = threading.Lock()
_token_totals = {"prompt": 0, "completion": 0}


def track_usage(completion) -> None:
    u = completion.usage
    with _lock:
        _token_totals["prompt"] += u.prompt_tokens
        _token_totals["completion"] += u.completion_tokens


def call_llm(messages, response_model, max_retries=3, temperature=0.0):
    result, comp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_model=response_model,
        max_retries=max_retries,
        temperature=temperature,
        with_completion=True,
    )
    track_usage(comp)
    return result


def load_and_sample_reviews(n: int = 50) -> list[dict]:
    if SAMPLE_JSON.exists():
        return json.loads(SAMPLE_JSON.read_text(encoding="utf-8"))

    all_rows: dict[str, list[dict]] = {}
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = row.get("score", "0")
            if score not in ("1", "2", "3", "4", "5"):
                continue
            if not row.get("content", "").strip():
                continue
            all_rows.setdefault(score, []).append({
                "author": row.get("userName", "anonymous"),
                "rating": int(score),
                "date": row.get("at", ""),
                "text": row["content"].strip(),
                "thumbs_up": row.get("thumbsUpCount", "0"),
            })

    per_bucket = max(n // 5, 1)
    sampled: list[dict] = []
    for score in sorted(all_rows.keys()):
        bucket = all_rows[score]
        sampled.extend(random.sample(bucket, min(per_bucket, len(bucket))))

    random.shuffle(sampled)
    SAMPLE_JSON.write_text(
        json.dumps(sampled, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Sampled {len(sampled)} reviews (balanced across ratings)")
    return sampled


def load_reviews() -> list[dict]:
    return load_and_sample_reviews(50)



def extract_reviews(reviews_json: str) -> list[Review]:
    return call_llm(
        [{"role": "system", "content": IE_SYSTEM}, {"role": "user", "content": reviews_json}],
        list[Review],
    )


def extract_aspects(reviews_json: str) -> list[ReviewSentiment]:
    return call_llm(
        [{"role": "system", "content": ASPECTS_SYSTEM}, {"role": "user", "content": reviews_json}],
        list[ReviewSentiment],
    )


def check_quotes(items, source_text: str) -> list[tuple[str, str]]:
    t = source_text.lower()
    ghosts: list[tuple[str, str]] = []
    for p in items:
        for a in p.aspects:
            probe = a.quote.strip().lower()[:30]
            if probe and probe not in t:
                ghosts.append((p.author, a.quote))
    return ghosts


def build_heatmap_fixed(sentiments: list[ReviewSentiment], out_path: str) -> None:
    ALL_ASPECTS = ["performance", "design", "support", "price", "ads", "reliability"]
    authors = [p.author for p in sentiments]
    sent_to_num = {"positive": 1, "negative": -1, "neutral": 0}
    matrix = np.full((len(authors), len(ALL_ASPECTS)), np.nan)
    for i, p in enumerate(sentiments):
        for a in p.aspects:
            if a.aspect in ALL_ASPECTS:
                j = ALL_ASPECTS.index(a.aspect)
                matrix[i, j] = sent_to_num[a.sentiment]

    fig, ax = plt.subplots(figsize=(max(10, len(authors) * 0.4), max(6, len(authors) * 0.3)))
    sns.heatmap(
        matrix,
        annot=len(authors) <= 25,
        fmt=".0f",
        xticklabels=ALL_ASPECTS,
        yticklabels=authors,
        center=0,
        cbar_kws={"label": "sentiment"},
        ax=ax,
    )
    plt.title("Aspect sentiment by reviewer (fixed)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def discover_aspects(reviews_json: str) -> DiscoveredAspects:
    return call_llm(
        [{"role": "system", "content": DISCOVER_SYSTEM}, {"role": "user", "content": reviews_json}],
        DiscoveredAspects,
    )


def extract_dynamic_aspects(
    reviews_json: str,
    discovered: DiscoveredAspects,
) -> list[DynamicReviewSentiment]:
    dynamic_block = "\n".join(
        f"- {a.name}: {a.description}" for a in discovered.aspects
    )
    sys_prompt = (
        "Ты — UX-аналитик. Перед тобой отзывы на мобильное приложение (текст на английском).\n\n"
        "Используй СТРОГО следующие обнаруженные аспекты:\n"
        + dynamic_block + "\n\n"
        "Возвращай ТОЛЬКО те аспекты, которые действительно упомянуты в отзыве.\n"
        "Для каждой оценки приведи ТОЧНУЮ дословную цитату из отзыва (на английском, оригинал)\n"
        "и свою уверенность (0.0–1.0).\n\n"
        "Ответ на русском. Цитаты — на английском, дословно."
    )
    return call_llm(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": reviews_json}],
        list[DynamicReviewSentiment],
    )


def build_heatmap_dynamic(sentiments: list[DynamicReviewSentiment], out_path: str) -> None:
    authors = [p.author for p in sentiments]
    all_dyn_aspects = sorted({a.aspect for p in sentiments for a in p.aspects})
    sent_to_num = {"positive": 1, "negative": -1, "neutral": 0}
    matrix = np.full((len(authors), len(all_dyn_aspects)), np.nan)
    for i, p in enumerate(sentiments):
        for a in p.aspects:
            if a.aspect in all_dyn_aspects:
                j = all_dyn_aspects.index(a.aspect)
                matrix[i, j] = sent_to_num[a.sentiment]

    fig, ax = plt.subplots(figsize=(max(10, len(all_dyn_aspects) * 1.2), max(6, len(authors) * 0.3)))
    sns.heatmap(
        matrix,
        annot=len(authors) <= 25,
        fmt=".0f",
        xticklabels=all_dyn_aspects,
        yticklabels=authors,
        center=0,
        cbar_kws={"label": "sentiment"},
        ax=ax,
    )
    plt.title("Aspect sentiment by reviewer (discovered)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def summarize_chunk(chunk_text: str) -> ChunkSummary:
    return call_llm(
        [{"role": "system", "content": CHUNK_SYSTEM}, {"role": "user", "content": chunk_text}],
        ChunkSummary,
    )


def reduce_summaries(summaries: list[ChunkSummary]) -> ReviewsSummary:
    joined = "\n\n".join(
        f"## {s.reviewer} ({s.sentiment})\n"
        + "\n".join(f"- {p}" for p in s.key_points)
        for s in summaries
    )
    return call_llm(
        [{"role": "system", "content": REDUCE_SYSTEM}, {"role": "user", "content": joined}],
        ReviewsSummary,
    )


def summarize_all(reviews: list[dict], workers: int = 6) -> ReviewsSummary:
    chunks = [
        f"Author: {r['author']} | Rating: {r['rating']}\n{r['text']}"
        for r in reviews
    ]
    n = len(chunks)
    t0 = time.time()
    print(f"  [MR] MAP: {n} reviews, up to {workers} parallel...")

    summaries: list[ChunkSummary | None] = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(summarize_chunk, c): i for i, c in enumerate(chunks)}
        done = 0
        for fut in as_completed(futures):
            summaries[futures[fut]] = fut.result()
            done += 1
            if done % 10 == 0 or done == n:
                print(f"  [MR] {done}/{n} done ({time.time() - t0:.1f}s)")

    print(f"  [MR] MAP {time.time() - t0:.1f}s -> REDUCE...")
    result = reduce_summaries([s for s in summaries if s is not None])
    print(f"  [MR] total {time.time() - t0:.1f}s")
    return result


def build_evidence_packet(reviews_data: list[dict], summary: dict) -> str:
    parts = ["## Recommendations under review"]
    for i, a in enumerate(summary.get("action_items", []), 1):
        parts.append(f"  {i}. {a}")
    parts.append("\n## Review issues (source data)")
    for r in reviews_data:
        author = r.get("author", "?")
        rating = r.get("rating", "?")
        issues_lines = [
            f"[{iss['category']}, sev={iss['severity']}] \"{iss['quote']}\""
            for iss in r.get("issues", [])
        ]
        pos_lines = [f"+ {p}" for p in r.get("positives", [])]
        detail = ""
        if issues_lines:
            detail += "\n    Issues:\n    " + "\n    ".join(issues_lines)
        if pos_lines:
            detail += "\n    Positives:\n    " + "\n    ".join(pos_lines)
        parts.append(f"  - {author}, rating={rating}{detail}")
    return "\n".join(parts)


def run_judge(reviews_data: list[dict], summary_dict: dict) -> JudgeReport:
    evidence = build_evidence_packet(reviews_data, summary_dict)
    return call_llm(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": evidence}],
        JudgeReport,
    )


def run_cache_benchmark(reviews_json: str) -> list[CacheStats]:

    stats: list[CacheStats] = []

    def run_once(label: str, system_prompt: str) -> CacheStats:
        t0 = time.time()
        _, completion = client.chat.completions.create(
            model=MODEL,
            response_model=list[ReviewSentiment],
            max_retries=2,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": reviews_json},
            ],
            with_completion=True,
        )
        track_usage(completion)
        usage = completion.usage
        cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        cache_miss = getattr(usage, "prompt_cache_miss_tokens", usage.prompt_tokens - cache_hit) or 0
        total_pt = usage.prompt_tokens
        return CacheStats(
            run_label=label,
            prompt_tokens=total_pt,
            cache_hit_tokens=cache_hit,
            cache_miss_tokens=cache_miss,
            hit_pct=round(cache_hit / total_pt * 100, 1) if total_pt else 0.0,
            latency_sec=round(time.time() - t0, 2),
        )

    print("  [CACHE] Run 1: cold start, prompt A")
    stats.append(run_once("cold_prompt_a", ASPECTS_SYSTEM))

    print("  [CACHE] Run 2: repeat, same prompt A")
    stats.append(run_once("repeat_prompt_a", ASPECTS_SYSTEM))

    modified = ASPECTS_SYSTEM + f"\n# random comment: {uuid.uuid4()}\n"
    print("  [CACHE] Run 3: modified prompt (prefix changed)")
    stats.append(run_once("modified_prompt", modified))

    print("  [CACHE] Run 4: restored prompt A")
    stats.append(run_once("restored_prompt_a", ASPECTS_SYSTEM))

    return stats


def analyze() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    raw_reviews = load_reviews()
    reviews_json = json.dumps(raw_reviews, ensure_ascii=False, indent=2)
    print(f"Loaded {len(raw_reviews)} reviews\n")

    print("Information Extraction...")
    reviews_parsed = extract_reviews(reviews_json)
    reviews_out = [r.model_dump() for r in reviews_parsed]
    Path(f'{OUTPUT_DIR}/reviews.json').write_text(
        json.dumps(reviews_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total_issues = sum(len(r.issues) for r in reviews_parsed)
    print(f"  Extracted {len(reviews_parsed)} reviews, {total_issues} issues\n")

    # Act 2: Fixed Aspects
    print("Fixed Aspect Analysis...")
    sentiments = extract_aspects(reviews_json)
    ghosts_fixed = check_quotes(sentiments, reviews_json)
    n_fixed = sum(len(p.aspects) for p in sentiments)
    print(f"  {len(sentiments)} reviews, {n_fixed} aspect ratings")
    if ghosts_fixed:
        print(f"  WARNING: {len(ghosts_fixed)} ghost quotes ({len(ghosts_fixed)/n_fixed*100:.1f}%)")
        for name, q in ghosts_fixed[:5]:
            print(f"    {name}: \"{q[:80]}...\"")
    else:
        print("  No ghost quotes detected")

    build_heatmap_fixed(sentiments, f'{OUTPUT_DIR}/heatmap.png')
    Path(f'{OUTPUT_DIR}/aspects.json').write_text(
        json.dumps([p.model_dump() for p in sentiments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("  Saved: heatmap.png, aspects.json\n")

    # Act 2.5: Autodiscovery
    print("Aspect Autodiscovery...")
    discovered = discover_aspects(reviews_json)
    print(f"  Discovered {len(discovered.aspects)} dynamic aspects:")
    for da in discovered.aspects:
        print(f"    • {da.name} — {da.description}")

    dyn_sentiments = extract_dynamic_aspects(reviews_json, discovered)
    ghosts_dyn = check_quotes(dyn_sentiments, reviews_json)
    n_dyn = sum(len(p.aspects) for p in dyn_sentiments)
    dyn_aspect_names = sorted({a.aspect for p in dyn_sentiments for a in p.aspects})
    print(f"\n  Dynamic: {n_dyn} ratings, aspects: {dyn_aspect_names}")
    if ghosts_dyn:
        print(f"  Ghost quotes: {len(ghosts_dyn)} ({len(ghosts_dyn)/n_dyn*100:.1f}%)")

    build_heatmap_dynamic(dyn_sentiments, f'{OUTPUT_DIR}/heatmap_discovered.png')
    Path(f'{OUTPUT_DIR}/aspects_discovered.json').write_text(
        json.dumps([p.model_dump() for p in dyn_sentiments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()

    # Act 3: Map-Reduce
    print("Map-Reduce Summary...")
    summary = summarize_all(raw_reviews)
    summary_dict = summary.model_dump()
    Path(f'{OUTPUT_DIR}/summary.json').write_text(
        json.dumps(summary_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  Headline: {summary.headline}")
    print(f"  Findings: {len(summary.key_findings)}, Actions: {len(summary.action_items)}\n")

    # Act 4: Judge
    print("LLM-as-Judge...")
    report = run_judge(reviews_out, summary_dict)
    Path(f'{OUTPUT_DIR}/judge_report.json').write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    counts = {"supported": 0, "weakly_supported": 0, "not_supported": 0}
    for v in report.verdicts:
        counts[v.support] += 1
    print(f"  Supported: {counts['supported']}, Weak: {counts['weakly_supported']}, Not: {counts['not_supported']}")
    print(f"  Score: {report.overall_score:.2f}\n")

    # Act 6: Caching
    print("Prompt Caching Benchmark...")
    cache_stats = run_cache_benchmark(reviews_json)
    Path(f'{OUTPUT_DIR}/cache_stats.json').write_text(
        json.dumps([s.model_dump() for s in cache_stats], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for cs in cache_stats:
        print(f"  {cs.run_label:<25} tokens={cs.prompt_tokens:>5}  "
              f"hits={cs.cache_hit_tokens:>5} ({cs.hit_pct:>4.0f}%)  t={cs.latency_sec:.1f}s")
    if len(cache_stats) >= 2:
        cold, cached = cache_stats[0], cache_stats[1]
        if cold.prompt_tokens > 0:
            savings = (1 - cached.cache_miss_tokens / max(cold.cache_miss_tokens, 1)) * 100
            print(f"  Savings on repeat: {savings:.0f}% fewer billed input tokens")
    print()

    # Metrics
    metrics = {
        "total_reviews": len(raw_reviews),
        "parsed_reviews": len(reviews_parsed),
        "total_issues": total_issues,
        "ghost_quotes": len(ghosts_fixed),
        "ghost_pct": round(len(ghosts_fixed) / max(n_fixed, 1) * 100, 1),
        "judge_score": report.overall_score,
        "total_time_sec": round(time.time() - t_total, 1),
        "cache_hit_pct": cache_stats[1].hit_pct if len(cache_stats) >= 2 else 0.0,
        "tokens_prompt": _token_totals["prompt"],
        "tokens_completion": _token_totals["completion"],
        "tokens_total": _token_totals["prompt"] + _token_totals["completion"],
    }
    Path(f'{OUTPUT_DIR}/metrics.json').write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("RESULTS")
    print(f"  Headline: {summary.headline}")
    for kf in summary.key_findings:
        print(f"  - {kf}")
    for ai in summary.action_items:
        print(f"  -> {ai}")
    print(f"\n  Judge: {report.overall_score:.2f}  Ghosts: {len(ghosts_fixed)} ({metrics['ghost_pct']}%)  Time: {metrics['total_time_sec']}s")
    print(f"  Artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    analyze()
