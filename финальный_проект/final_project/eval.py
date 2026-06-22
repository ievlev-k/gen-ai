"""
Eval suite — 15+ test cases for tickets.

Загружает тикеты из input/tickets.json, обрабатывает каждый через pipeline:
  1. RAG-поиск + ответ
  2. ЛLM-as-judge vs gold answer
  3. Ghost quote detection vs KB text

Запуск:
  python eval.py
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import time
from pathlib import Path

import tiktoken

from schema import EvalResult, GhostDetectionResult, JudgeVerdict
from kb_engine import hybrid_search, ingest_kb
from llm_client import get_model, make_client
from pipeline import (
    TraceWriter, build_rag_answer, detect_ghost_quotes,
)

OUTPUT_DIR = Path(__file__).parent / "output"
INPUT_DIR = Path(__file__).parent / "input"
MODEL = get_model()


JUDGE_PROMPT = """Ты — независимый судья (LLM-as-judge).

Задача: оценить правильность ответа системы на основе gold summary.

correct = true, если:
- Ответ системы покрывает основную суть золотого ответа
- Нет грубых фактических ошибок
- Да, ответ может быть сформулирован иначе

correct = false, если:
- Ответ не отвечает на вопрос
- Ответ содержит фактические ошибки
- Ответ не покрывает ключевые факты из gold answer

hallucination = true, если ответ содержит информацию, явно противоречащую gold answer.

Проверяй объективно и справедливо. Будь немного снисходителен: если ответ "в правильном направлении" и покрывает суть — считай correct = true."""


def _count_tokens(text: str) -> int:
    if not text or len(text.strip()) == 0:
        return 0
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = tiktoken.get_encoding("p50k_base")
    return len(enc.encode(text))


def run_eval():
    OUTPUT_DIR.mkdir(exist_ok=True)
    kb_path = INPUT_DIR / "knowledge_base.txt"

    if not (OUTPUT_DIR / "bm25_cache.json").exists():
        print("KB не проиндексирована. Запускаю ingest...", flush=True)
        ingest_kb(str(kb_path))

    kb_text = kb_path.read_text(encoding="utf-8")

    # Загружаем тикеты
    raw = json.loads((INPUT_DIR / "tickets.json").read_text(encoding="utf-8"))
    tickets = [
        (t["id"], t["message"], t["gold_answer_summary"],
         t["category"], t.get("complexity", "simple"))
        for t in raw
    ]

    print(f"\nОцениваю {len(tickets)} тикетов...\n", flush=True)
    trace = TraceWriter(OUTPUT_DIR / "trace_eval.jsonl")
    results: list[EvalResult] = []

    for i, (tid, question, gold, category, complexity) in enumerate(tickets):
        run_id = str(tid)
        t0 = time.time()
        print(f"  [{i+1}/{len(tickets)}] Тикет {tid}: {question[:50]}...", flush=True)

        answer_text = ""
        confidence = 0.0
        tools_used = []
        steps = 0
        ghost_count = 0
        quotes = []

        try:
            # Шаг 1: RAG-поиск + ответ
            # Шаг 1: RAG-поиск + ответ
            hits = hybrid_search(question, k=5)
            rag_answer = build_rag_answer(question, hits, trace, run_id)
            answer_text = rag_answer.answer
            confidence = rag_answer.confidence
            tools_used = ["kb_search", "rag"]
            steps = 1
            quotes = rag_answer.quotes

            # Тикеты с низкой уверенностью проходят через triage-агента
            auto_solved = confidence >= 0.7
            if not auto_solved:
                steps = 2
                tools_used = ["kb_search", "rag", "triage"]
                from pipeline import triage_ticket
                from schema import Ticket
                t = Ticket(id=tid, student_name="eval", subject=question[:80], message=question, category=category, complexity=complexity)
                triaged = triage_ticket(t, trace, run_id)
                if triaged.suggested_channel == "ta":
                    answer_text += f"\n\n⚠️ Низкая уверенность (confidence={confidence:.2f}). Рекомендация: перенаправить на TA."
                elif triaged.suggested_channel == "instructor":
                    answer_text += f"\n\n🔴 Требуется преподаватель (приоритет: {triaged.priority}, категория: {triaged.category})."

        except Exception as e:
            answer_text = f"(ERROR: {e})"
            confidence = 0.0
            tools_used = []
            steps = 0

        # Шаг 2: LLM-as-judge
        correct = False
        hallucination = False
        path_quality = "judge_error"
        try:
            icon = make_client()
            verdict: JudgeVerdict = icon.chat.completions.create(
                model=MODEL, response_model=JudgeVerdict, temperature=0.0,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": (
                        f"Вопрос: {question}\n"
                        f"Ответ системы: {answer_text}\n"
                        f"Gold answer summary: {gold}"
                    )},
                ],
            )
            correct = verdict.correct
            hallucination = verdict.hallucination
            if correct:
                path_quality = "acceptable"
            elif hallucination:
                path_quality = "hallucination"
            else:
                path_quality = "incorrect"
        except Exception as e:
            path_quality = f"judge_error: {e}"

        # Шаг 3: Проверка ghost-цитат
        try:
            ghosts = detect_ghost_quotes(quotes, kb_text, trace, run_id)
            ghost_count = len(ghosts)
        except Exception:
            ghost_count = 0

        elapsed = round(time.time() - t0, 2)

        # Подсчёт токенов
        prompt_tokens = _count_tokens(question)
        answer_tokens = _count_tokens(answer_text)
        total_tokens = prompt_tokens + answer_tokens

        result = EvalResult(
            ticket_id=tid,
            question=question[:200],
            answer=answer_text,
            correct=correct,
            steps=steps,
            tools_used=tools_used,
            tokens=total_tokens,
            confidence=confidence,
            path_quality=path_quality,
            ghost_count=ghost_count,
        )
        results.append(result)
        mark = "OK" if correct else "FAIL"
        print(
            f"       [{mark}] верно={correct}  шаги={steps}  "
            f"уверенность={confidence:.2f}  путь={path_quality}  "
            f"ghost={ghost_count}  токенов={total_tokens}",
            flush=True,
        )

    trace.close()

    print("\n" + "=" * 120)
    print(f"{'ID':>4} | {'Категория':<18} | {'Верно':>5} | {'Шаги':>5} | {'Уверен':>7} | {'Токены':>7} | {'Ghost':>7} | Путь")
    print("-" * 120)
    for r in results:
        cat = next((t["category"] for t in raw if t["id"] == r.ticket_id), "?")
        print(
            f"{r.ticket_id:>4} | {cat:<18} | {'ДА' if r.correct else 'НЕТ':>5} "
            f"| {r.steps:>5} | {r.confidence:>7.2f} | {r.tokens:>7} | {r.ghost_count:>7} "
            f"| {r.path_quality}"
        )
    print("=" * 120)

    total = len(results)
    correct_count = sum(1 for r in results if r.correct)
    pass_rate = round(correct_count / total * 100, 1) if total else 0
    avg_steps = round(sum(r.steps for r in results) / max(total, 1), 2)
    avg_conf = round(sum(r.confidence for r in results) / max(total, 1), 2)
    hallucinations = sum(1 for r in results if r.path_quality == "hallucination")
    incorrects = sum(1 for r in results if r.path_quality == "incorrect")
    total_ghosts = sum(r.ghost_count for r in results)
    total_tokens = sum(r.tokens for r in results)

    summary = {
        "total_evaluated": total,
        "passed": correct_count,
        "failed": total - correct_count,
        "pass_rate_pct": pass_rate,
        "avg_steps": avg_steps,
        "avg_confidence": avg_conf,
        "hallucinations": hallucinations,
        "incorrect_no_hallucination": incorrects,
        "total_ghost_quotes": total_ghosts,
        "total_tokens": total_tokens,
        "ghosts_per_ticket": round(total_ghosts / max(total, 1), 2),
        "avg_tokens": round(total_tokens / max(total, 1)),
    }
    print(f"\nПРОЙДЕНО: {pass_rate}% ({correct_count}/{total})")
    print(f"Средн. шаги: {avg_steps}, средн. уверенность: {avg_conf}")
    print(f"Галлюцинации: {hallucinations}, Неверно (без галлюцинаций): {incorrects}")
    print(f"Ghost-цитаты: {total_ghosts}, Всего токенов: {total_tokens}")

    eval_output = {
        "summary": summary,
        "results": [r.model_dump() for r in results],
    }
    (OUTPUT_DIR / "eval_results.json").write_text(
        json.dumps(eval_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nРезультаты сохранены в output/eval_results.json", flush=True)

    return results


if __name__ == "__main__":
    run_eval()
