"""
Основной пайплайн — Ticket Triage System.

Команды:
  python pipeline.py generate    — сгенерировать синтетические персоны и тикеты
  python pipeline.py ingest      — индексация KB
  python pipeline.py triage      — обработка всех тикетов
  python pipeline.py evaluate    — eval на gold-наборе
  python pipeline.py full        — весь пайплайн
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

from schema import CategorizedTicket, GhostDetectionResult, JudgeVerdict, RAGAnswer, Ticket
from kb_engine import hybrid_search, ingest_kb
from llm_client import get_model, make_client, make_raw_client
from synthetic import generate_all_tickets, generate_personas

OUTPUT_DIR = Path(__file__).parent / "output"
INPUT_DIR = Path(__file__).parent / "input"
MODEL = get_model()



class TraceWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(exist_ok=True)
        self.fh = open(self.path, "w", encoding="utf-8")

    def write(self, event: dict[str, Any]):
        self.fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()



RAG_SYSTEM = """Ты — помощник по стажировкам и поиску работы в IT.
Отвечай на вопросы студентов, используя ТОЛЬКО контекст ниже (база знаний из реального Telegram-чата sns_internships).

Правила:
1. Используй ТОЛЬКО предоставленный контекст. Не добавляй факты из общих знаний.
2. В quotes — от 1 до 5 коротких точных цитат из контекста, подтверждающих ответ.
3. В sources — ID фрагментов (формат: 'kb__N').
4. В confidence — 0.9+ для прямого ответа, 0.5-0.8 для составного, <0.5 если нет ответа.
5. Если в контексте нет ответа — так и скажи, честно, поставь confidence < 0.3."""


def build_rag_answer(query: str, hits: list[dict], trace: TraceWriter, run_id: str) -> RAGAnswer:
    """Ищет в KB через RAG и возвращает ответ LLM."""
    ctx = "\n\n---\n\n".join(
        f"[{h['id']}]\n{h['text']}" for h in hits
    )
    prompt = (
        f"Контекст:\n{ctx}\n\nВопрос студента: {query}\n\nОтвет:"
    )

    t0 = time.time()
    icon = make_client()
    answer: RAGAnswer = icon.chat.completions.create(
        model=MODEL,
        response_model=RAGAnswer,
        messages=[
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    trace.write({
        "kind": "rag_answer", "run_id": run_id,
        "query": query, "elapsed": round(time.time() - t0, 2),
        "sources": answer.sources, "confidence": answer.confidence,
    })
    return answer



TRIAGE_SYSTEM = """Ты — сортировочный агент для помощи по стажировкам и поиску работы в IT.
На основе текста тикета определи категорию, приоритет и канал обработки.

Категории: interview_prep, resume_help, salary_info, company_review, internship_finding, career_advice, contract_advice, test_task, hr_screening, education
Приоритет: low (информационный), medium (нужен ответ), high (влияет на подачу/собес), urgent (кризис — дедлайн подачи сегодня или слитный пароль)
Канал: auto_reply (система ответит сама), ta (нужен ассистент), instructor (нужен преподаватель)

Правила приоритета:
- urgent: дедлайн подачи сегодня, критическая ошибка в резюме перед отправкой
- high: связан с собеседованием в ближайшие дни, тестовое задание
- medium: общий вопрос о стажировках, зарплатах, компаниях
- low: информационный запрос, FAQ"""


def triage_ticket(ticket: Ticket, trace: TraceWriter, run_id: str) -> CategorizedTicket:
    """Агент-сортировщик: классификация + маршрутизация тикета."""
    icon = make_client()
    result: CategorizedTicket = icon.chat.completions.create(
        model=MODEL,
        response_model=CategorizedTicket,
        messages=[
            {"role": "system", "content": TRIAGE_SYSTEM},
            {"role": "user", "content": f"Тикет:\nТема: {ticket.subject}\nСообщение: {ticket.message}"},
        ],
        temperature=0.1,
    )
    trace.write({
        "kind": "triage", "run_id": run_id,
        "ticket_id": ticket.id, "category": result.category,
        "priority": result.priority, "channel": result.suggested_channel,
    })
    return result



JUDGE_SYSTEM = """Ты — судья (LLM-as-judge).
Сравни ответ системы с эталонным ответом (gold answer).

Критерии:
- correct: ответ покрывает основную суть gold_answer (не обязательно дословно)
- hallucination: ответ содержит информацию, которой нет в gold_answer
- ghost_quotes: цитаты, которые не подтверждаются золотым ответом

Будь объективным, но не слишком строгим. Ответ может быть сформулирован иначе."""


def judge_answer(answer: str, gold_summary: str, trace: TraceWriter, run_id: str) -> JudgeVerdict:
    ic = make_client()
    verdict: JudgeVerdict = ic.chat.completions.create(
        model=MODEL, response_model=JudgeVerdict, temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": (
                f"Ответ системы: {answer}\n"
                f"Эталонный ответ (gold): {gold_summary}"
            )},
        ],
    )
    trace.write({"kind": "judge", "run_id": run_id,
                 "correct": verdict.correct, "hallucination": verdict.hallucination,
                 "explanation": verdict.explanation[:200]})
    return verdict



GHOST_SYSTEM = """Ты — детектор ghost quotes.
Проверь, соответствуют ли приведённые цитаты фактическому тексту KB.

Если цитата совпадает с текстом KB — она валидна.
Если цитата выдумана или неточно перефразирована — она ghost.

В ghosts — только фактические ghost-цитаты (пустой, если все валидны)."""


def detect_ghost_quotes(quotes: list[str], kb_text: str, trace: TraceWriter, run_id: str) -> list[str]:
    ic = make_client()
    verdict: GhostDetectionResult = ic.chat.completions.create(
        model=MODEL, response_model=GhostDetectionResult, temperature=0.0,
        messages=[
            {"role": "system", "content": GHOST_SYSTEM},
            {"role": "user", "content": (
                f"Цитаты ответа:\n{chr(10).join(f'  {q}' for q in quotes)}\n"
                f"Текст KB (фрагмент, {len(kb_text)} chars): {kb_text[:3000]}"
            )},
        ],
    )
    trace.write({"kind": "ghost_detection", "run_id": run_id,
                 "ghost_count": len(verdict.ghost_quotes),
                 "ghosts": verdict.ghost_quotes})
    return verdict.ghost_quotes



def stage_generate(out_dir: Path) -> list[Ticket]:
    print("\n=== ЭТАП 1: Генерация персон и тикетов ===", flush=True)
    personas = generate_personas(5)
    print(f"Сгенерировано {len(personas)} персон:", flush=True)
    for p in personas:
        print(f"  {p.name}, {p.year} год, {p.major}, {p.experience_level}, {p.preferred_language}", flush=True)

    synthetic = generate_all_tickets(personas, per_persona=5)
    print(f"Сгенерировано {len(synthetic)} синтетических тикетов", flush=True)

    gold = json.loads((INPUT_DIR / "tickets.json").read_text(encoding="utf-8"))
    gold_tickets = [Ticket(**t) for t in gold]
    for i, t in enumerate(gold_tickets):
        t.id = len(synthetic) + i + 1
    all_tickets = synthetic + gold_tickets

    (out_dir / "tickets.json").write_text(
        json.dumps([t.model_dump() for t in all_tickets],
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Всего: {len(all_tickets)} тикетов сохранено в output/tickets.json", flush=True)
    return all_tickets


def stage_ingest() -> int:
    print("\n=== ЭТАП 2: Индексация KB ===", flush=True)
    n = ingest_kb(str(INPUT_DIR / "knowledge_base.txt"))
    print(f"KB индексирована: {n} чанков", flush=True)
    return n


def stage_triage(tickets: list[Ticket], out_dir: Path, trace: TraceWriter) -> list[dict]:
    print("\n=== ЭТАП 3: Триаж тикетов ===", flush=True)
    results = []
    kb_text = (INPUT_DIR / "knowledge_base.txt").read_text(encoding="utf-8")

    for i, t in enumerate(tickets):
        run_id = str(uuid.uuid4())
        print(f"\n[{i+1}/{len(tickets)}] Тикет {t.id}: {t.subject[:60]}...", flush=True)

        hits = hybrid_search(t.message, k=5)
        rag = build_rag_answer(t.message, hits, trace, run_id)

        auto_solved = rag.confidence >= 0.7
        if auto_solved:
            answer_text = rag.answer
            steps = 1
            tools_used = ["kb_search", "rag"]
            cat_info = {"category": t.category, "priority": "medium", "channel": "auto_reply"}
        else:
            cat = triage_ticket(t, trace, run_id)
            tools_used = ["kb_search", "rag", "triage"]
            cat_info = {"category": cat.category, "priority": cat.priority, "channel": cat.suggested_channel}
            if cat.suggested_channel == "auto_reply" and rag.confidence >= 0.4:
                answer_text = rag.answer
                steps = 2
            elif cat.suggested_channel == "ta":
                answer_text = rag.answer + f"\n\n⚠️ Тикет низкого приоритета (confidence={rag.confidence:.2f}). Рекомендация: перенаправить на TA для ручной проверки."
                steps = 2
            elif cat.suggested_channel == "instructor":
                answer_text = rag.answer + f"\n\n🔴 Тикет требует внимания преподавателя (приоритет: {cat.priority}, категория: {cat.category}). RAG-контекст недостаточен."
                steps = 2
            else:
                answer_text = rag.answer
                steps = 2

        res = {
            "ticket": t.model_dump(),
            "answer": answer_text,
            "rag_answer": rag.model_dump(),
            "auto_solved": auto_solved,
            "steps": steps,
            "tools_used": tools_used,
            "triage": cat_info,
        }
        results.append(res)

    print("\n=== ЭТАП 4-5: Judge + Проверка ghost-цитат ===", flush=True)
    gold_data = json.loads((INPUT_DIR / "tickets.json").read_text(encoding="utf-8"))
    gold_map = {g["id"]: g for g in gold_data}

    judge_results = []
    for r in results:
        run_id = str(uuid.uuid4())
        tid = r["ticket"]["id"]
        if tid in gold_map:
            gold = gold_map[tid]
            jv = judge_answer(r["answer"], gold["gold_answer_summary"], trace, run_id)
            ghosts = detect_ghost_quotes(
                r["rag_answer"]["quotes"],
                kb_text, trace, run_id,
            )
            r["judge"] = jv.model_dump()
            r["ghost_quotes"] = ghosts
            judge_results.append({"ticket_id": tid, "correct": jv.correct,
                                  "hallucination": jv.hallucination,
                                  "ghosts": ghosts})

    (out_dir / "answers.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    (out_dir / "judge_report.json").write_text(
        json.dumps(judge_results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n=== ЭТАП 6: Метрики ===", flush=True)
    if judge_results:
        correct_count = sum(1 for j in judge_results if j["correct"])
        hall_count = sum(1 for j in judge_results if j["hallucination"])
        ghost_count = sum(len(j["ghosts"]) for j in judge_results)
        auto_count = sum(1 for r in results if r["auto_solved"])
        metrics = {
            "total_tickets": len(results),
            "gold_evaluated": len(judge_results),
            "pass_rate": round(correct_count / len(judge_results), 4) if judge_results else 0,
            "ghost_quote_rate": round(ghost_count / max(len(results), 1), 4),
            "hallucination_rate": round(hall_count / len(judge_results), 4) if judge_results else 0,
            "auto_solve_rate": round(auto_count / len(results), 4),
            "avg_steps": round(sum(r["steps"] for r in results) / max(len(results), 1), 2),
        }
    else:
        metrics = {"total_tickets": len(results), "gold_evaluated": 0,
                    "pass_rate": 0, "auto_solve_rate": 0, "avg_steps": 0}

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return results



def main():
    parser = argparse.ArgumentParser(description="Система триаж тикетов")
    parser.add_argument("cmd", choices=["generate", "ingest", "triage", "evaluate", "full"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    trace = TraceWriter(OUTPUT_DIR / "trace.jsonl")

    tickets: list[Ticket] | None = None

    if args.cmd in ("generate", "ingest", "triage", "evaluate"):
        if args.cmd == "generate":
            tickets = stage_generate(OUTPUT_DIR)
        elif args.cmd == "ingest":
            stage_ingest()
        elif args.cmd == "triage":
            data = json.loads((OUTPUT_DIR / "tickets.json").read_text(encoding="utf-8"))
            tickets = [Ticket(**t) for t in data]
            stage_triage(tickets, OUTPUT_DIR, trace)
        elif args.cmd == "evaluate":
            data = json.loads((INPUT_DIR / "tickets.json").read_text(encoding="utf-8"))
            tickets = [Ticket(**t) for t in data]
            stage_triage(tickets, OUTPUT_DIR, trace)

    elif args.cmd == "full":
        tickets = stage_generate(OUTPUT_DIR)
        stage_ingest()
        stage_triage(tickets, OUTPUT_DIR, trace)

    trace.close()
    print("\nПайплайн завершён.", flush=True)


if __name__ == "__main__":
    main()
