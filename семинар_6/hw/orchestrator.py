from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from llm_client import get_model, make_raw_client
from planner import planner
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from worker import worker

VALID_TOOLS = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def validate_plan(plan: Plan) -> list[str]:
    errors: list[str] = []
    for sq in plan.subquestions:
        for tool in sq.expected_tools:
            if tool not in VALID_TOOLS:
                errors.append(f"SubQuestion #{sq.id}: выдуманный инструмент «{tool}»")
        for dep_id in sq.depends_on:
            if dep_id not in {s.id for s in plan.subquestions}:
                errors.append(f"SubQuestion #{sq.id}: зависит от несуществующего #{dep_id}")
    return errors


def _topological_sort(subqs: list[SubQuestion]) -> list[SubQuestion]:
    by_id = {s.id: s for s in subqs}
    ordered: list[SubQuestion] = []
    visited: set[int] = set()

    def visit(node_id: int, path: list[int]):
        if node_id in visited:
            return
        if node_id in path:
            raise ValueError(f"Цикл: {path + [node_id]}")
        if node_id not in by_id:
            return
        for dep in by_id[node_id].depends_on:
            visit(dep, path + [node_id])
        visited.add(node_id)
        ordered.append(by_id[node_id])

    for sq in subqs:
        visit(sq.id, [])
    return ordered


def _topological_levels(subqs: list[SubQuestion]) -> list[list[SubQuestion]]:
    if not subqs:
        return []
    by_id = {s.id: s for s in subqs}
    in_degree: dict[int, int] = {s.id: 0 for s in subqs}
    dependents: dict[int, list[int]] = {s.id: [] for s in subqs}

    for sq in subqs:
        for dep_id in sq.depends_on:
            if dep_id in in_degree:
                in_degree[sq.id] += 1
                dependents[dep_id].append(sq.id)

    levels: list[list[SubQuestion]] = []
    remaining = dict(in_degree)
    while remaining:
        level_ids = [nid for nid, deg in remaining.items() if deg == 0]
        if not level_ids:
            raise ValueError("Цикл в зависимости")
        levels.append([by_id[nid] for nid in sorted(level_ids)])
        for nid in level_ids:
            del remaining[nid]
            for child_id in dependents.get(nid, []):
                if child_id in remaining:
                    remaining[child_id] -= 1
    return levels


def execute_level(level: list[SubQuestion], prev_answers: dict[int, WorkerAnswer]) -> dict[int, WorkerAnswer]:
    results: dict[int, WorkerAnswer] = {}

    def _run(sq: SubQuestion) -> WorkerAnswer:
        return worker(sq, prev_answers=prev_answers)

    if len(level) <= 1:
        for sq in level:
            results[sq.id] = _run(sq)
    else:
        with ThreadPoolExecutor(max_workers=len(level)) as exe:
            futures = {exe.submit(_run, sq): sq for sq in level}
            for fut in as_completed(futures):
                sq = futures[fut]
                results[sq.id] = fut.result()

    return results


def _synthesize(question: str, plan: Plan, answers: dict[int, WorkerAnswer]) -> str:
    if not answers:
        return "(нет ответов)"

    parts = []
    for i in sorted(answers):
        a = answers[i]
        parts.append(f"  Подвопрос {i}: {a.answer}")
    answers_text = "\n".join(parts)

    prompt = f"""Исходный вопрос: «{question}»
Результаты подвопросов:
{answers_text}
Собери в финальный ответ: 1-2 предложения, с числами и единицами."""

    client = make_raw_client()
    resp = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": "Синтезируй ответ кратко и по делу."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content or " · ".join(a.answer for a in answers.values())


def run_pwc(
    question: str,
    *,
    max_iter: int = 3,
    verbose: bool = True,
    use_validator: bool = False,
    parallel: bool = False,
) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []

    plan = planner(question)
    trace.append({
        "iter": 0, "kind": "plan",
        "reasoning": plan.reasoning,
        "subquestions": [sq.model_dump() for sq in plan.subquestions],
    })

    if use_validator:
        validation_errors = validate_plan(plan)
        if validation_errors:
            if verbose:
                print(f"[validator] Ошибки: {validation_errors}")
            trace.append({"iter": 0, "kind": "validation", "errors": validation_errors})
            feedback = f"Инструменты не существуют: {validation_errors}. Используй только: {VALID_TOOLS}."
            plan = planner(question, feedback=feedback)
            trace.append({
                "iter": 0, "kind": "plan_validated",
                "reasoning": plan.reasoning,
                "subquestions": [sq.model_dump() for sq in plan.subquestions],
            })

    answers: dict[int, WorkerAnswer] = {}

    for iter_num in range(1, max_iter + 1):
        answers = {}

        if parallel:
            levels = _topological_levels(plan.subquestions)
            for lvl in levels:
                level_results = execute_level(lvl, answers)
                answers.update(level_results)
                for sq_id, a in level_results.items():
                    trace.append({"iter": iter_num, "kind": "worker", "sq_id": sq_id, "used_tools": a.used_tools, "answer": a.answer})
        else:
            ordered = _topological_sort(plan.subquestions)
            for sq in ordered:
                ans = worker(sq, prev_answers=answers)
                answers[sq.id] = ans
                trace.append({"iter": iter_num, "kind": "worker", "sq_id": sq.id, "used_tools": ans.used_tools, "answer": ans.answer})

        verdict = critic(question, plan, answers)
        trace.append({"iter": iter_num, "kind": "verdict", "ok": verdict.ok, "action": verdict.action, "reason": verdict.reason, "rework_ids": verdict.rework_ids})

        if verdict.ok:
            final = _synthesize(question, plan, answers)
            return {"answer": final, "plan": plan, "answers": answers, "trace": trace, "iterations": iter_num}

        if verdict.action == "replan":
            plan = planner(question, feedback=verdict.reason)
            trace.append({"iter": iter_num, "kind": "replan", "reasoning": plan.reasoning, "subquestions": [sq.model_dump() for sq in plan.subquestions]})
        elif verdict.action == "rework":
            feedback = f"Нужно переделать подвопросы {verdict.rework_ids}. Замечание: {verdict.reason}"
            plan = planner(question, feedback=feedback)
            trace.append({"iter": iter_num, "kind": "rework", "rework_ids": verdict.rework_ids, "reasoning": plan.reasoning})

    return {"answer": None, "error": f"не удалось за {max_iter} итераций", "plan": plan, "answers": answers, "trace": trace, "iterations": max_iter}


def run_pwc_timed(question: str, *, parallel: bool = False, **kw) -> tuple[dict[str, Any], float]:
    t0 = time.time()
    res = run_pwc(question, parallel=parallel, verbose=False, **kw)
    return res, time.time() - t0


def _serialize(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in res.items():
        if k == "plan" and v is not None:
            out[k] = v.model_dump()
        elif k == "answers":
            out[k] = {i: a.model_dump() for i, a in v.items()}
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--validator", action="store_true")
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--trace", type=Path, default=None)
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_pwc(q, max_iter=args.max_iter, verbose=not args.quiet, use_validator=args.validator, parallel=args.parallel)

    print(f"\nОтвет: {res.get('answer') or res.get('error')}")
    print(f"(итераций: {res.get('iterations', '?')})")

    if args.trace:
        args.trace.write_text(json.dumps({"query": q, **_serialize(res)}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
