from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_s5 import run_agent
from orchestrator import run_pwc, VALID_TOOLS


CASES = [
    # --- Оригинальные Q1-Q3 ---
    {
        "id": "Q1",
        "query": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
        "comment": "Класс ошибки C: одиночный считает в уме, не зовёт calculate.",
        "must_have_keywords": ["раз", "usd"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q2",
        "query": "Какая сейчас реальная ключевая ставка, если инфляцию брать по последнему доступному месяцу, а не по году?",
        "comment": "Класс ошибки B: одиночный не умеет искать последний доступный месяц.",
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
    {
        "id": "Q3",
        "query": "Какова накопленная инфляция с января 2022 по март 2026? Рассчитай как произведение всех (1 + ипц_м/100) по месяцам.",
        "comment": "Галлюцинации get_cumulative_inflation. Валидатор должен починить.",
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
        "validator_fixes": True,
    },
    # --- Бонусные ---
    # Q4: вопрос, где Планировщик естественно галлюцинирует get_real_rate.
    # Одиночный агент тоже не справляется (не знает, как посчитать реальную ставку по
    # годам без готового инструмента). PWC без валидатора добавит get_real_rate в план.
    # Валидатор отловит и заставит перепланировать на честные инструменты.
    {
        "id": "Q4",
        "query": "Какая была реальная ключевая ставка ЦБ в 2022, 2023, 2024, 2025 годах? Реальная ставка = номинальная ставка минус годовой уровень инфляции. Покажи для каждого года.",
        "comment": "Для расчёта реальной ставки по каждому году Планировщик склонен добавить get_real_rate (не существует). Валидатор ловит и перепланирует на get_key_rate + get_inflation + calculate.",
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
        "validator_fixes": True,
    },
    # Q5: параллельный вопрос (3+ независимых подвопроса)
    {
        "id": "Q5",
        "query": "Какие курсы USD, EUR и CNY к рублю на сегодня? Сравни их по величине.",
        "comment": "3 независимых get_fx_rate-подвопроса + calculate для сравнения. Идеален для параллелизации.",
        "must_have_keywords": ["usd", "eur", "cny"],
        "forbid_hallucinated_tools": True,
        "parallel": True,
    },
    # Q6: реальный вопрос по макроэкономике
    {
        "id": "Q6",
        "query": "На сколько процентов выросла ключевая ставка ЦБ РФ с начала 2022 года по сегодняшнее время?",
        "comment": "Интересный макро-вопрос: требует курс ставки на начало 2022 и сегодня + calculate.",
        "must_have_keywords": ["%"],
        "forbid_hallucinated_tools": True,
    },
]


def _check_single(case: dict, result: dict) -> dict:
    used = {e["call"] for e in result.get("trace", []) if "call" in e}
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOLS
    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    hallucination_ok = not case.get("forbid_hallucinated_tools", False)
    # Вопросы, требующие арифметики: без calculate — брак
    arith_questions = {"Q1", "Q2", "Q3", "Q4", "Q6"}
    arith_without_calc = (
        case["id"] in arith_questions
        and "calculate" not in used
        and bool(ans)
    )
    ok = bool(ans) and (hallucination_ok or not hallucinated) and must and not arith_without_calc
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "hallucinated": sorted(hallucinated),
        "must_have_ok": must,
        "arith_without_calc": arith_without_calc,
        "answer_preview": (result.get("answer") or "")[:180],
    }


def _check_pwc(case: dict, result: dict, *, use_validator: bool = False) -> dict:
    used = set()
    for t in result.get("trace", []):
        if t.get("kind") == "worker":
            used.update(t.get("used_tools") or [])
    ans = (result.get("answer") or "").lower()
    hallucinated = used - VALID_TOOLS

    plan_tools = set()
    plan_hallucinated = []
    plan = result.get("plan")
    if plan is not None and hasattr(plan, "subquestions"):
        for sq in plan.subquestions:
            plan_tools.update(sq.expected_tools)
        plan_hallucinated = sorted(plan_tools - VALID_TOOLS)

    must = all(kw.lower() in ans for kw in case["must_have_keywords"])

    if use_validator:
        tool_ok = not hallucinated
    else:
        tool_ok = not hallucinated and not plan_hallucinated

    ok = bool(result.get("answer")) and tool_ok and must
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "plan_tools": sorted(plan_tools),
        "hallucinated_in_workers": sorted(hallucinated),
        "hallucinated_in_plan": plan_hallucinated,
        "must_have_ok": must,
        "iterations": result.get("iterations", -1),
        "answer_preview": (result.get("answer") or "")[:180],
    }


def run_case(case: dict, *, n: int = 5) -> dict:
    single_pass = 0
    pwc_pass = 0
    pwc_val_pass = 0

    single_runs = []
    pwc_runs = []
    pwc_val_runs = []

    for i in range(n):
        # Одиночный агент
        try:
            r1 = run_agent(case["query"], max_iter=8, verbose=False)
        except Exception as e:
            r1 = {"answer": None, "error": f"{type(e).__name__}: {e}", "trace": []}
        c1 = _check_single(case, r1)
        single_runs.append(c1)
        if c1["ok"]:
            single_pass += 1

        # PWC без валидатора
        try:
            r2 = run_pwc(case["query"], max_iter=3, verbose=False, use_validator=False)
        except Exception as e:
            r2 = {"answer": None, "error": f"{type(e).__name__}: {e}", "trace": [], "plan": None}
        c2 = _check_pwc(case, r2, use_validator=False)
        pwc_runs.append(c2)
        if c2["ok"]:
            pwc_pass += 1

        # PWC + валидатор
        try:
            r3 = run_pwc(case["query"], max_iter=3, verbose=False, use_validator=True)
        except Exception as e:
            r3 = {"answer": None, "error": f"{type(e).__name__}: {e}", "trace": [], "plan": None}
        c3 = _check_pwc(case, r3, use_validator=True)
        pwc_val_runs.append(c3)
        if c3["ok"]:
            pwc_val_pass += 1

    return {
        "id": case["id"],
        "query": case["query"],
        "comment": case["comment"],
        "n": n,
        "single": {"pass": single_pass, "runs": single_runs},
        "pwc": {"pass": pwc_pass, "runs": pwc_runs},
        "pwc_validator": {"pass": pwc_val_pass, "runs": pwc_val_runs},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true", help="Один прогон каждого кейса")
    ap.add_argument("-n", type=int, default=5, help="Количество прогонов на кейс")
    args = ap.parse_args()
    n = 1 if args.single else args.n

    print(f"Eval 6x3: {len(CASES)} кейсов x 3 конф. x {n} прогонов\n")
    results = []
    for case in CASES:
        print(f"=== {case['id']}: {case['query'][:70]}...")
        r = run_case(case, n=n)
        results.append(r)
        print(f"   single: {r['single']['pass']}/{n}    pwc: {r['pwc']['pass']}/{n}    pwc+val: {r['pwc_validator']['pass']}/{n}")

        for run in r["pwc"]["runs"][:1]:
            if run["hallucinated_in_plan"]:
                print(f"   ⚠ Plan hallucinated: {run['hallucinated_in_plan']}")
        print()

    # Итог
    print("=" * 70)
    print("ИТОГО:")
    print(f"  {'ID':<4} {'Query':<50} {'Single':>8} {'PWC':>8} {'PWC+Val':>8}")
    print(f"  {'-'*4} {'-'*50} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        q_short = r["query"][:48]
        print(f"  {r['id']:<4} {q_short:<50} {r['single']['pass']}/{r['n']:>6} "
              f"{r['pwc']['pass']}/{r['n']:>6} {r['pwc_validator']['pass']}/{r['n']:>6}")

    out = Path(__file__).parent / "eval_6x3_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
