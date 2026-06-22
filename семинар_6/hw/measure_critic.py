from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from schemas_pwc import Plan, SubQuestion, WorkerAnswer

FAKE_BROKEN = [
    {
        "name": "арифметика без calculate",
        "plan": Plan(
            reasoning="Узнать курсы USD и EUR, посчитать разницу.",
            subquestions=[
                SubQuestion(id=1, question="Курс USD к рублю?", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="Курс EUR к рублю?", expected_tools=["get_fx_rate"]),
            ],
        ),
        "answers": {
            1: WorkerAnswer(subquestion_id=1, question_snippet="USD?", answer="USD=82.5, EUR=89, разница=6.5", used_tools=["get_fx_rate"]),
            2: WorkerAnswer(subquestion_id=2, question_snippet="EUR?", answer="EUR=89 руб.", used_tools=["get_fx_rate"]),
        },
    },
    {
        "name": "выдуманное число",
        "plan": Plan(reasoning="Нашли ключевую ставку.", subquestions=[
            SubQuestion(id=1, question="Ключевая ставка ЦБ?", expected_tools=["get_key_rate"]),
        ]),
        "answers": {1: WorkerAnswer(subquestion_id=1, question_snippet="Ставка?", answer="Ключевая ставка сейчас 42.0% годовых.", used_tools=["get_key_rate"])},
    },
    {
        "name": "несогласованные данные",
        "plan": Plan(reasoning="Нашли курс, посчитали отношение.", subquestions=[
            SubQuestion(id=1, question="Курс USD на 01.01.2022?", expected_tools=["get_fx_rate"]),
            SubQuestion(id=2, question="Во сколько раз вырос?", expected_tools=["calculate"], depends_on=[1]),
        ]),
        "answers": {
            1: WorkerAnswer(subquestion_id=1, question_snippet="USD 01.01.22?", answer="Курс USD = 74.29 руб.", used_tools=["get_fx_rate"]),
            2: WorkerAnswer(subquestion_id=2, question_snippet="Отношение?", answer="80 / 74.29 = 1.08 раз", used_tools=["calculate"]),
        },
    },
    {
        "name": "ответ содержит ошибку",
        "plan": Plan(reasoning="Сравнили инфляцию за два месяца.", subquestions=[
            SubQuestion(id=1, question="ИПЦ март 2024?", expected_tools=["get_inflation"]),
            SubQuestion(id=2, question="ИПЦ апрель 2024?", expected_tools=["get_inflation"]),
        ]),
        "answers": {
            1: WorkerAnswer(subquestion_id=1, question_snippet="Март 2024?", answer="ИПЦ = 7.72% г/г", used_tools=["get_inflation"]),
            2: WorkerAnswer(subquestion_id=2, question_snippet="Апрель 2024?", answer="(ошибка: нет данных ИПЦ на 2024-04)", used_tools=["get_inflation"]),
        },
    },
    {
        "name": "непокрытая часть вопроса",
        "plan": Plan(reasoning="Нашли курс USD.", subquestions=[
            SubQuestion(id=1, question="Курс USD сегодня?", expected_tools=["get_fx_rate"]),
        ]),
        "answers": {1: WorkerAnswer(subquestion_id=1, question_snippet="USD?", answer="Курс USD = 82.5 руб.", used_tools=["get_fx_rate"])},
    },
]


def _critic_at_temperature(question: str, plan: Plan, answers: dict[int, WorkerAnswer], *, temperature: float = 0.7):
    from llm_client import get_model, make_client

    cr_prompt = """Ты — критик мульти-агентной системы. Проверь ответы.

Исходный вопрос: «{question}»
План:
{plan_text}
Ответы:
{answers_text}

Проверь:
1. Производные числа через calculate?
2. Согласованы ли числа между подвопросами?
3. Покрывает ли план весь вопрос?
4. Нет ли ошибок в ответах?

Вердикт: accept / rework / replan"""

    plan_lines = []
    for sq in plan.subquestions:
        plan_lines.append(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")
    plan_text = "\n".join(plan_lines) or "  (пусто)"

    ans_lines = []
    for sq_id in sorted(answers):
        a = answers[sq_id]
        ans_lines.append(f"  {sq_id}. [{','.join(a.used_tools)}] {a.answer}")
    answers_text = "\n".join(ans_lines) or "(нет)"

    client = make_client()
    return client.chat.completions.create(
        model=get_model(),
        messages=[{"role": "system", "content": cr_prompt.format(question=question, plan_text=plan_text, answers_text=answers_text)}],
        response_model=__import__("schemas_pwc", fromlist=["Verdict"]).Verdict,
        temperature=temperature,
        max_retries=2,
    )


def measure_critic(n_runs: int = 10):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    case_results = []
    for case in FAKE_BROKEN:
        false_accepts_t0 = 0
        false_accepts_t07 = 0

        futures = {}
        with ThreadPoolExecutor(max_workers=10) as exe:
            for i in range(n_runs):
                futures[exe.submit(critic, "Проверь ответы.", case["plan"], case["answers"])] = "t0"
                futures[exe.submit(_critic_at_temperature, "Проверь ответы.", case["plan"], case["answers"], temperature=0.7)] = "t07"
            for fut in as_completed(futures):
                if futures[fut] == "t0":
                    if fut.result().ok:
                        false_accepts_t0 += 1
                else:
                    if fut.result().ok:
                        false_accepts_t07 += 1

        case_results.append({
            "case": case["name"],
            "t0_false_accepts": f"{false_accepts_t0}/{n_runs}",
            "t07_false_accepts": f"{false_accepts_t07}/{n_runs}",
        })
        print(f"  {case['name']:45} T=0.0: {false_accepts_t0}/{n_runs}  T=0.7: {false_accepts_t07}/{n_runs}")

    total_t0 = sum(int(r["t0_false_accepts"].split("/")[0]) for r in case_results)
    total_t07 = sum(int(r["t07_false_accepts"].split("/")[0]) for r in case_results)
    total_n = len(FAKE_BROKEN) * n_runs
    print(f"\n  Итого: T=0.0: {total_t0}/{total_n}  T=0.7: {total_t07}/{total_n}")

    out = Path(__file__).parent / "critic_measurement.json"
    out.write_text(json.dumps(case_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Сохранено: {out}")
    return case_results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"Замер угодливости Критика: {len(FAKE_BROKEN)} кейсов * {n} прогонов\n")
    measure_critic(n_runs=n)
