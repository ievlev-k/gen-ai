from __future__ import annotations

from llm_client import get_model, make_client
from schemas_pwc import Plan, Verdict, WorkerAnswer

CRITIC_PROMPT = """\
Ты — критик мульти-агентной системы. Проверь, что ответы отвечают на вопрос, согласованы и получены честно.

Исходный вопрос пользователя:
  «{question}»

План:
{plan_text}

Ответы Исполнителей (без трейса):
{answers_text}

Проверь ПОШАГОВО:
1. Все ли производные числа получены через calculate? Если в ответе есть расчёт, но в used_tools нет calculate — БРАК.
2. Согласованы ли числа между подвопросами?
3. Покрывает ли план ВЕСЬ вопрос?
4. Нет ли ответов вида «(ошибка: ...)» — автоматически БРАК.

Вердикт:
- ok=True, action=accept — всё чисто.
- ok=False, action=rework, rework_ids=[X] — переделать подвопросы.
- ok=False, action=replan — план не охватывает вопрос.
"""


def critic(question: str, plan: Plan, answers: dict[int, WorkerAnswer]) -> Verdict:
    plan_lines = []
    for sq in plan.subquestions:
        tools = ",".join(sq.expected_tools) or "—"
        deps = f" depends_on={sq.depends_on}" if sq.depends_on else ""
        plan_lines.append(f"  {sq.id}. [{tools}]{deps}  «{sq.question}»")
    plan_text = "\n".join(plan_lines) or "  (пустой план)"

    ans_lines = []
    for sq_id in sorted(answers):
        a = answers[sq_id]
        tools = ",".join(a.used_tools) or "—"
        ans_lines.append(f"  {sq_id}. [{tools}] {a.answer}")
    answers_text = "\n".join(ans_lines) or "(ответов нет)"

    client = make_client()
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": CRITIC_PROMPT.format(
                question=question, plan_text=plan_text, answers_text=answers_text)}
        ],
        response_model=Verdict,
        temperature=0.7,
        max_retries=2,
    )
