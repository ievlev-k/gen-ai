from __future__ import annotations

from llm_client import get_model, make_client
from schemas_pwc import Plan

SYSTEM_PROMPT = """\
Ты — планировщик макроэкономического агента. Разложи сложный вопрос пользователя на 1-5 подвопросов.

Доступные инструменты (НЕ придумывай других):
- get_fx_rate(currency, on_date): курс валюты к рублю на дату.
- get_key_rate(on_date): ключевая ставка ЦБ на дату.
- get_inflation(year, month): ИПЦ г/г на конец месяца.
- calculate(expression): безопасный калькулятор.

ПРАВИЛА:
1. Любую арифметику — отдельный подвопрос с calculate.
2. Если подвопрос N зависит от K — поставь K в depends_on.
3. Для «последний доступный период» — первым шагом узнай доступный период.
4. Если задача не решается tools — верни reasoning с объяснением и subquestions=[].
Цель — минимальный корректный план.
"""


def planner(question: str, *, feedback: str | None = None) -> Plan:
    client = make_client()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    if feedback:
        messages.append(
            {"role": "user", "content": f"Предыдущая попытка не прошла проверку. Замечание: {feedback}"}
        )

    return client.chat.completions.create(
        model=get_model(),
        messages=messages,
        response_model=Plan,
        temperature=0.0,
        max_retries=2,
    )


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Во сколько раз USD подорожал с 1 января 2022 по сегодня?"
    plan = planner(q)
    print(f"План (reasoning): {plan.reasoning}\n")
    for sq in plan.subquestions:
        deps = f" ← ждёт {sq.depends_on}" if sq.depends_on else ""
        print(f"  {sq.id}. [{','.join(sq.expected_tools)}]{deps}  {sq.question}")
