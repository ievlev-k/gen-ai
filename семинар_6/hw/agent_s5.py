from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import get_model, make_raw_client
from schemas import TOOL_SCHEMAS
from tools import calculate, get_fx_rate, get_inflation, get_key_rate

TOOLS_IMPL = {
    "get_fx_rate": get_fx_rate,
    "get_key_rate": get_key_rate,
    "get_inflation": get_inflation,
    "calculate": calculate,
}

SYSTEM_PROMPT = """\
Ты — макроэкономический аналитик. ЧИСЛА НИКОГДА НЕ ПРИДУМЫВАЙ — всегда используй tool calls.

Инструменты:
- get_fx_rate: курс валюты к рублю
- get_key_rate: ключевая ставка ЦБ
- get_inflation: ИПЦ % г/г
- calculate: калькулятор

Алгоритм:
1. Для каждого числа — вызов инструмента.
2. Арифметику ТОЛЬКО через calculate.
3. Выдай финальный ответ текстом БЕЗ tool_calls. 1-2 фразы, с числами и единицами.
Формат даты — YYYY-MM-DD."""


def run_agent(
    user_query: str,
    *,
    max_iter: int = 8,
    verbose: bool = True,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    client = make_raw_client()
    model = get_model()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    trace: list[dict[str, Any]] = []

    for step in range(1, max_iter + 1):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto", temperature=0.0,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            trace.append({"step": step, "final": msg.content})
            return {"answer": msg.content, "trace": trace, "steps": step}

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                args = {}
                obs = {"error": f"invalid JSON: {e}"}
            else:
                fn = TOOLS_IMPL.get(name)
                if fn is None:
                    obs = {"error": f"unknown tool: {name}"}
                else:
                    try:
                        obs = fn(**args)
                    except Exception as e:
                        obs = {"error": f"{type(e).__name__}: {e}"}

            trace.append({"step": step, "call": name, "args": args, "obs": obs})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(obs, ensure_ascii=False)})

    return {"answer": None, "trace": trace, "steps": max_iter, "error": f"exceeded max_iter={max_iter}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--trace", type=Path, default=None)
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_agent(q, max_iter=args.max_iter, verbose=not args.quiet)
    print(f"\nОтвет: {res.get('answer') or res.get('error')}")

    if args.trace:
        args.trace.write_text(json.dumps({"query": q, **res}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
