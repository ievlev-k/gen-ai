"""
Синтетические данные — семинар 2 техника.

Генерация реалистических персон студентов и тикетов через LLM.
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

from schema import Persona, Ticket
from llm_client import get_model, make_client

IC = make_client()
MODEL = get_model()


SYSTEM_PERSONAS = """Ты генерируешь студенческие персоны IT-студентов, ищущих стажировки.
Каждая persona должна быть реалистичной: разные факультеты, уровень опыта, стек технологий.
Имена — русские или распространённые в России."""


def generate_personas(count: int = 5) -> list[Persona]:
    resp: list[Persona] = IC.chat.completions.create(
        model=MODEL,
        response_model=list[Persona],
        messages=[
            {"role": "system", "content": SYSTEM_PERSONAS},
            {"role": "user", "content": f"Создай {count} реалистичных персон IT-студентов, ищущих стажировки."},
        ],
        temperature=0.7,
        max_retries=2,
    )
    return resp


SYSTEM_TICKETS = """Ты генерируешь тикеты от IT-студентов, которые ищут стажировки или работу.
Тикеты должны быть реалистичными: с опечатками, неформальными фразами, как в Telegram-чате.
Категории: interview_prep, resume_help, salary_info, company_review, internship_finding, career_advice, contract_advice, test_task, hr_screening, education.
Сложность: simple — один вопрос; medium — нужен контекст; complex — нюансно или несколько аспектов.
Тикеты должны отражать опыт и уровень персонажа."""


def generate_tickets_from_persona(persona: Persona, count: int = 5) -> list[Ticket]:
    """Сгенерировать тикеты от лица конкретной персоны."""
    desc = (
        f"Персона: {persona.name}, {persona.year} курс, {persona.major}, "
        f"опыт {persona.experience_level}, предпочитает {persona.preferred_language}."
    )
    resp: list[Ticket] = IC.chat.completions.create(
        model=MODEL,
        response_model=list[Ticket],
        messages=[
            {"role": "system", "content": SYSTEM_TICKETS},
            {"role": "user", "content": f"{desc} Создай {count} тикетов, которые эта persona могла бы написать про поиск стажировки/работы."},
        ],
        temperature=0.7,
        max_retries=2,
    )
    return resp


def generate_all_tickets(personas: list[Persona], per_persona: int = 5) -> list[Ticket]:
    """Сгенерировать тикеты для всех персон."""
    all_tickets: list[Ticket] = []
    for i, p in enumerate(personas):
        tickets = generate_tickets_from_persona(p, per_persona)
        # Назначаем последовательные ID
        for j, t in enumerate(tickets):
            t.id = i * per_persona + j + 1
            t.student_name = p.name
            all_tickets.extend(tickets)
            print(f"  Создан тикет {t.id}: {t.subject[:60]}...", flush=True)
    return all_tickets
