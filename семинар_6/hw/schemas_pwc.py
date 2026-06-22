from __future__ import annotations
from typing import Any, Literal

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    id: int = Field(..., description="Порядковый номер, начинается с 1")
    question: str = Field(..., description="Конкретный вопрос")
    expected_tools: list[str] = Field(..., description="Инструменты для подвопроса")
    depends_on: list[int] = Field(default_factory=list, description="ID подвопросов-предшественников")


class Plan(BaseModel):
    reasoning: str = Field(..., description="Почему такая декомпозиция")
    subquestions: list[SubQuestion] = Field(..., description="Может быть пустой, если вопрос нерешаем")


class WorkerAnswer(BaseModel):
    subquestion_id: int
    question_snippet: str = Field(..., description="Первые ~60 символов вопроса")
    answer: str = Field(..., description="Короткий ответ")
    used_tools: list[str] = Field(default_factory=list)
    raw_trace: list[dict[str, Any]] = Field(default_factory=list)


class Verdict(BaseModel):
    ok: bool = Field(..., description="True — всё правильно")
    reason: str = Field(..., description="Почему такое решение")
    action: Literal["accept", "replan", "rework"]
    rework_ids: list[int] = Field(default_factory=list)
