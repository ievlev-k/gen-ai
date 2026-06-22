from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator


CATEGORIES = Literal[
    "interview_prep",
    "resume_help",
    "salary_info",
    "company_review",
    "internship_finding",
    "career_advice",
    "contract_advice",
    "test_task",
    "hr_screening",
    "education",
]


class Persona(BaseModel):
    name: str
    year: int
    major: str
    experience_level: Literal["beginner", "intermediate", "advanced"]
    preferred_language: Literal["Python", "C++", "JavaScript"]

    @field_validator("year")
    @classmethod
    def _year(cls, v: int) -> int:
        if v < 1 or v > 4:
            raise ValueError("year must be 1-4")
        return v

    @field_validator("experience_level")
    @classmethod
    def _exp(cls, v: str) -> str:
        if v not in ("beginner", "intermediate", "advanced"):
            raise ValueError("experience_level must be beginner|intermediate|advanced")
        return v


class Ticket(BaseModel):
    """Тикет студента о поиске стажировки или работы."""

    id: int = Field(description="Уникальный идентификатор")
    student_name: str = Field(description="Полное имя студента")
    subject: str = Field(description="Короткая тема (до 120 символов)")
    message: str = Field(description="Текст тикета")

    # ── поля для авто-ответа и judge ─────────────────────────────────
    category: CATEGORIES
    complexity: Literal["simple", "medium", "complex"]

    @field_validator("category")
    @classmethod
    def _cat(cls, v: str) -> str:
        allowed = {"interview_prep", "resume_help", "salary_info", "company_review",
                    "internship_finding", "career_advice", "contract_advice",
                    "test_task", "hr_screening", "education"}
        if v not in allowed:
            raise ValueError(f"category must be one of {allowed}")
        return v


class CategorizedTicket(Ticket):
    priority: Literal["low", "medium", "high", "urgent"]
    suggested_channel: Literal["auto_reply", "ta", "instructor"]
    auto_solved: bool


class RAGAnswer(BaseModel):
    answer: str
    quotes: list[str] = Field(min_length=1, max_length=5)
    confidence: float
    sources: list[str]

    @field_validator("confidence")
    @classmethod
    def _conf(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("confidence must be between 0 and 1")
        return v


class JudgeVerdict(BaseModel):
    correct: bool
    hallucination: bool
    ghost_quotes: list[str]
    explanation: str


class GhostDetectionResult(BaseModel):
    ghost_quotes: list[str] = Field(default_factory=list)
    explanation: str = ""


class EvalResult(BaseModel):
    ticket_id: int
    question: str
    answer: str
    correct: bool
    steps: int
    tools_used: list[str]
    tokens: int
    confidence: float
    path_quality: str
    ghost_count: int = 0
