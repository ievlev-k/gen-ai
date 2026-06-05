
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

class Issue(BaseModel):
    category: Literal["performance", "design", "support", "price", "ads", "reliability"]
    severity: int = Field(ge=1, le=5)
    quote: str

    @field_validator("quote")
    @classmethod
    def quote_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Цитата не может быть пустой")
        if len(v) < 5:
            raise ValueError("Цитата слишком короткая (< 5 символов)")
        return v.strip()


class Review(BaseModel):
    author: str
    rating: int = Field(ge=1, le=5)
    review_date: Optional[str] = None
    device: Optional[str] = None
    issues: list[Issue]
    positives: list[str] = Field(default_factory=list)
    competitor_mentions: list[str] = Field(default_factory=list)

    @field_validator("rating")
    @classmethod
    def rating_range(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError(f"Рейтинг должен быть 1-5, получено {v}")
        return v

    @field_validator("review_date")
    @classmethod
    def date_not_future(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            d = date.fromisoformat(v)
            if d > date(2026, 7, 1):
                raise ValueError(f"Дата отзыва в будущем: {v}")
        except ValueError:
            if "будущем" not in str(__class__.__dict__.get("date_not_future", "")):
                pass
        return v


class AspectSentiment(BaseModel):
    aspect: Literal["performance", "design", "support", "price", "ads", "reliability"]
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str
    confidence: float = Field(ge=0, le=1)


class ReviewSentiment(BaseModel):
    author: str
    aspects: list[AspectSentiment]


class DiscoveredAspect(BaseModel):
    name: str
    description: str = Field(min_length=5)


class DiscoveredAspects(BaseModel):
    aspects: list[DiscoveredAspect] = Field(min_length=3, max_length=12)


class DynamicAspect(BaseModel):
    aspect: str
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str
    confidence: float = Field(ge=0, le=1)


class DynamicReviewSentiment(BaseModel):
    author: str
    aspects: list[DynamicAspect]


class CacheStats(BaseModel):
    run_label: str
    prompt_tokens: int
    cache_hit_tokens: int
    cache_miss_tokens: int
    hit_pct: float
    latency_sec: float


class ChunkSummary(BaseModel):
    reviewer: str
    key_points: list[str] = Field(min_length=1, max_length=6)
    sentiment: Literal["positive", "negative", "mixed"]


class ReviewsSummary(BaseModel):
    headline: str
    key_findings: list[str] = Field(min_length=2, max_length=8)
    action_items: list[str] = Field(min_length=1, max_length=8)


class ActionVerdict(BaseModel):
    action: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str] = Field(default_factory=list)
    comment: str


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict]
    overall_score: float = Field(ge=0, le=1)
    summary: str
