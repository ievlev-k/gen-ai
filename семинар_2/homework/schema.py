"""
Pydantic-схемы для заявок на курсы повышения квалификации (ДПО).
"""

from datetime import date
from typing import Any

from pydantic import BaseModel, Field, field_validator


CITIES_LIST = [
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Нижний Новгород",
    "Самара",
    "Краснодар",
    "Ростов-на-Дону",
    "Уфа",
]


DISTRICTS_BY_CITY: dict[str, list[str]] = {
    "Москва": ["Арбат", "Чертаново", "Марьино", "Бутово", "Сокольники"],
    "Санкт-Петербург": ["Петроградский", "Адмиралтейский", "Выборгский", "Московский"],
    "Новосибирск": ["Центральный", "Заельцовский", "Академгородок", "Дзержинский"],
    "Екатеринбург": ["Чкаловский", "Верх-Исетский", "Железнодорожный", "Ленинский"],
    "Казань": ["Советский", "Вахитовский", "Приволжский", "Ново-Савиновский"],
    "Нижний Новгород": ["Московский", "Сормовский", "Приокский", "Канавино"],
    "Самара": ["Кировский", "Приволжский", "Советский", "Октябрьский"],
    "Краснодар": ["Центральный", "Карасунский", "Алмазный", "Западный"],
    "Ростов-на-Дону": ["Советский", "Первомайский", "Кировский", "Пролетарский"],
    "Уфа": ["Октябрьский", "Кировский", "Ленинский", "Советский"],
}


SPECIALITIES = tuple([
    "инженер-механик",
    "программист",
    "бухгалтер",
    "учитель математики",
    "медсестра",
    "юрист",
    "экономист",
    "менеджер проектов",
    "архитектор",
    "логист",
])


DESIRED_COURSES = tuple([
    "Python для анализа данных",
    "Управление проектами по PMBoK",
    "Машинное обучение и нейросети",
    "Цифровая трансформация бизнеса",
    "Кибербезопасность и защита информации",
    "DevOps и облачные технологии",
    "Финансовый менеджмент",
    "Педагогика высшей школы",
])


class Address(BaseModel):
    city: str
    district: str = Field(min_length=2, max_length=40)

    @field_validator("city")
    @classmethod
    def city_must_be_in_list(cls, v: str) -> str:
        if v not in CITIES_LIST:
            raise ValueError(f"Город «{v}» не из утверждённого списка")
        return v


class Application(BaseModel):
    full_name: str
    age: int = Field(ge=22, le=65)
    address: Address
    speciality: str
    desired_course: str
    graduation_year: int = Field(ge=1980, le=2024)
    years_of_experience: int = Field(ge=0, le=40)

    @field_validator("graduation_year")
    @classmethod
    def graduation_not_too_old(cls, v: int) -> int:
        current_year = date.today().year
        if v < 1980 or v > current_year:
            raise ValueError(
                f"Год выпуска {v} вне диапазона [1980..{current_year}]"
            )
        return v

    @field_validator("age")
    @classmethod
    def age_reasonable(cls, v: int) -> int:
        if v < 22 or v > 65:
            raise ValueError(f"Возраст {v} вне диапазона [22..65]")
        return v

    @field_validator("speciality")
    @classmethod
    def speciality_must_be_valid(cls, v: str) -> str:
        if v not in SPECIALITIES:
            raise ValueError(
                f"Специальность «{v}» не из утверждённого списка. "
                f"Допустимо: {', '.join(SPECIALITIES)}"
            )
        return v

    @field_validator("desired_course")
    @classmethod
    def course_must_be_valid(cls, v: str) -> str:
        if v not in DESIRED_COURSES:
            raise ValueError(
                f"Курс «{v}» не из утверждённого списка. "
                f"Допустимо: {', '.join(DESIRED_COURSES)}"
            )
        return v

    @field_validator("years_of_experience")
    @classmethod
    def experience_matches_graduation(cls, v: int, info: Any) -> int:
        grad_year = info.data.get("graduation_year")
        if grad_year is not None:
            max_exp = date.today().year - grad_year
            if v > max_exp:
                raise ValueError(
                    f"Стаж {v} лет невозможен при выпуске в {grad_year} году "
                    f"(макс. {max_exp})"
                )
        return v

    @property
    def city(self) -> str:
        return self.address.city
