"""
Генератор синтетических заявок на курсы повышения квалификации.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pydantic import ValidationError

from llm_client import get_model, make_client
from schema import Application, CITIES_LIST, DISTRICTS_BY_CITY, SPECIALITIES, DESIRED_COURSES
from prompts import SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE

client = make_client()
MODEL = get_model()

N_APPLICATIONS = 50


def build_prompts(seed_city: str, seed_speciality: str):
    districts = DISTRICTS_BY_CITY.get(seed_city, [])
    system = SYSTEM_PROMPT_TEMPLATE.format(
        cities=", ".join(CITIES_LIST),
        specialities=", ".join(SPECIALITIES),
        courses=", ".join(DESIRED_COURSES),
    )
    user = USER_PROMPT_TEMPLATE.format(
        seed_city=seed_city,
        seed_speciality=seed_speciality,
    )
    return system, user


def generate_one() -> Application:
    seed_city = random.choice(CITIES_LIST)
    seed_speciality = random.choice(SPECIALITIES)
    system, user = build_prompts(seed_city, seed_speciality)
    app = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_model=Application,
        max_retries=3,
        temperature=0.9,
    )
    return app


def flatten_app(app: Application) -> dict:
    d = app.model_dump()
    addr = d.pop("address", {})
    d["city"] = addr.get("city", "")
    d["district"] = addr.get("district", "")
    return d


def save_csv(applications: list[dict], path: str):
    df = pd.DataFrame(applications)
    cols = ["full_name", "age", "city", "district", "speciality",
            "desired_course", "years_of_experience", "graduation_year"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(path, index=False, encoding="utf-8-sig")


def plot_histogram(values: list[str], title: str, xlabel: str, out: str):
    df = pd.DataFrame({"val": values})
    counts = df["val"].value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    counts.plot.bar(color="#4A90D9", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Количество заявок")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def check_distribution(values: list[str], threshold: float, label: str):
    total = len(values)
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    max_val = max(counts.values())
    max_pct = max_val / total * 100
    status = "OK" if max_pct <= threshold else "PREVYSHEN"
    print(f"  {label}: макс {max_pct:.0f}% (порог {threshold}%) → {status}")
    return max_pct


def main():
    print(f"Модель: {MODEL}")
    print(f"Генерирую {N_APPLICATIONS} заявок...\n")

    apps: list[Application] = []
    validator_hits = 0

    for i in range(N_APPLICATIONS):
        print(f"[{i + 1}/{N_APPLICATIONS}] запрос...", end=" ")
        try:
            app = generate_one()
            apps.append(app)
            print(f"{app.full_name}, {app.city}, {app.speciality}")
        except ValidationError as e:
            validator_hits += 1
            print(f"Валидация: {e.errors()[0]['msg'][:80]}")
        except Exception as e:
            print(f"Ошибка: {type(e).__name__}: {str(e)[:80]}")
        time.sleep(0.3)

    valid_count = len(apps)
    print(f"\nСгенерировано валидных: {valid_count} из {N_APPLICATIONS}")
    if validator_hits:
        print(f"@field_validator поймал ошибок: {validator_hits}")

    flat_apps = [flatten_app(a) for a in apps]

    csv_path = Path(__file__).parent / "applications.csv"
    save_csv(flat_apps, str(csv_path))
    print(f"\nCSV сохранён: {csv_path}")

    cities = [a.city for a in apps]
    specialities = [a.speciality for a in apps]

    cities_png = Path(__file__).parent / "cities.png"
    plot_histogram(cities, "Распределение по городам", "Город", str(cities_png))
    print(f"График городов: {cities_png}")

    spec_png = Path(__file__).parent / "specialities.png"
    plot_histogram(specialities, "Распределение по специальностям", "Специальность", str(spec_png))
    print(f"График специальностей: {spec_png}")

    print("\n── Проверка распределения ──")
    check_distribution(cities, 40, "Города")
    check_distribution(specialities, 35, "Специальности")

    if valid_count < N_APPLICATIONS:
        print(f"{valid_count}/{N_APPLICATIONS} валидных заявок.")


if __name__ == "__main__":
    main()