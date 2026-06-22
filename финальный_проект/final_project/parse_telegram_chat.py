import os
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup

input_dir = Path(r"C:\Users\kiril\hw\gen-ai\финальный_проект\final_project\input\ChatExport_2026-06-22")
output_path = Path(r"C:\Users\kiril\hw\gen-ai\финальный_проект\final_project\input\chat_messages.json")

html_files = sorted(input_dir.glob("messages*.html"))
print(f"Найдено {len(html_files)} HTML-файлов")

messages = []
current_sender = None

for html_file in html_files:
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    message_divs = soup.find_all("div", class_=re.compile(r"^message(?!\s+service)"))

    for div in message_divs:
        if "service" in div.get("class", []):
            continue

        date_div = div.find("div", class_="pull_right date details")
        if not date_div:
            continue

        datetime_str = date_div.get("title", "").strip()

        from_name_div = div.find("div", class_="from_name")
        if from_name_div:
            current_sender = from_name_div.get_text(strip=True)

        if not current_sender:
            continue

        text_div = div.find("div", class_="text")
        text = ""
        if text_div:
            for child in text_div.children:
                if isinstance(child, str):
                    text += child
                elif child.name == "br":
                    text += "\n"
                else:
                    text += child.get_text()
            text = re.sub(r"\s+", " ", text).strip()

        messages.append({
            "sender": current_sender,
            "datetime": datetime_str,
            "text": text,
        })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(messages, f, ensure_ascii=False, indent=2)

unique_senders = set(m["sender"] for m in messages)
print(f"Всего сообщений: {len(messages)}")
print(f"Уникальных отправителей: {len(unique_senders)}")
import sys
sys.stdout.reconfigure(encoding="utf-8")
print(f"Отправители: {sorted(unique_senders)}")

print("\nПервые 5 сообщений:")
for m in messages[:5]:
    print(f"  [{m['datetime']}] {m['sender']}: {m['text'][:80]}")
