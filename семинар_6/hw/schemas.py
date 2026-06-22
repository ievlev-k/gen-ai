TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_fx_rate",
            "description": "Официальный курс валюты к рублю на дату по данным ЦБ РФ.",
            "parameters": {
                "type": "object",
                "properties": {
                    "currency": {"type": "string", "description": "ISO-код: USD, EUR, CNY, GBP, JPY"},
                    "on_date": {"type": ["string", "null"], "description": "YYYY-MM-DD. Ноль = сегодня."},
                },
                "required": ["currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_key_rate",
            "description": "Ключевая ставка ЦБ РФ на дату, в % годовых.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on_date": {"type": ["string", "null"], "description": "YYYY-MM-DD. Ноль = сегодня."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_inflation",
            "description": "ИПЦ Росстата, % г/г, на конец месяца.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer"},
                    "month": {"type": "integer", "minimum": 1, "maximum": 12},
                },
                "required": ["year", "month"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Калькулятор: +, -, *, /, ^, sqrt, ln, log, exp, скобки.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Математическое выражение."},
                },
                "required": ["expression"],
            },
        },
    },
]
