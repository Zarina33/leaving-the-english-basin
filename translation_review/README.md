# Translation Review

Self-contained Streamlit UI для проверки и правки переводов из NLLB API + Claude.

## Что внутри

```
translation_review/
├── review_app.py         # Streamlit интерфейс
├── run_review.sh         # запускалка (создаёт venv если надо)
├── requirements.txt      # streamlit + pandas
├── sample_100_review.csv # 102 строки: EN + Claude + 4 NLLB-варианта (исходник)
├── sample_100_edited.csv # твои правки (создаётся/обновляется при Save)
└── README.md
```

## Запуск

Требования: Python ≥ 3.10. Всё остальное скрипт поставит сам.

```bash
./run_review.sh
```

Откроется на `http://localhost:8520`. Другой порт: `./run_review.sh 9000`.

Если `streamlit` уже стоит в системе — запустится сразу. Если нет — создаст
venv в `./.venv/` и поставит туда (системный pip не трогается, PEP 668-safe).

## Как пользоваться

- **EN** — read-only, источник
- **✏️ RU (Claude)** — Claude-перевод, правь если нужно
- **✏️ KY — Claude** и **✏️ KY — NLLB** — два варианта кыргызского
- **notes** — заметки

**Логика для KY:** оставь правильный вариант, неправильный полностью очисти
(двойной клик в ячейку → Ctrl-A → Delete). Если оба плохие — отредактируй один.

Нажми **💾 Save** внизу — иначе изменения не сохранятся.

## Выходной файл

`sample_100_edited.csv` содержит:

| колонка | что |
|---|---|
| `id, text_en, emotion` | как в исходнике |
| `text_ru` | финальный русский |
| `text_ky` | автоматически выбранный: Claude если непустой, иначе NLLB |
| `text_ky_source` | `claude` / `nllb` / `both` / пусто — какой источник победил |
| `text_ky_claude`, `text_ky_nllb` | сохраняем оба чтобы знать что было |
| `notes` | твои заметки |

## Перенос между компами

Скопируй `sample_100_edited.csv` на другой комп в эту же папку — приложение
подхватит правки по `id` и продолжишь с того же места.

## Откат

Удали `sample_100_edited.csv` → приложение перезаполнит RU из Claude,
KY-варианты из Claude и NLLB v1.
