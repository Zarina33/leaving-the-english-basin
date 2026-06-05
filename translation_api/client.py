"""
NLLB Translator API client.

Reads config from environment:
  TRANSLATION_API_BASE_URL  — base URL (e.g. http://host:port)
  TRANSLATION_API_KEY       — API key, sent as X-API-Key header
  TRANSLATION_NUM_VARIANTS  — translation variants per call (1-5)

Endpoint: POST {BASE_URL}/api/v1/translate
Languages: eng_Latn, rus_Cyrl, kir_Cyrl
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Missing required env var {name}. "
            f"Copy translation_api/.env.example to translation_api/.env and fill it in."
        )
    return val


BASE_URL = _require_env("TRANSLATION_API_BASE_URL").rstrip("/")
API_KEY = _require_env("TRANSLATION_API_KEY")
NUM_VARIANTS = int(os.environ.get("TRANSLATION_NUM_VARIANTS", "1"))
TIMEOUT = int(os.environ.get("TRANSLATION_TIMEOUT", "120"))

LANG_CODES = {
    "en": "eng_Latn", "english": "eng_Latn", "eng_Latn": "eng_Latn",
    "ru": "rus_Cyrl", "russian": "rus_Cyrl", "rus_Cyrl": "rus_Cyrl",
    "ky": "kir_Cyrl", "kyrgyz": "kir_Cyrl", "kir_Cyrl": "kir_Cyrl",
}

# NLLB translates whatever placeholder we use, so we substitute [NAME] with
# something the model preserves verbatim (`@USER` survives in RU; KY may attach
# a case suffix like `@USERге`, which the regex below strips off).
NAME_TOKEN = {"rus_Cyrl": "[ИМЯ]", "kir_Cyrl": "[АТ]", "eng_Latn": "[NAME]"}
PLACEHOLDER = "@nm42"
import re as _re
# Catches the placeholder in all forms NLLB produces:
#   - "@nm42"          (kept verbatim — common)
#   - "@nm42ге"        (Kyrgyz case suffix glued on)
#   - "@nm42 ге"       (case suffix with space)
#   - "нм42"           (Cyrillic transliteration, @ dropped)
#   - "Нм42", "НМ42"   (case variants of the transliteration)
_PLACEHOLDER_RE = _re.compile(
    r"@?(?:nm42|нм42)[A-Za-zА-Яа-яЁёҢңӨөҮү]{0,4}",
    flags=_re.IGNORECASE,
)


@dataclass
class Variant:
    text: str
    confidence: float


@dataclass
class TranslationResult:
    variants: list[Variant]  # variants[0] is best

    @property
    def best(self) -> str:
        return self.variants[0].text if self.variants else ""


def _norm_lang(lang: str) -> str:
    key = lang.strip().lower()
    if key not in LANG_CODES and lang not in LANG_CODES:
        raise ValueError(f"Unknown language: {lang!r}. Use en/ru/ky or NLLB code.")
    return LANG_CODES.get(key, LANG_CODES.get(lang))


def _call_api(text: str, src: str, tgt: str, n: int) -> dict:
    url = f"{BASE_URL}/api/v1/translate"
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    payload = {"text": text, "src_lang": src, "tgt_lang": tgt, "num_variants": n}
    resp = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def translate(
    text: str,
    tgt_lang: str,
    src_lang: str = "eng_Latn",
    num_variants: int | None = None,
    max_retries: int = 3,
) -> TranslationResult:
    """Translate one sentence. Returns up to `num_variants` ranked variants."""
    if not text or not text.strip():
        return TranslationResult(variants=[Variant("", 0.0)])

    src = _norm_lang(src_lang)
    tgt = _norm_lang(tgt_lang)
    n = max(1, min(5, num_variants if num_variants is not None else NUM_VARIANTS))

    # Preserve [NAME] tokens across NMT
    src_placeholder = "[NAME]" if src == "eng_Latn" else NAME_TOKEN[src]
    sent = text.replace(src_placeholder, PLACEHOLDER)

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            data = _call_api(sent, src, tgt, n)
            variants = [Variant(data["translation"], data.get("confidence", 0.0))]
            for alt in data.get("alternatives", []):
                variants.append(Variant(alt["text"], alt.get("confidence", 0.0)))

            tgt_token = NAME_TOKEN[tgt]
            variants = [Variant(_PLACEHOLDER_RE.sub(tgt_token, v.text), v.confidence) for v in variants]
            return TranslationResult(variants=variants[:n])
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"[client] attempt {attempt+1}/{max_retries} failed: {e}; retrying in {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"translate failed after {max_retries} attempts: {last_err}")


def translate_many(
    texts: list[str],
    tgt_lang: str,
    src_lang: str = "eng_Latn",
    num_variants: int | None = None,
    progress: bool = False,
) -> list[TranslationResult]:
    """Translate a list of sentences sequentially (API has no batch endpoint)."""
    out = []
    for i, t in enumerate(texts):
        if progress and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(texts)}]", flush=True)
        out.append(translate(t, tgt_lang, src_lang=src_lang, num_variants=num_variants))
    return out


if __name__ == "__main__":
    samples = [
        "I love this so much!",
        "[NAME] is being annoying again.",
    ]
    for s in samples:
        print(f"\nEN: {s}")
        for lang in ("ru", "ky"):
            res = translate(s, tgt_lang=lang)
            print(f"  {lang}: {res.best}  (conf={res.variants[0].confidence:.3f})")
            for i, v in enumerate(res.variants[1:], 2):
                print(f"    v{i}: {v.text}  (conf={v.confidence:.3f})")
