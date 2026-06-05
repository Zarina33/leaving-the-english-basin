# translation_api

NLLB Translator API client — drop-in replacement for `scripts/2_translate.py`
(which uses the `claude` CLI).

## API

- Endpoint: `POST {BASE_URL}/api/v1/translate`
- Auth: `X-API-Key` header
- Request: `{text, src_lang, tgt_lang, num_variants}`
  Languages: `eng_Latn`, `rus_Cyrl`, `kir_Cyrl`. `num_variants` ∈ [1, 5].
- Response: `{translation, confidence, alternatives: [{text, confidence}, ...]}`

## Setup

```bash
cp translation_api/.env.example translation_api/.env
# fill in TRANSLATION_API_BASE_URL, TRANSLATION_API_KEY, TRANSLATION_NUM_VARIANTS

pip install requests python-dotenv
```

## Usage

Smoke test:
```bash
python -m translation_api.client
```

Full corpus translation (resumable):
```bash
python -m translation_api.translate
```

## `[NAME]` handling

NLLB is a direct MT model — it doesn't understand instructions. To preserve
`[NAME]` tokens through translation, the client swaps `[NAME]` for `@nm42`
(picked because NLLB leaves it intact across en→ru/ky in all variants),
translates, then maps it to `[ИМЯ]` (RU) / `[АТ]` (KY) via regex that also
strips Kyrgyz case suffixes (`@nm42ге` → `[АТ]`). See [client.py](client.py).

## Output schema

`data/translated/parallel_corpus.csv`:
- Base: `id, text_en, text_ru, text_ky, emotion, conf_ru, conf_ky`
- If `NUM_VARIANTS > 1`: also `text_{ru,ky}_v{1..N}` and `conf_{ru,ky}_v{1..N}`
- `text_ru`/`text_ky` always = best variant (`v1`)
