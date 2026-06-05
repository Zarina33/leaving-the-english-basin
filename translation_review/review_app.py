"""
Translation review UI — editable table.

Self-contained: loads sample_100_review.csv from same folder as this script.
Saves edits to sample_100_edited.csv (also in same folder).

Layout:
  EN                (read-only)   — source
  ✏️ RU (Claude)     (editable)    — Claude RU as base, edit if needed
  ✏️ KY — Claude    (editable)    — Claude KY variant — DELETE if wrong
  ✏️ KY — NLLB      (editable)    — NLLB KY variant   — DELETE if wrong
  notes             (editable)

For KY: keep the correct variant, delete the wrong one. Save merges them into
one `text_ky` column on write (Claude wins if both non-empty).

Run from project root:
    ./translation_review/run_review.sh
or:
    streamlit run translation_review/review_app.py
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

_SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("REVIEW_DATA_DIR", _SCRIPT_DIR))
SOURCE = DATA_DIR / "sample_100_review.csv"
OUTPUT = DATA_DIR / "sample_100_edited.csv"


def load_initial() -> pd.DataFrame:
    src = pd.read_csv(SOURCE)
    base = pd.DataFrame({
        "id":             src["id"],
        "emotion":        src["emotion"],
        "text_en":        src["text_en"],
        "text_ru":        src["claude_ru"].fillna(""),
        "text_ky_claude": src["claude_ky"].fillna(""),
        "text_ky_nllb":   src["nllb_ky_v1"].fillna(""),
        "notes":          "",
    })
    if OUTPUT.exists():
        prev = pd.read_csv(OUTPUT).fillna("")
        prev_idx = prev.set_index("id")
        for col in ["text_ru", "text_ky_claude", "text_ky_nllb", "notes"]:
            if col in prev_idx.columns:
                base[col] = base["id"].map(prev_idx[col]).fillna(base[col])
    return base


def main():
    st.set_page_config(page_title="Translation Review", layout="wide")
    st.title("Перевод — проверка и правка")

    if not SOURCE.exists():
        st.error(
            f"Не найден файл с переводами:\n`{SOURCE}`\n\n"
            f"Положи `sample_100_review.csv` рядом с `review_app.py` "
            f"или укажи путь через переменную окружения `REVIEW_DATA_DIR`."
        )
        st.stop()

    if "df" not in st.session_state:
        st.session_state.df = load_initial()
    df = st.session_state.df

    st.caption(
        "**RU**: правь Claude-перевод. "
        "**KY**: два варианта — оставь правильный, удали неправильный (двойной клик → Ctrl-A → Delete). "
        "Если оба неправильные — отредактируй один. "
        "Нажми **💾 Save** внизу чтобы сохранить."
    )

    view_cols = ["emotion", "text_en", "text_ru", "text_ky_claude", "text_ky_nllb", "notes"]

    edited = st.data_editor(
        df[view_cols],
        column_config={
            "emotion":        st.column_config.TextColumn("emotion",        width="small",  disabled=True),
            "text_en":        st.column_config.TextColumn("EN",             width="large",  disabled=True),
            "text_ru":        st.column_config.TextColumn("✏️ RU (Claude)",  width="large"),
            "text_ky_claude": st.column_config.TextColumn("✏️ KY — Claude", width="medium"),
            "text_ky_nllb":   st.column_config.TextColumn("✏️ KY — NLLB",   width="medium"),
            "notes":          st.column_config.TextColumn("notes",          width="medium"),
        },
        hide_index=False,
        use_container_width=True,
        height=700,
        num_rows="fixed",
        key="editor",
    )

    col_save, col_status = st.columns([1, 4])
    if col_save.button("💾 Save", type="primary", use_container_width=True):
        ky_claude = edited["text_ky_claude"].fillna("").astype(str).str.strip()
        ky_nllb   = edited["text_ky_nllb"].fillna("").astype(str).str.strip()
        text_ky_final = ky_claude.where(ky_claude != "", ky_nllb)
        ky_source = pd.Series("", index=edited.index)
        ky_source[(ky_claude != "") & (ky_nllb == "")] = "claude"
        ky_source[(ky_claude == "") & (ky_nllb != "")] = "nllb"
        ky_source[(ky_claude != "") & (ky_nllb != "")] = "both"

        out = pd.DataFrame({
            "id":             df["id"],
            "emotion":        df["emotion"],
            "text_en":        df["text_en"],
            "text_ru":        edited["text_ru"],
            "text_ky":        text_ky_final,
            "text_ky_source": ky_source,
            "text_ky_claude": edited["text_ky_claude"],
            "text_ky_nllb":   edited["text_ky_nllb"],
            "notes":          edited["notes"],
        })
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUTPUT, index=False)
        st.session_state.df.loc[:, view_cols] = edited[view_cols].values
        col_status.success(
            f"Сохранено в {OUTPUT.name}  ·  KY: "
            f"claude={int((ky_source=='claude').sum())}, "
            f"nllb={int((ky_source=='nllb').sum())}, "
            f"both={int((ky_source=='both').sum())}, "
            f"empty={int(text_ky_final.eq('').sum())}"
        )


if __name__ == "__main__":
    main()
