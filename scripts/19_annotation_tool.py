"""
Step 19: Annotation tool for Kyrgyz translation quality validation.

Opens a simple web interface (Gradio) where two annotators independently
rate each Kyrgyz translation on a 4-point scale:
  OK / Minor edit / Major edit / Reject

Usage:
  python scripts/19_annotation_tool.py --annotator reviewer1
  python scripts/19_annotation_tool.py --annotator reviewer2

Output: data/annotations/{annotator}_annotations.csv
After both are done, run:
  python scripts/19_annotation_tool.py --compute-iaa
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

CORPUS_PATH = Path("data/translated/parallel_corpus_clean.csv")
ANNOTATIONS_DIR = Path("data/annotations")
LABELS = ["OK", "Minor edit", "Major edit", "Reject"]


def run_gradio(annotator: str):
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Install with: pip install gradio")
        print("Or use the CSV mode instead: python scripts/19_annotation_tool.py --csv")
        return

    df = pd.read_csv(CORPUS_PATH)
    out_path = ANNOTATIONS_DIR / f"{annotator}_annotations.csv"
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing progress
    if out_path.exists():
        done = pd.read_csv(out_path)
        annotations = dict(zip(done["id"], done["quality"]))
        print(f"Loaded {len(annotations)} existing annotations.")
    else:
        annotations = {}

    current_idx = [len(annotations)]

    def get_example(idx):
        if idx < 0:
            idx = 0
        if idx >= len(df):
            idx = len(df) - 1
        current_idx[0] = idx
        row = df.iloc[idx]
        progress = f"**{idx + 1} / {len(df)}** ({len(annotations)} annotated)"
        existing = annotations.get(row["id"], "")
        return (
            progress,
            row["text_en"],
            row["text_ky"],
            row["emotion"],
            existing if existing else None,
            row["id"],
        )

    def save_and_next(quality, item_id):
        if quality and item_id:
            annotations[item_id] = quality
            result = pd.DataFrame([
                {"id": k, "quality": v} for k, v in annotations.items()
            ])
            result.to_csv(out_path, index=False)
        return get_example(current_idx[0] + 1)

    def go_prev():
        return get_example(current_idx[0] - 1)

    def jump_to(idx):
        return get_example(int(idx) - 1)

    with gr.Blocks(title=f"KY Annotation — {annotator}") as app:
        gr.Markdown(f"# Kyrgyz Translation Quality — {annotator}")
        gr.Markdown("Rate how well the Kyrgyz translation preserves the "
                     "emotional content of the English original.")

        progress = gr.Markdown()
        en_text = gr.Textbox(label="English (original)", interactive=False)
        ky_text = gr.Textbox(label="Кыргызча (баалаңыз / rate this)", interactive=False)
        emotion = gr.Textbox(label="Emotion label", interactive=False)
        item_id = gr.Textbox(visible=False)

        quality = gr.Radio(LABELS, label="Quality")

        with gr.Row():
            prev_btn = gr.Button("← Prev")
            next_btn = gr.Button("Save & Next →", variant="primary")
        with gr.Row():
            jump_input = gr.Number(label="Jump to #", value=1, precision=0)
            jump_btn = gr.Button("Go")

        outputs = [progress, en_text, ky_text, emotion, quality, item_id]

        next_btn.click(save_and_next, [quality, item_id], outputs)
        prev_btn.click(go_prev, [], outputs)
        jump_btn.click(jump_to, [jump_input], outputs)
        app.load(lambda: get_example(current_idx[0]), outputs=outputs)

    app.launch(share=False)


def export_csv(annotator: str):
    """Export a simple CSV for annotation without Gradio."""
    df = pd.read_csv(CORPUS_PATH)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANNOTATIONS_DIR / f"{annotator}_template.csv"

    export = df[["id", "text_en", "text_ky", "emotion"]].copy()
    export = export.rename(columns={"text_en": "english_original", "text_ky": "kyrgyz_translation"})
    export["quality"] = ""
    export["notes"] = ""
    export.to_csv(out_path, index=False)
    print(f"Exported {len(export)} rows → {out_path}")
    print(f"\nFill the 'quality' column with: OK / Minor edit / Major edit / Reject")
    print(f"Then rename to {annotator}_annotations.csv")


def compute_iaa():
    """Compute Inter-Annotator Agreement (Cohen's Kappa)."""
    from sklearn.metrics import cohen_kappa_score

    files = list(ANNOTATIONS_DIR.glob("*_annotations.csv"))
    files = [f for f in files if "template" not in f.name]

    if len(files) < 2:
        print(f"Need at least 2 annotation files in {ANNOTATIONS_DIR}/")
        print(f"Found: {[f.name for f in files]}")
        return

    dfs = {}
    for f in files:
        name = f.stem.replace("_annotations", "")
        dfs[name] = pd.read_csv(f).set_index("id")
        print(f"  {name}: {len(dfs[name])} annotations")

    names = list(dfs.keys())
    a1, a2 = dfs[names[0]], dfs[names[1]]

    # Find common IDs
    common = a1.index.intersection(a2.index)
    print(f"\nOverlapping annotations: {len(common)}")

    if len(common) == 0:
        print("No overlap found!")
        return

    y1 = a1.loc[common, "quality"].values
    y2 = a2.loc[common, "quality"].values

    kappa = cohen_kappa_score(y1, y2)
    agree = np.mean(y1 == y2)

    print(f"\n{'='*50}")
    print(f"  Cohen's Kappa:       {kappa:.3f}")
    print(f"  Raw agreement:       {agree:.1%}")
    print(f"  N overlapping:       {len(common)}")
    print(f"{'='*50}")

    # Distribution
    print(f"\nDistribution ({names[0]}):")
    for label in LABELS:
        n = np.sum(y1 == label)
        print(f"  {label:<12}: {n:4d} ({n/len(y1):.1%})")

    print(f"\nDistribution ({names[1]}):")
    for label in LABELS:
        n = np.sum(y2 == label)
        print(f"  {label:<12}: {n:4d} ({n/len(y2):.1%})")

    # Save summary
    summary = {
        "annotator_1": names[0],
        "annotator_2": names[1],
        "n_overlap": len(common),
        "cohens_kappa": round(kappa, 3),
        "raw_agreement": round(agree, 3),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(ANNOTATIONS_DIR / "iaa_summary.csv", index=False)
    print(f"\nSaved → {ANNOTATIONS_DIR / 'iaa_summary.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", type=str, help="Annotator name")
    parser.add_argument("--csv", action="store_true",
                        help="Export CSV template instead of Gradio")
    parser.add_argument("--compute-iaa", action="store_true",
                        help="Compute IAA from two annotation files")
    args = parser.parse_args()

    if args.compute_iaa:
        compute_iaa()
    elif args.csv:
        if not args.annotator:
            args.annotator = "annotator1"
        export_csv(args.annotator)
    else:
        if not args.annotator:
            parser.error("--annotator NAME required (or use --csv / --compute-iaa)")
        run_gradio(args.annotator)


if __name__ == "__main__":
    main()
