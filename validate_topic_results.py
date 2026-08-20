"""
validate_topic_results.py

Quick-read validation report generator for run_topic_modelling.py output.

Reads the per-run output folder produced by run_topic_modelling.py
(e.g. BERTopic_results/raw/137_raw/) and turns it into ONE Markdown file
you can scroll through quickly, instead of opening N separate
center_<topic_id>_processed.json files by hand.

Expects, in --results-dir:
    - topic_info.csv                     topic sizes + keywords (added by the
                                          patched run_topic_modelling.py)
    - center_<topic_id>_processed.json   representative paragraphs per topic
                                          (already produced by the original script)

If topic_info.csv is missing (i.e. you're validating an older run), the
script still works, just without size/keyword columns.

WHAT THIS SCRIPT DOES vs. DOES NOT DO
--------------------------------------
It does NOT decide whether your topics are "philosophically coherent" --
that judgment call is yours, the same way your README treats "do retrieved
passages read as on-topic?" as a human checkpoint, not something a script
certifies. What it DOES do is surface the evidence you need to make that
call quickly and consistently across runs:

    1. Topic sizes + keywords, sorted largest-first
    2. Outlier ("-1") share -- a high value usually means HDBSCAN/UMAP
       parameters need retuning, not that your topic is real but small
    3. The saved representative paragraphs for every topic, in one place
    4. Cross-topic near-duplicate representative paragraphs -- a same/near-
       identical passage assigned to two DIFFERENT topics usually signals
       over-fragmentation (two topics that should really be one), or a
       boilerplate phrase (e.g. citation format) dominating the embedding
    5. Key domain terms (direct, indirect, perception, vision, ...) bolded
       throughout, so the words that matter for eyeballing "is this really
       about direct perception?" jump out while scrolling

Usage:
    python validate_topic_results.py --results-dir BERTopic_results/raw/137_raw
    python validate_topic_results.py --results-dir BERTopic_results/raw/137_raw --output report.md
    python validate_topic_results.py --results-dir BERTopic_results/raw/137_raw --dup-threshold 0.9
    python validate_topic_results.py --results-dir BERTopic_results/raw/137_raw \\
        --highlight-terms "direct,indirect,perception,vision,affordance"
    python validate_topic_results.py --results-dir BERTopic_results/raw/137_raw --highlight-terms ""

Comparing multiple runs (e.g. different --min-cluster-size settings):
    python validate_topic_results.py --results-dir BERTopic_results/raw/137_raw
    python validate_topic_results.py --results-dir BERTopic_results/raw/afb_raw
    # then diff the two VALIDATION_REPORT.md "Summary" sections
"""

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def load_topic_info(results_dir: Path) -> Optional[pd.DataFrame]:
    """Load topic_info.csv (Topic, Count, Name, Representation) if present."""
    path = results_dir / "topic_info.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_representative_paragraphs(results_dir: Path) -> Dict[int, List[str]]:
    """Load every center_<topic_id>_processed.json file into {topic_id: [texts]}."""
    topic_paragraphs = {}
    pattern = re.compile(r"center_(-?\d+)_processed\.json$")

    for f in sorted(results_dir.glob("center_*_processed.json")):
        m = pattern.match(f.name)
        if not m:
            continue
        topic_id = int(m.group(1))
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        texts = [p["text"] for p in data.get("paragraphs", [])]
        topic_paragraphs[topic_id] = texts

    return topic_paragraphs


def find_cross_topic_duplicates(
    topic_paragraphs: Dict[int, List[str]], threshold: float = 0.85
) -> List[tuple]:
    """
    Flag representative paragraphs that are near-identical across DIFFERENT
    topics. Only runs over the small set of *representative* examples saved
    per topic (a handful each), so a simple O(n^2) SequenceMatcher comparison
    is cheap -- this is not run over the full corpus.
    """
    flat = []
    for topic_id, texts in topic_paragraphs.items():
        for t in texts:
            flat.append((topic_id, t))

    duplicates = []
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            topic_a, text_a = flat[i]
            topic_b, text_b = flat[j]
            if topic_a == topic_b:
                continue
            ratio = SequenceMatcher(None, text_a, text_b).ratio()
            if ratio >= threshold:
                duplicates.append((topic_a, topic_b, ratio, text_a, text_b))

    return duplicates


DEFAULT_HIGHLIGHT_TERMS = [
    "direct", "indirect", "direct perception", "direct realism", "directly", "indirectly"
]


def bold_terms(text: str, terms: List[str]) -> str:
    """
    Wrap whole-word (case-insensitive) matches of the given terms in **bold**,
    convert them to CAPITALS, and color them red.

    Longer terms are matched first so multi-word phrases (e.g. "naive realism")
    take priority over any single-word overlap, and \\b boundaries ensure
    "direct" inside "indirect" is never matched.
    """
    if not terms:
        return text
    sorted_terms = sorted(set(t for t in terms if t.strip()), key=len, reverse=True)
    if not sorted_terms:
        return text
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(t) for t in sorted_terms) + r")\b",
        re.IGNORECASE,
    )
    return pattern.sub(
        lambda m: f'<span style="color:red">**{m.group(0).upper()}**</span>', text
    )


def markdown_table(
    df: pd.DataFrame,
    columns: List[str],
    highlight_terms: Optional[List[str]] = None,
    highlight_cols: tuple = ("Name", "Representation"),
) -> str:
    """Minimal Markdown table formatter (avoids depending on the optional
    'tabulate' package that pandas.to_markdown() requires)."""
    cols = [c for c in columns if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            cell = str(row[c]).replace("\n", " ").replace("|", "/")
            if highlight_terms and c in highlight_cols:
                cell = bold_terms(cell, highlight_terms)
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def topic_sort_key(topic_id: int, topic_info: Optional[pd.DataFrame]):
    """Outliers (-1) last; otherwise sort by size descending if available,
    else by topic id."""
    if topic_id == -1:
        return (1, 0)
    if topic_info is not None and "Count" in topic_info.columns:
        row = topic_info[topic_info["Topic"] == topic_id]
        size = int(row["Count"].iloc[0]) if not row.empty else 0
        return (0, -size)
    return (0, topic_id)


def build_report(
    results_dir: Path,
    topic_info: Optional[pd.DataFrame],
    topic_paragraphs: Dict[int, List[str]],
    duplicates: List[tuple],
    dup_threshold: float,
    highlight_terms: Optional[List[str]] = None,
) -> str:
    lines = ["# Topic Modeling Validation Report", f"\nSource: `{results_dir}`\n"]

    total_docs = None
    outlier_pct = None
    if topic_info is not None and "Count" in topic_info.columns:
        total_docs = int(topic_info["Count"].sum())
        outlier_row = topic_info[topic_info["Topic"] == -1]
        if not outlier_row.empty:
            outlier_count = int(outlier_row["Count"].iloc[0])
            outlier_pct = 100 * outlier_count / total_docs

    n_topics = len([t for t in topic_paragraphs if t != -1])

    lines.append("## Summary\n")
    lines.append(f"- Topics found (excluding outliers): **{n_topics}**")
    if total_docs is not None:
        lines.append(f"- Total documents: **{total_docs}**")
    if outlier_pct is not None:
        flag = " -- HIGH, consider retuning UMAP/HDBSCAN parameters" if outlier_pct > 30 else ""
        lines.append(f"- Outlier share (topic -1): **{outlier_pct:.1f}%**{flag}")
    if topic_info is None:
        lines.append(
            "- `topic_info.csv` not found in this results dir -- sizes and keywords "
            "unavailable. Re-run with the patched `run_topic_modelling.py` to get this."
        )
    dup_flag = " -- check for over-fragmented topics below" if duplicates else ""
    lines.append(
        f"- Near-duplicate representative paragraphs across different topics "
        f"(similarity >= {dup_threshold}): **{len(duplicates)}**{dup_flag}"
    )
    if highlight_terms:
        lines.append(f"- Bolded terms below: {', '.join(sorted(highlight_terms))}")
    lines.append("")

    if topic_info is not None:
        lines.append("## Topics overview (sorted by size)\n")
        sort_col = "Count" if "Count" in topic_info.columns else "Topic"
        sorted_info = topic_info.sort_values(sort_col, ascending=False)
        lines.append(markdown_table(
            sorted_info, ["Topic", "Count", "Name", "Representation"],
            highlight_terms=highlight_terms,
        ))
        lines.append("")

    lines.append("## Representative paragraphs by topic\n")
    order = sorted(topic_paragraphs.keys(), key=lambda t: topic_sort_key(t, topic_info))

    for topic_id in order:
        label = "Outliers (-1)" if topic_id == -1 else f"Topic {topic_id}"
        keywords = ""
        size_str = ""
        if topic_info is not None:
            row = topic_info[topic_info["Topic"] == topic_id]
            if not row.empty:
                if "Representation" in row.columns:
                    keywords = str(row["Representation"].iloc[0])
                if "Count" in row.columns:
                    size_str = f" ({int(row['Count'].iloc[0])} docs)"

        lines.append(f"### {label}{size_str}\n")
        if keywords:
            lines.append(f"**Keywords:** {bold_terms(keywords, highlight_terms)}\n")
        for i, text in enumerate(topic_paragraphs[topic_id]):
            snippet = " ".join(text.strip().split())
            snippet = bold_terms(snippet, highlight_terms)
            lines.append(f"{i + 1}. {snippet}\n")
        lines.append("")

    if duplicates:
        lines.append("## Cross-topic near-duplicates (check these)\n")
        lines.append(
            "These representative paragraphs are near-identical but landed in "
            "*different* topics. Worth a quick look at whether these topics "
            "should be merged, or whether it's just a repeated boilerplate "
            "phrase (citation format, abstract template, etc.) rather than "
            "genuine thematic content.\n"
        )
        for topic_a, topic_b, ratio, text_a, text_b in sorted(duplicates, key=lambda d: -d[2]):
            snippet_a = " ".join(text_a.strip().split())[:150]
            snippet_b = " ".join(text_b.strip().split())[:150]
            snippet_a = bold_terms(snippet_a, highlight_terms)
            snippet_b = bold_terms(snippet_b, highlight_terms)
            lines.append(f"- Topic {topic_a} vs Topic {topic_b} (similarity {ratio:.2f}):")
            lines.append(f"  - \"{snippet_a}...\"")
            lines.append(f"  - \"{snippet_b}...\"")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a single quick-read Markdown validation report "
        "from a run_topic_modelling.py output directory."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Output dir from run_topic_modelling.py, e.g. BERTopic_results/raw/137_raw",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the Markdown report (default: <results-dir>/VALIDATION_REPORT.md)",
    )
    parser.add_argument(
        "--dup-threshold",
        type=float,
        default=0.85,
        help="Similarity ratio (0-1) above which two representative paragraphs in "
        "different topics are flagged as near-duplicates (default: 0.85)",
    )
    parser.add_argument(
        "--highlight-terms",
        type=str,
        default=",".join(DEFAULT_HIGHLIGHT_TERMS),
        help="Comma-separated list of words/phrases to bold in the report "
        "(case-insensitive, whole-word match). Pass an empty string to disable. "
        f"Default: {', '.join(DEFAULT_HIGHLIGHT_TERMS)}",
    )
    args = parser.parse_args()

    highlight_terms = [t.strip() for t in args.highlight_terms.split(",") if t.strip()]

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    topic_info = load_topic_info(results_dir)
    topic_paragraphs = load_representative_paragraphs(results_dir)

    if not topic_paragraphs:
        raise SystemExit(f"No center_*_processed.json files found in {results_dir}")

    duplicates = find_cross_topic_duplicates(topic_paragraphs, threshold=args.dup_threshold)
    report = build_report(
        results_dir, topic_info, topic_paragraphs, duplicates, args.dup_threshold,
        highlight_terms=highlight_terms,
    )

    output_path = Path(args.output) if args.output else results_dir / "VALIDATION_REPORT.md"
    output_path.write_text(report, encoding="utf-8")

    # Terminal summary for a quick look over SSH/screen without opening the file
    print("=" * 70)
    print("TOPIC VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Results dir: {results_dir}")
    print(f"Topics found: {len([t for t in topic_paragraphs if t != -1])}")
    if topic_info is not None and "Count" in topic_info.columns:
        total = int(topic_info["Count"].sum())
        print(f"Total documents: {total}")
        outlier_row = topic_info[topic_info["Topic"] == -1]
        if not outlier_row.empty:
            pct = 100 * int(outlier_row["Count"].iloc[0]) / total
            note = "  <-- high, consider retuning" if pct > 30 else ""
            print(f"Outlier share: {pct:.1f}%{note}")
    else:
        print("topic_info.csv not found -- sizes/keywords unavailable "
              "(re-run with the patched run_topic_modelling.py)")
    print(f"Cross-topic near-duplicates flagged: {len(duplicates)}")
    print(f"\nFull report written to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
