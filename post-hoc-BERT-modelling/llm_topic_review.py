#!/usr/bin/env python3
"""
Send BERTopic outputs to a local LLM served by LM Studio and print its review.

Point it at a results directory; it finds topic_info.csv and VALIDATION_REPORT.md,
packs them into a single prompt under a character budget, and streams the answer.

    python llm_topic_review.py BERTopic_results/raw/407_raw
    python llm_topic_review.py <dir> --prompt-file prompts/coherence.txt
    python llm_topic_review.py <dir> --dry-run          # inspect the prompt, send nothing
    python llm_topic_review.py --list-models            # check what LM Studio has loaded
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
import requests

DEFAULT_BASE_URL = "http://localhost:1234/v1"

DEFAULT_SYSTEM = (
    "You are a careful research assistant helping evaluate a BERTopic model fitted "
    "over philosophy-of-mind and cognitive-science literature. Refer to topics by "
    "their number. Ground every claim in the keywords and passages you were given. "
    "If the evidence for a judgement is thin, say so instead of guessing."
)

DEFAULT_PROMPT = """\
Review the topic model below.

1. For each topic, propose a short label (<= 6 words) and a one-sentence gloss.
2. Flag any topic whose keywords do not hang together as a single concept, and name
   the specific keywords that do not fit.
3. Flag pairs of topics that look like the same underlying distinction split in two.
4. Comment on whether the outlier share is acceptable for this corpus.

Do not invent topics that are not in the data. Do not speculate about papers you
cannot see.
"""

# ---------------------------------------------------------------- file discovery


def find_file(root: Path, name: str) -> Path | None:
    """Shallowest match wins; ties broken by most recently modified."""
    hits = sorted(
        root.rglob(name),
        key=lambda p: (len(p.relative_to(root).parts), -p.stat().st_mtime),
    )
    return hits[0] if hits else None


# ---------------------------------------------------------------- rendering


def render_topic_info(path: Path, max_cell: int, max_rows: int) -> str:
    """Flatten topic_info.csv to compact text, one topic per block."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    blocks, dropped = [], 0
    for i, row in enumerate(rows):
        if i >= max_rows:
            dropped = len(rows) - max_rows
            break
        fields = []
        for key, val in row.items():
            val = "" if val is None else " ".join(str(val).split())
            if len(val) > max_cell:
                val = val[:max_cell] + " [...]"
            fields.append(f"  {key}: {val}")
        blocks.append("\n".join(fields))

    out = "\n\n".join(blocks)
    if dropped:
        out += f"\n\n[{dropped} further rows omitted to fit the context budget]"
    return out


def split_sections(md: str) -> list[str]:
    """Split markdown on level-2 headings so truncation drops whole topics."""
    sections: list[list[str]] = []
    current: list[str] = []
    for line in md.splitlines(keepends=True):
        if line.startswith("## ") and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)
    return ["".join(s) for s in sections]


def render_report(path: Path, budget: int) -> str:
    """Keep as many whole sections as fit in `budget` characters."""
    md = path.read_text(encoding="utf-8")
    if len(md) <= budget:
        return md

    kept, used, dropped = [], 0, 0
    for section in split_sections(md):
        if used + len(section) <= budget:
            kept.append(section)
            used += len(section)
        else:
            dropped += 1
    if not kept:  # one giant section, no headings to cut on
        return md[:budget] + "\n\n[report truncated]"
    return "".join(kept) + f"\n\n[{dropped} further sections omitted to fit the context budget]\n"


def build_user_message(prompt: str, csv_path: Path, csv_text: str,
                       md_path: Path, md_text: str) -> str:
    return (
        f"{prompt}\n\n"
        f"===== FILE: {csv_path.name} (topic table) =====\n{csv_text}\n\n"
        f"===== FILE: {md_path.name} (validation report) =====\n{md_text}\n"
        f"===== END OF DATA =====\n"
    )


# ---------------------------------------------------------------- LM Studio


def list_models(base_url: str, timeout: float) -> list[str]:
    r = requests.get(f"{base_url}/models", timeout=timeout)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


def resolve_model(base_url: str, requested: str | None, timeout: float) -> str:
    if requested:
        return requested
    models = list_models(base_url, timeout)
    if not models:
        sys.exit("LM Studio reports no loaded model. Load one, or pass --model.")
    return models[0]


def stream_chat(base_url: str, payload: dict, timeout: float):
    r = requests.post(f"{base_url}/chat/completions", json=payload,
                      stream=True, timeout=timeout)
    r.raise_for_status()
    for raw in r.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        delta = chunk["choices"][0].get("delta", {}).get("content")
        if delta:
            yield delta


def blocking_chat(base_url: str, payload: dict, timeout: float) -> str:
    r = requests.post(f"{base_url}/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", nargs="?", type=Path,
                   help="Directory containing the BERTopic outputs")
    p.add_argument("--prompt", help="Instruction sent with the data")
    p.add_argument("--prompt-file", type=Path, help="Read the instruction from a file")
    p.add_argument("--system", default=DEFAULT_SYSTEM, help="System message")
    p.add_argument("--csv-name", default="topic_info.csv")
    p.add_argument("--report-name", default="VALIDATION_REPORT.md")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", help="Model id; defaults to whatever LM Studio has loaded")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--budget-chars", type=int, default=24000,
                   help="Character budget for the two files (~4 chars/token)")
    p.add_argument("--csv-share", type=float, default=0.4,
                   help="Fraction of the budget reserved for topic_info.csv")
    p.add_argument("--max-cell", type=int, default=400, help="Max chars per CSV cell")
    p.add_argument("--max-rows", type=int, default=200, help="Max CSV rows to include")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the assembled prompt and exit")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--out", type=Path,
                   help="Where to save the transcript (default: <results_dir>/LLM_REVIEW_<ts>.md)")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_models:
        try:
            for m in list_models(args.base_url, args.timeout):
                print(m)
        except requests.exceptions.ConnectionError:
            sys.exit(f"No server at {args.base_url}. Start it with: lms server start")
        return

    if args.results_dir is None:
        sys.exit("results_dir is required (or use --list-models)")
    if not args.results_dir.is_dir():
        sys.exit(f"Not a directory: {args.results_dir}")
    if args.prompt and args.prompt_file:
        sys.exit("Use --prompt or --prompt-file, not both")

    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    if not prompt:
        prompt = DEFAULT_PROMPT

    csv_path = find_file(args.results_dir, args.csv_name)
    md_path = find_file(args.results_dir, args.report_name)
    missing = [n for n, p in ((args.csv_name, csv_path), (args.report_name, md_path)) if p is None]
    if missing:
        sys.exit(f"Not found under {args.results_dir}: {', '.join(missing)}")

    print(f"csv    : {csv_path}", file=sys.stderr)
    print(f"report : {md_path}", file=sys.stderr)

    csv_budget = int(args.budget_chars * args.csv_share)
    csv_text = render_topic_info(csv_path, args.max_cell, args.max_rows)
    if len(csv_text) > csv_budget:
        csv_text = csv_text[:csv_budget] + "\n\n[topic table truncated]"
    md_text = render_report(md_path, args.budget_chars - len(csv_text))

    user_msg = build_user_message(prompt, csv_path, csv_text, md_path, md_text)
    approx_tokens = len(user_msg) // 4
    print(f"prompt : {len(user_msg)} chars (~{approx_tokens} tokens)", file=sys.stderr)

    if args.dry_run:
        print(user_msg)
        return

    try:
        model = resolve_model(args.base_url, args.model, args.timeout)
    except requests.exceptions.ConnectionError:
        sys.exit(f"No server at {args.base_url}. Start it with: lms server start")
    print(f"model  : {model}\n", file=sys.stderr)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": args.system},
            {"role": "user", "content": user_msg},
        ],
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "stream": not args.no_stream,
    }

    try:
        if args.no_stream:
            answer = blocking_chat(args.base_url, payload, args.timeout)
            print(answer)
        else:
            parts = []
            for piece in stream_chat(args.base_url, payload, args.timeout):
                print(piece, end="", flush=True)
                parts.append(piece)
            print()
            answer = "".join(parts)
    except requests.exceptions.ConnectionError:
        sys.exit(f"\nNo server at {args.base_url}. Start it with: lms server start")
    except requests.exceptions.HTTPError as exc:
        sys.exit(f"\nLM Studio returned an error: {exc}\n{exc.response.text[:500]}")

    if args.no_save:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or args.results_dir / f"LLM_REVIEW_{stamp}.md"
    out.write_text(
        f"# LLM review\n\n"
        f"- generated: {datetime.now().isoformat(timespec='seconds')}\n"
        f"- model: `{model}`\n"
        f"- endpoint: {args.base_url}\n"
        f"- temperature: {args.temperature}, seed: {args.seed}, max_tokens: {args.max_tokens}\n"
        f"- topic table: `{csv_path}`\n"
        f"- validation report: `{md_path}`\n"
        f"- prompt chars: {len(user_msg)}\n\n"
        f"## Instruction\n\n```\n{prompt.strip()}\n```\n\n"
        f"## System message\n\n```\n{args.system.strip()}\n```\n\n"
        f"## Response\n\n{answer.strip()}\n",
        encoding="utf-8",
    )
    print(f"\nsaved  : {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
