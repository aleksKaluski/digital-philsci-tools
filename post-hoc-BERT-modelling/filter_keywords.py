"""
This script enables you filter out paragraphs with a certian set of keywords before BERTopic analysis.
"""

import json
import re
import sys

import os
from pathlib import Path

def find_repo_root(start_path=None):
    if start_path is None:
        start_path = Path(__file__).resolve().parent
    else:
        start_path = Path(start_path).resolve()

    for parent in [start_path] + list(start_path.parents):
        if (parent / '.git').exists() or (parent / '.git').is_dir():
            return parent
    return start_path

# change to repository root - works from any subfolder
os.chdir(find_repo_root())

# words/forms to match (case-insensitive, whole-word)
KEYWORDS = [
    "direct perception"
]

PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def keep_record(record, field="text"):
    value = record.get(field, "")
    return bool(PATTERN.search(value))


def filter_json(input_path, output_path, field="text"):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = [rec for rec in data if keep_record(rec, field)]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Kept {len(filtered)} of {len(data)} records.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)

    filter_json(sys.argv[1], sys.argv[2])

"""
Use-case:
 python filter_keywords.py files/operatonal_files/final_dp/results_with_text.json files/operatonal_files/final_dp/results_with_text_dp.json

"""