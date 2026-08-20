import json
import re
import sys

# Words/forms to match (case-insensitive, whole-word)
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
        print("Usage: python filter_direct_perception files/operatonal_files/final_dp/results_with_text.json files/operatonal_files/final_dp/results_with_text_dp.json")
        sys.exit(1)

    filter_json(sys.argv[1], sys.argv[2])