"""
This script creates a mock corpus from a corpus file to work with this framework on MongoDB.
"""


import json
from pathlib import Path
from pymongo import MongoClient

# set the addresses
MONGO_HOST = "localhost"
MONGO_PORT = 27017

DB_NAME = "micro_subcorpus"
COLLECTION_NAME = "papers"

# path to your mock datset
CORPUS_PATH = Path("mock_testing/data/micro_s2orc.jsonl")


def main() -> None:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Missing corpus file: {CORPUS_PATH}."
        )

    client = MongoClient(MONGO_HOST, MONGO_PORT)

    db = client[DB_NAME]
    col = db[COLLECTION_NAME]

    # clean corpus
    col.delete_many({})

    # inset the files to the corpus
    inserted = 0
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}\nLine: {line}") from e

            # validate
            if "corpusid" not in doc:
                raise ValueError(
                    f"Line {line_no} is missing required keys. "
                    f"Required: corpusid, text. Got keys: {list(doc.keys())}"
                )

            col.insert_one(doc)
            inserted += 1

    print(f"Inserted {inserted} docs into {DB_NAME}.{COLLECTION_NAME}")

    # count
    count = col.count_documents({})
    print(f"Mongo now contains {count} docs.")

"""
After running the script, you may verify if it's working well by communicating with MongoDB directly:
python -c "from pymongo import MongoClient; c=MongoClient('localhost',27017); print(c['mock_philsci']['papers'].count_documents({}))"
"""

if __name__ == "__main__":
    main()