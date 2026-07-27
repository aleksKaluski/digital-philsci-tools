"""
This script connects the text results retrived by BERTopic with the metadata connected with the text.
"""

import os
from pathlib import Path
import hashlib
import json
import uuid

print(f"Current path {os.getcwd()}")


def compute_hash(text: str) -> str:
    """
    Compute SHA-256 hash of normalized text.
    """
    normalized = text.strip().lower().replace('\n', ' ').replace('\r', ' ').encode('utf-8')
    return hashlib.sha256(normalized).hexdigest()


def build_text_hashmap(input_file: str | Path, field: str = "text") -> dict:
    """
    Build a hashmap based on a chosen field in a JSON file.
    """
    hashmap = {}
    input_file = Path(input_file)

    assert input_file.is_file(), "You passed sth that is not a file!"
    with open(input_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        if not isinstance(metadata, list):
            metadata = [metadata]

        for meta in metadata:
            text = meta.get(field, '')
            if text:
                text_hash = compute_hash(text)
                if text_hash not in hashmap:
                    hashmap[text_hash] = []
                hashmap[text_hash].append(meta)
    return hashmap



# 1. Compute hashmap for the source file with metadata
input_file = Path(r'files/operational_files/results_with_text.json')
results_hashmap = build_text_hashmap(input_file=input_file)

# pass the folder
folder_path = Path("BERTopic_results/raw/raw_bba")

for file_path in folder_path.iterdir():
    # iterate through files
    if file_path.is_file():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            for paragraph in data.get("paragraphs", []):
                para_text = paragraph['text']
                new_hash = compute_hash(para_text)

                # DEBUG: Show what we're hashing
                print(f"Paragraph text: {repr(para_text[:100])}...")  # repr() shows hidden chars
                print(f"Computed hash: {new_hash}")

                matched_metadata = results_hashmap.get(new_hash)
                print(f"Matched metadata: {matched_metadata}")
                print("---")