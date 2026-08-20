"""
This script connects the text results retrived by BERTopic with the metadata connected with the text.
"""

import os
from pathlib import Path
import hashlib
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description='Enrich BERTopic modeling results by matching it with metadata.')
    parser.add_argument(
        '-file', '--input-file',
        type=str,
        required=True,
        help='Path to the input JSON file containing metadata'
    )

    parser.add_argument(
        '-dir', '--input-folder',
        type=str,
        required=True,
        help='Path to the folder with the results of run_topic_modelling.py'
    )

    parser.add_argument(
        '-field', '--text-field',
        type=str,
        required=True,
        help='Text field of the JSON file.'
    )

    args = parser.parse_args()

    print(f"\nCurrent path: {os.getcwd()}")
    print(f"Input file: {args.input_file}")
    print(f"Input folder: {args.input_folder}")
    print(f"Text field: {args.text_field}\n")


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

    # compute hashmap for the source file with metadata
    input_file = Path(args.input_file)
    results_hashmap = build_text_hashmap(input_file=input_file, field=str(args.text_field))

    # pass the input folder
    folder_path = Path(args.input_folder)

    # create an output folder for processed files
    new_folder_path = r"BERTopic_results/processed/" + folder_path.name.split("_")[0] + "_processed"
    new_folder_path = Path(new_folder_path)
    new_folder_path.mkdir(parents=True, exist_ok=True)

    for file_path in folder_path.iterdir():
        # iterate through files
        if file_path.is_file():
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                for i, paragraph in enumerate(data.get("paragraphs", [])):
                    para_text = paragraph[str(args.text_field)]

                    # compute hashmap representation
                    new_hash = compute_hash(para_text)

                    # match metadata based on hash
                    matched_metadata = results_hashmap.get(new_hash)

                    if matched_metadata:
                        # avoid doubling identical data
                        # take the first metadata item and convert to dict
                        meta = matched_metadata[0]  # assuming there's at least one item
                        meta.pop("text", None)
                        meta["rrf_rank"] = meta.pop("rank", None)
                        paragraph["metadata"] = meta  # assign the dict directly
                    else:
                        paragraph["metadata"] = {}  # empty dict instead of empty list
                        print(f"[!] No metadata for {file_path.name} (paragraph {i})")


            output_file_path = new_folder_path.joinpath(file_path.name.replace("_raw.json", "_processed.json"))
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Results were saved to {new_folder_path}")


if __name__ == "__main__":
    main()

"""
Use-case:

python enrich_tm_with_metadata.py \
    -file files/operational_files/results_with_text.json\
    -dir BERTopic_results/raw/e80_raw\
    -field text

"""

