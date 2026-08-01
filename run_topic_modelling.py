"""
This scripts runs BERTopic modeling on paragraph, returning JSON files with the
results - number of observations close to the center of space.
"""

from topic_modeling_analysis import (DataLoader, TopicModeler, ModelConfig)
import os
from pathlib import Path
import json
import uuid
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='Run BERTopic modeling on paragraphs or sentences.')

    parser.add_argument(
        '-i', '--input-file',
        type=str,
        required=True,
        help='Path to the input JSON file containing paragraphs and metadata'
    )

    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=15,
        help='Minimum cluster size for HDBSCAN (default: 15)'
    )

    parser.add_argument(
        '--umap-components',
        type=int,
        default=5,
        help='Number of UMAP components (default: 5)'
    )

    parser.add_argument(
        '--docs',
        type=int,
        default=3,
        help='Number of saved docs (default: 3)'
    )

    parser.add_argument(
        '--use-gpu',
        action='store_true',
        default=True,
        help='Use GPU for computation (default: True)'
    )


    args = parser.parse_args()

    print(f"Current path: {os.getcwd()}")
    print(f"Input file: {args.input_file}")
    print(f"Min cluster size: {args.min_cluster_size}")
    print(f"UMAP components: {args.umap_components}")
    print(f"Use GPU: {args.use_gpu}\n")

    # load paragraph results
    try:
        loader = DataLoader(args.input_file)

        paragraphs, metadata = loader.load()
        # text_mode='result-with-context') - hash maps work well only without the context!
        print(f"Loaded {len(paragraphs)} paragraphs with metadata")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


    # configure for paragraph-level clustering
    config = ModelConfig(
        query_level='paragraph',
        text_mode='result',
        hdbscan_min_cluster_size=args.min_cluster_size,
        umap_n_components=args.umap_components,
        use_gpu=args.use_gpu
    )

    # run BERTopic and retrieve raw results
    modeler = TopicModeler(config)
    model, topics, probs = modeler.fit(paragraphs)

    output_dir = Path(f"BERTopic_results/raw/{str(uuid.uuid4())[:3]}_raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    # the most representative paragraph for each cluster
    for topic_id in model.get_topics():
        center_paragraphs = model.get_representative_docs(topic_id)[:args.docs]

        # create the data structure
        data = {
            "topic": topic_id,
            "paragraphs": [
                {"bert_position": i, "text": p}
                for i, p in enumerate(center_paragraphs)
            ]
        }

        output_file = output_dir / f'center_{topic_id}_processed.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()

"""
Sample run:
python run_topic_modelling.py\
    -i files/operational_files/results_with_text.json\
    --min-cluster-size 20\
    --umap-components 8\
    --docs 3\
    --use-gpu

"""









