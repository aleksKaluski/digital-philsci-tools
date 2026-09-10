"""
This scripts runs BERTopic modeling on paragraph, returning JSON files with the
results - number of observations close to the center of space.
"""

from pathlib import Path
import sys
import os

def find_repo_root(start_path=None):
    """
    Traverse up from the script's location until we find the .git directory
    (which marks the repository root). Works on any OS and any machine.
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent
    else:
        start_path = Path(start_path).resolve()

    for parent in [start_path] + list(start_path.parents):
        if (parent / '.git').exists() or (parent / '.git').is_dir():
            return parent
    return start_path

# get repo root and change to it
repo_root = find_repo_root()
os.chdir(repo_root)

# add repo root to Python path so imports work from any subfolder
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from topic_modeling_analysis import (DataLoader, TopicModeler, ModelConfig)
import json
import uuid
import argparse
import pandas as pd


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
        '--model',
        type=str,
        default='sentence-transformers/all-mpnet-base-v2',
        help='Embedding model name (default: sentence-transformers/all-mpnet-base-v2, '
             'to match build_subcorpus_milvus.py / query_subcorpus.py). '
             'ModelConfig otherwise silently falls back to all-MiniLM-L6-v2 (384-dim).'
    )

    parser.add_argument(
        '--use-gpu',
        dest='use_gpu',
        action='store_true',
        default=True,
        help='Use GPU for computation (default: True)'
    )

    parser.add_argument(
        '--no-gpu',
        dest='use_gpu',
        action='store_false',
        help='Disable GPU for computation'
    )


    args = parser.parse_args()

    print(f"\nCurrent path: {os.getcwd()}")
    print(f"Input file: {args.input_file}")
    print(f"Min cluster size: {args.min_cluster_size}")
    print(f"UMAP components: {args.umap_components}")
    print(f"Embedding model: {args.model}")
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
        embedding_model_name=args.model,
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

    # save topic sizes + keywords for downstream validation
    topic_info = model.get_topic_info()
    topic_info.to_csv(output_dir / "topic_info.csv", index=False)

    # Claude's fix on hardcoded BERTopic engine
    # Reuse BERTopic's own representative-doc machinery - the same c-TF-IDF
    # cosine-similarity ranking that feeds get_representative_docs() and the
    # built-in LLM labeling prompts - but with nr_repr_docs=args.docs instead
    # of BERTopic's hardcoded 3.
    documents_df = pd.DataFrame({
        "Document": paragraphs,
        "Topic": topics,
        "ID": range(len(paragraphs)),
        "Image": [None] * len(paragraphs),
    })
    repr_docs_mappings, _, _, _ = model._extract_representative_docs(
        model.c_tf_idf_,
        documents_df,
        model.get_topics(),
        500, # nr_samples: candidate pool per topic before ranking
        args.docs,  # nr_repr_docs: how many to keep after ranking
    )

    # the most representative paragraphs for each cluster
    for topic_id in model.get_topics():
        center_paragraphs = repr_docs_mappings[topic_id][:args.docs]

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
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()

"""
Sample run:
python run_topic_modelling.py \
    -i files/operational_files/results_with_text.json \
    --min-cluster-size 20 \
    --umap-components 8 \
    --docs 10 \
    --use-gpu
    
python run_topic_modelling.py -i files/operational_files/results_with_text.json --min-cluster-size 20 --umap-components 8 --docs 10 --use-gpu

"""