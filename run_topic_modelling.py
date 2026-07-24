from topic_modeling_analysis import (DataLoader, TopicModeler, ModelConfig)
import os
from pathlib import Path
import json
import uuid


print(f"Current path {os.getcwd()}")

# load paragraph results
loader = DataLoader('files/operational_files/results_with_text.json',
                    text_mode='result-with-context')


paragraphs, metadata = loader.load()

# configure for paragraph-level clustering
config = ModelConfig(
    query_level='paragraph',
    text_mode='result',
    hdbscan_min_cluster_size=15,
    umap_n_components=5,
    use_gpu=True
)

# run BERTopic and retrieve raw results
modeler = TopicModeler(config)
model, topics, probs = modeler.fit(paragraphs)

output_dir = Path(f"BERTopic_results/raw/raw_{str(uuid.uuid4())[:3]}")
output_dir.mkdir(parents=True, exist_ok=True)

# the most representative paragraph for each cluster
for topic_id in model.get_topics():
    center_paragraphs = model.get_representative_docs(topic_id)[:3]

    # save to file
    with open(output_dir / f'center_{topic_id}.json', 'w', encoding='utf-8') as f:

        data = {"topic": topic_id,
                "paragraphs": []}

        for i, p in enumerate(center_paragraphs):
            data = {
                "topic": topic_id,
                "paragraphs": [
                    {"number": i, "text": p}
                    for i, p in enumerate(center_paragraphs)
                ]
            }
        json.dump(data, f, indent=4, ensure_ascii=False)









