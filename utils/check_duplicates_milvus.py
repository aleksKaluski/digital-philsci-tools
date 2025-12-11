# Using subcorpus file approach (faster for large collections)
import pickle
from pymilvus import MilvusClient
from collections import Counter
from tqdm import tqdm
import os, sys

print(os.getcwd())

with open('/home/wikror/gdrive/compositionality-data/20251208/subcorpus_20251208_163659.pkl', 'rb') as f:
    subcorpus_data = pickle.load(f)

corpus_ids = subcorpus_data['corpus_ids']

client = MilvusClient(host='localhost', port='19530', db_name='compositionality_subcorpus_specter')
client.load_collection(collection_name='sentences')

duplicates_found = []

for corpus_id in tqdm(corpus_ids):
    results = client.query(
        collection_name='sentences',
        output_fields=['corpusid', 'sentence_number'],
        filter=f'corpusid == {corpus_id}',
        limit=16384
    )
    
    # Check for duplicates within this corpus_id
    sentence_nums = [r['sentence_number'] for r in results]
    sentence_counts = Counter(sentence_nums)
    
    local_dups = {num: count for num, count in sentence_counts.items() if count > 1}
    if local_dups:
        duplicates_found.append((corpus_id, local_dups))

if duplicates_found:
    print(f"Found duplicates in {len(duplicates_found)} papers:")
    for corpus_id, dups in duplicates_found[:10]:
        print(f"  Corpus {corpus_id}: {dups}")
else:
    print("No duplicates found!")