from time import time
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from colorama import Fore, Style

HOST = "localhost"
PORT = "19530"
COLLECTION_NAME = "neopositivistic_papers"

def main():
    t0 = time()

    connections.connect(host=HOST, port=PORT)

    # Drop existing collection for clean test
    if utility.has_collection(COLLECTION_NAME):
        Collection(COLLECTION_NAME).drop()

    # Define schema
    fields = [
        FieldSchema(name="paper_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4),
    ]
    schema = CollectionSchema(fields, description="Mock neopositivistic papers")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # Mock data
    papers = [
        (1, "Verificationism Then and Now", [0.11, 0.22, 0.33, 0.44]),
        (2, "Logical Positivism and Its Critics", [0.10, 0.21, 0.31, 0.41]),
        (3, "Ayer's Emotivism Revisited", [0.12, 0.25, 0.35, 0.48]),
    ]

    ids, titles, vectors = zip(*papers)

    print('\n' + Fore.BLUE + Style.BRIGHT + '=' * 60)
    print(Fore.YELLOW + Style.BRIGHT + f"{'Testing Milvus':^60}")
    print(Fore.BLUE + Style.BRIGHT + '-' * 60)

    # Insert once
    collection.insert([ids, titles, vectors])
    print(Fore.GREEN + "[+] " + Fore.RESET + f"Inserted {len(papers)} papers.")

    # Flush immediately so count is accurate
    collection.flush()

    # Create index
    index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 16}}
    collection.create_index(field_name="embedding", index_params=index_params)
    print(Fore.GREEN + "[+] " + Fore.RESET + f"Created vector index on 'embedding'.")

    # Load collection to memory
    collection.load()

    # Search
    query_vec = [[0.11, 0.22, 0.33, 0.44]]
    results = collection.search(
        data=query_vec,
        anns_field="embedding",
        param={"nprobe": 8},
        limit=1,
        output_fields=["title"]
    )

    hit = results[0][0]
    print(Fore.GREEN + "[+] " + Fore.RESET + f"Found nearest paper: {hit.entity.get('title')} (distance={hit.distance:.4f})")

    # Count
    count = collection.num_entities
    print(Fore.GREEN + "[+] " + Fore.RESET + f"Total papers in collection: {count}")

    print(Fore.GREEN + Style.BRIGHT + f"[SUCCESS] Total runtime: {time() - t0:.2f} seconds" + Style.RESET_ALL)
    print(Fore.BLUE + Style.BRIGHT + '=' * 60 + Style.RESET_ALL)

if __name__ == "__main__":
    main()