from __future__ import annotations

from pymongo import MongoClient
from pymilvus import connections, db, MilvusClient, DataType
from sentence_transformers import SentenceTransformer


MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_TOKEN = "root:Milvus"

DB_NAME = "mock_philsci"
COLLECTION_NAME = "paragraphs"

MONGO_HOST = "localhost"
MONGO_PORT = 27017
MONGO_DB = "mock_philsci"
MONGO_COLLECTION = "papers"

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"


def recreate_db_and_collection(client: MilvusClient, dim: int) -> None:
    # Ensure DB exists and use it
    existing = db.list_database()
    if DB_NAME not in existing:
        db.create_database(DB_NAME)
    db.using_database(DB_NAME)

    # Drop collection if exists (so we can re-run safely)
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="corpusid", datatype=DataType.INT64)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

    idx_params = client.prepare_index_params()
    idx_params.add_index(field_name="corpusid", index_type="INVERTED")
    idx_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=idx_params,
    )


def main() -> None:
    # Connect Milvus
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    milvus = MilvusClient(
        uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}",
        token=MILVUS_TOKEN,
        db_name=DB_NAME,
    )

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"Loaded model {MODEL_NAME} (dim={dim})")

    recreate_db_and_collection(milvus, dim)

    # Read docs from Mongo
    mongo = MongoClient(MONGO_HOST, MONGO_PORT)
    docs = list(mongo[MONGO_DB][MONGO_COLLECTION].find({}, {"_id": 0}))
    docs.sort(key=lambda d: int(d["corpusid"]))
    print(f"Loaded {len(docs)} docs from Mongo")

    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)

    rows = []
    for d, emb in zip(docs, embeddings):
        rows.append(
            {
                "corpusid": int(d["corpusid"]),
                "vector": emb.tolist(),
            }
        )

    milvus.insert(collection_name=COLLECTION_NAME, data=rows)
    milvus.flush(collection_name=COLLECTION_NAME)

    # Quick sanity check
    stats = milvus.get_collection_stats(COLLECTION_NAME)
    print("Milvus collection stats:", stats)
    print(f"Inserted {len(rows)} vectors into {DB_NAME}.{COLLECTION_NAME}")


if __name__ == "__main__":
    main()