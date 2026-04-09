"""
Future developement: potential new tests for MongoDB.
"""


import json
import pytest
import gzip
import uuid
from pymongo import MongoClient


@pytest.fixture(scope="session")
def mongo_uri():
    return "mongodb://localhost:27017"


@pytest.fixture(scope="session")
def mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    yield client
    client.close()


@pytest.fixture
def mongo_db(mongo_client):
    # unique DB name per test
    db_name = f"test_db_{uuid.uuid4().hex}"
    db = mongo_client[db_name]
    yield db
    # drop the DB after test
    mongo_client.drop_database(db_name)


@pytest.fixture
def mock_json_docs():
    json_path = r'data/mongo_mock_structure.json'
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@pytest.fixture
def mock_jsonl_gz_path(tmp_path, mock_json_docs):
    # create temporary path for json.gz
    file_path = tmp_path / "papers.json.gz"

    # create mock json.gz
    with gzip.open(file_path, "wt", encoding="utf-8") as f:
        for entry in mock_json_docs:
            json_record = json.dumps(entry)
            f.write(json_record + "\n")
    return file_path


def test_connection(mongo_client):
    """
    Test if you can connect to mongodb.
    """
    result = mongo_client.admin.command("ping")
    assert result["ok"] == 1.0
