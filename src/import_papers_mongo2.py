"""
Modularization of original import_mongo_papers.py
"""

import json

from pymongo.errors import ConnectionFailure
from tqdm import tqdm
import os
import gzip
from pymongo import MongoClient, InsertOne
import time

def connect_mongo_db(mongo_ip: str = "localhost", mongo_port: int = 27017):
    try:
        client = MongoClient(mongo_ip, mongo_port)
    except ConnectionFailure:
        print("Could not connect to MongoDB!")
    except Exception as e:
        print("Could not connect to MongoDB! Error: ", e)

    db = client.papers_db
    collection = db.papers