# Digital Philosophical Tools
This repository plays with the code designed for Wiktor Rorot's PhD, _Scale-Free Communication? An investigation of the use of the concept of "communication" in biology and cognitive sciences._ The code is a text processing pipeline based on BERTopic, enriched with the integration of external tools such as MongoDB and Milvus. However, in order to play with it, we need to create mock instances of these objects with Docker. Here is how to do it:

## Complex way of setting the environment
First, make sure that you are using Python 3.11 by using:
`python --version`. After that we may activate the environment by using the following commands:
1. `py -3.11 -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1`
3. `python --version`\
After than you should upgrade pip and install setuptool. 
4. `python -m pip install --upgrade pip wheel`
5. `pip install "setuptools<82"`\
Finally, install requirements. 
6. `python -m pip install -r requirements.txt`\
Then, make the sanity check.
7. `python -c "import sys; print(sys.executable)"`
8. `python -c "import pymongo; print('pymongo ok')"`
9. `python -c "import pymilvus; print('pymilvus ok')"`

## Docker for MongoDB
1. Start the container: `docker-compose up -d` (remember to start Docker Desktop ;))
2. Check if it's running: `docker-compose ps`
3. Verify that the MongoDB container works: \
`python -c "from pymongo import MongoClient; client = MongoClient('localhost', 27017); client.server_info(); print('MongoDB connected!')"`
4. Verify that Milvus conteiner works: \
`python -c "from pymilvus import connections; connections.connect(host='localhost', port='19530'); print('Milvus connected!')"`
5. Make a simple test of MongoDB functionalities to verify the container. Populate it with a mock dataset of neopositivistic papers and test it! \
`python mock_testing\mongo_simple_test.py` \
You should see: 
```
[+] Inserted 3 papers. 
[+] Created corpusid index 
[+] Found paper: Verificationism Then and Now 
[+] Total papers in collection: 36 
[SUCCESS] Total runtime: 0.02 seconds
```

6. Make a simple test of Milvus functionalities to verify the containter. \
`python mock_testing/milvus_simple_test.py`

You should see: 
```
[+] Inserted 3 papers.
[+] Created vector index on 'embedding'.
[+] Found nearest paper: Verificationism Then and Now (distance=0.0000)
[+] Total papers in collection: 3
[SUCCESS] Total runtime: 6.83 seconds
```

To clean the database and check it once again use: \
`python -c "from pymongo import MongoClient; client = MongoClient('localhost', 27017); client.drop_database('papers_db'); print('Database dropped')"`\

`python -c "from pymilvus import connections, utility; connections.connect(host='localhost', port='19530'); utility.drop_collection('neopositivistic_papers'); print('Collection dropped')"`

If you would like to check the connection, you can use:\
`Test-NetConnection -ComputerName localhost -Port 19530`\
`Test-NetConnection -ComputerName localhost -Port 27017`