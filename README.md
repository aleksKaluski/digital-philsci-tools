# Digital Philosophical Tools
This repository plays with the code designed for Wiktor Rorot's PhD, _Scale-Free Communication? An investigation of the use of the concept of "communication" in biology and cognitive sciences._ The code is a text processing pipeline based on BERTopic, enriched with the integration of external tools such as MongoDB and Milvus. However, in order to play with it, we need to create mock instances of these objects with Docker. Here is how to do it:
## Docker
1. Start the container: `docker-compose up -d`
2. Check if it's running: `docker-compose ps`
3. Verify that the MongoDB container works: \
`python -c "from pymongo import MongoClient; client = MongoClient('localhost', 27017); client.server_info(); print('MongoDB connected!')"`

4. Make a simple test of MongoDB functionalities to verify the container. Populate it with a mock dataset of neopositivistic papers and test it! \
`python mock_testing\mongo_simple_test.py`

You should see: \
[+] Inserted 3 papers. \
[+] Created corpusid index \
[+] Found paper: Verificationism Then and Now \
[+] Total papers in collection: 36 \
[SUCCESS] Total runtime: 0.02 seconds

To check clean the database and check it once again use: \
`python -c "from pymongo import MongoClient; client = MongoClient('localhost', 27017); client.drop_database('papers_db'); print('Database dropped')"`

