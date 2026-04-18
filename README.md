# Digital Philosophical Tools
This repository plays with the code designed for Wiktor Rorot's PhD, _Scale-Free Communication? An investigation of the use
of the concept of "communication" in biology and cognitive sciences._ The code is a text processing pipeline based on BERTopic,
enriched with the integration of external tools such as MongoDB and Milvus. However, in order to play with it, we need
to create mock instances of these objects with Docker. Here is how to do it:

## Create Initial Python Environment
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

## Creating BERT Environment
First, you have to have conda installed (verify by using `conda --version`). Then, follow the steps:
1. Create brand-new special folder for keeping you bertopic environment.\
`mkdir C:\Users\your_name\conda_envs`
2. Create the environment:\
`conda env create -f environment_bertopic.yml --prefix C:\Users\your_name\conda_envs\bertopic`\
It should take 5-15 mins.\
At this point you face a danger of mixing your conda environment
with your normal `(.venv)` environment. If you see sth like that:  `(.venv) PS C:\Python_files\digital-philsci-tools`
you should avoid mixing by deactivating the `(.venv)`. Use `deactivate` and then
`conda activate C:\Users\your_name\conda_envs\bertopic`. After that I advise to verify this step by using
`python -c "from topic_modeling_analysis import *; print('✅ Success')"`                                                                                 

## Last Steps
All right! If you read this, you are a very brave person! You have established your environment, and now you can 
run your analysis on a mock corpus and test the actual program. It was hard, wasn't it? :) 

Now, let's create mock objects! As you recall, we deactivated `(.venv)` for the sake of clarity. Now it's time
to bring it back.
1. Deactivate conda: `conda deactivate`
2. Activate `(.venv)`: `.\.venv\Scripts\Activate.ps1`
3. Verify. You should se Python 3.11: `python --version`

Finally, we can create mock database in MongoDB:\
`python mock_testing\create_mock_mongo_corpus.py`

Then, create Milvus database:\
`python mock_testing\create_mock_milvus_corpus.py` and then run `python -c "from pymilvus import connections, db; connections.connect(host='localhost', port=19530); print('dbs:', db.list_database())"`
to be sure that Milvus includes **mock_philsci**

## Running a Complete Analysis on Mock Database
To intialize Milvus corpus, run: 
```pycon
python build_subcorpus_milvus.py 
  --subcorpus mock_testing/data/micro_subcorpus.pkl 
  --s2orc-path . 
  --db-name micro_subcorpus 
  --sentence-collection sentences 
  --paragraph-collection paragraphs 
  --model multi-qa-MiniLM-L6-cos-v1 
  --no-gpu 
  --mongo-db-name mock_philsci 
  --mongo-collection-name papers
```
To check if you have the right amount of papers loaded use:

```pycon
python -c "from pymilvus import connections, db, MilvusClient; connections.connect(alias='default', host='localhost', port='19530'); db.using_database('micro_subcorpus'); c=MilvusClient(uri='http://localhost:19530', token='root:Milvus', db_name='micro_subcorpus'); print('sentences', c.get_collection_stats('sentences')); print('paragraphs', c.get_collection_stats('paragraphs'))"  
```

It might be the case that you resumed the work from the checkpoint and then accidentally didn't load the papers in place.
In this case delete the checkpoint `del subcorpus_checkpoint.pkl -ErrorAction SilentlyContinue` and run drop empty collections

```pycon
python -c "from pymilvus import connections, db, MilvusClient; connections.connect(alias='default', host='localhost', port='19530'); db.using_database('micro_subcorpus'); c=MilvusClient(uri='http://localhost:19530', token='root:Milvus', db_name='micro_subcorpus'); \
[print('dropping', name) or c.drop_collection(name) for name in ['sentences','paragraphs'] if c.has_collection(name)]; print('done')"
```

And re-run making corpus from skratch. 

```pycon
python query_subcorpus.py `
  --db-name micro_subcorpus `
  --collection paragraphs `
  --queries queries.txt `
  --output micro_query_results.json `
  --limit 50 `
  --model multi-qa-MiniLM-L6-cos-v1 `
  --no-gpu
```
