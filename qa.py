import time

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

DATASET_NAME = 'squad_v2'  # Huggingface Dataset to use
MODEL = 'distilbert-base-uncased'  # Transformer to use for embeddings
TOKENIZATION_BATCH_SIZE = 1000  # Batch size for tokenizing operation
INFERENCE_BATCH_SIZE = 64  # batch size for transformer
INSERT_RATIO = .1  # How many titles to embed and insert
COLLECTION_NAME = 'huggingface_squad_db'  # Collection name
DIMENSION = 768  # Embeddings size
LIMIT = 10  # How many results to search for
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
REPLICA_NUMBER = 1

connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
if utility.has_collection(COLLECTION_NAME):
    qa_collection = Collection(COLLECTION_NAME)
    qa_collection.load(replica_number=REPLICA_NUMBER)
else:
    raise RuntimeError

qa_collection.release()