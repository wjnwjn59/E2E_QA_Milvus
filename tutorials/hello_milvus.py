import random
import time

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

start_time = time.time()

connections.connect("default", host="localhost", port="19530")
print(utility.has_collection("hello_milvus"))
fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="random", dtype=DataType.DOUBLE),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=8)
]
schema = CollectionSchema(fields, "hello_milvus is the simplest demo to introduce the APIs")
hello_milvus = Collection("hello_milvus", schema) # This is our database
print(utility.has_collection("hello_milvus"))

# Add 3k records to the database
entities = [
    [i for i in range(3000)],  # field pk
    [float(random.randrange(-20, -10)) for _ in range(3000)],  # field random
    [[random.random() for _ in range(8)] for _ in range(3000)],  # field embeddings
]
insert_result = hello_milvus.insert(entities)

# Build indexes to the entities
index = {
    "index_type": "IVF_FLAT", # Index mode (quantization + FLAT)
    "metric_type": "L2", # Similarity search metric
    "params": {"nlist": 128}, # ?
}
hello_milvus.create_index("embeddings", index)
hello_milvus.load()

vectors_to_search = entities[-1][-1:]
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}

top_k=2
print('I. Vector similarity search')
result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=top_k, output_fields=["pk", "random"])
for q_idx, q in enumerate(result):
    print(f"Query {q_idx + 1}")
    for idx in range(top_k):
        print(f'Candidate {idx + 1}: {q[idx]}')
    print('\n')

print('II. Vector query')
# Select all records that have "random" > -100
result = hello_milvus.query(expr="random > -100", output_fields=["random", "embeddings"])
print(f'Num matched: {len(result)}\n')

print('III. Hybrid search')
# Serach similar embeddings and have random value > -12
result = hello_milvus.search(vectors_to_search, "embeddings", search_params, limit=100, expr="random > -12", output_fields=["random"])
print(result, '\n')

# Delete entities by primary key
print('IV. Delete some entities')
print(f'Num entities before deletion: {hello_milvus.num_entities}')
expr = f"pk in [10, 100]"
hello_milvus.delete(expr)
print(f'Num entities after deletion: {hello_milvus.num_entities}\n')

# Drop the collections
print('V. Drop collections')
utility.drop_collection("hello_milvus")
print(f'Is hello_milvus exists: {utility.has_collection("hello_milvus")}\n')


end_time = time.time() - start_time
print(f'Time executed: {end_time}') 