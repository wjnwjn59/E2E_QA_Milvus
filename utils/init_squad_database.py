import argparse
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from datasets import load_dataset_builder, load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel
from torch import clamp, sum

DATASET_NAME = 'squad_v2'  # Huggingface Dataset to use
MODEL_NAME = 'distilbert-base-uncased'  # Transformer to use for embeddings
TOKENIZATION_BATCH_SIZE = 1000  # Batch size for tokenizing operation
INFERENCE_BATCH_SIZE = 64  # batch size for transformer
INSERT_RATIO = .1  # How many titles to embed and insert
COLLECTION_NAME = 'huggingface_squad_db'  # Collection name
DIMENSION = 768  # Embeddings size
LIMIT = 3  # How many results to search for
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
REPLICA_NUMBER = 1

def create_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name='question', dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name='context', dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name='answer', dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name='question_embedding', dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='question search')
    collection = Collection(name=collection_name, schema=schema)

    # create IVF_FLAT index for collection.
    index_params = {
        'metric_type':'L2',
        'index_type':"IVF_FLAT",
        'params':{"nlist":2048}
    }
    collection.create_index(field_name="question_embedding", index_params=index_params)

    return collection

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize_question(batch):
    results = tokenizer(
        batch['question'], 
        add_special_tokens=True, 
        truncation=True, 
        padding="max_length", 
        return_attention_mask=True, 
        return_tensors="pt"
    )

    batch['input_ids'] = results['input_ids']
    batch['attention_mask'] = results['attention_mask']

    return batch

# Embed the tokenized question and take the mean pool with respect to attention mask of hidden layer.
model = AutoModel.from_pretrained(MODEL_NAME)
def quest_embedding(batch):
    sentence_embs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask']
    )[0]
    input_mask_expanded = batch['attention_mask'].unsqueeze(-1).expand(sentence_embs.size()).float()
    batch['question_embedding'] = sum(sentence_embs * input_mask_expanded, 1) / clamp(input_mask_expanded.sum(1), min=1e-9)

    return batch

def create_squad_database(qa_collection):
    squad_v2_dataset = load_dataset(DATASET_NAME, split='all')
    squad_v2_dataset = squad_v2_dataset.train_test_split(test_size=INSERT_RATIO, seed=42)['test']
    squad_v2_dataset = squad_v2_dataset.map(lambda val: {'answer': val['answers']['text'][0]} if val['answers']['text'] else {'answer': ''}, remove_columns=['answers'])

    # Generate the tokens for each entry.
    squad_v2_dataset = squad_v2_dataset.map(tokenize_question, batch_size=TOKENIZATION_BATCH_SIZE, batched=True)
    squad_v2_dataset.set_format('torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    squad_v2_dataset = squad_v2_dataset.map(
        quest_embedding, 
        remove_columns=['input_ids', 'attention_mask'], 
        batched=True, 
        batch_size=INFERENCE_BATCH_SIZE
    )

    # Due to the varchar constraint we are going to limit the question size when inserting
    def insert_function(batch):
        insertable = [
            batch['title'],    
            batch['question'],
            [x[:9995] + '...' if len(x) > 9999 else x for x in batch['context']],
            [x[:995] + '...' if len(x) > 999 else x for x in batch['answer']],
            batch['question_embedding'].tolist()
        ]    
        qa_collection.insert(insertable)

    squad_v2_dataset.map(insert_function, batched=True, batch_size=64)
    qa_collection.flush()

connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
if not utility.has_collection(COLLECTION_NAME):
    qa_collection = create_collection(COLLECTION_NAME, DIMENSION)
    qa_collection.load(replica_number=REPLICA_NUMBER)
else:
    qa_collection = Collection(COLLECTION_NAME)
    qa_collection.load(replica_number=REPLICA_NUMBER)

if qa_collection.is_empty:
    create_squad_database(qa_collection)

def search(question_batch):
    res = qa_collection.search(
        question_batch['question_embedding'].tolist(), 
        anns_field='question_embedding', 
        param={
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }, 
        output_fields=['question', 'context'], 
        limit=LIMIT
    )
    overall_id = []
    overall_distance = []
    overall_question = []
    overall_context = []

    for hits in res:
        ids = []
        distances = []
        questions = []
        contexts = []

        for hit in hits:
            ids.append(hit.id)
            distances.append(hit.distance)
            questions.append(hit.entity.get('question'))
            contexts.append(hit.entity.get('context'))

        overall_id.append(ids)
        overall_distance.append(distances)
        overall_question.append(questions)
        overall_context.append(contexts)

    return {
        'id': overall_id,
        'distance': overall_distance,
        'context': overall_context,
        'similar_question': overall_question
    }

def main():
    questions = {'question': ['When was chemistry invented?', 'When was Eisenhower born?']}
    question_dataset = Dataset.from_dict(questions)

    question_dataset = question_dataset.map(
        tokenize_question, 
        batched=True, 
        batch_size=TOKENIZATION_BATCH_SIZE
    )
    question_dataset.set_format(
        'torch', 
        columns=['input_ids', 'attention_mask'], 
        output_all_columns=True
    )
    question_dataset = question_dataset.map(
        quest_embedding, 
        remove_columns=['input_ids', 'attention_mask'], 
        batched=True, 
        batch_size=INFERENCE_BATCH_SIZE
    )

    retrieval_results = question_dataset.map(search, batched=True, batch_size=1)
    for result in retrieval_results:
        print()
        print('Input Question:')
        print(result['question'])
        print()
        for rank_idx, candidate in enumerate(zip(result['similar_question'], result['context'], result['distance'])):
            similar_question = candidate[0]
            context = candidate[1]
            distance = candidate[2].tolist()

            print(f'Similar Question Rank {rank_idx+1}: {similar_question}')
            print(f'Context: {context}')
            print(f'Score: {distance}')
            print()

    
if __name__ == '__main__':
    main()
    qa_collection.release()

