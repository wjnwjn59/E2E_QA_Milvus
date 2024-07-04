import json
import pandas as pd
import os

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data

def save_to_csv(records, save_path):
    df = pd.DataFrame(records)
    df.to_csv(save_path)

def load_csv(csv_path):
    df = pd.read_csv(csv_path)

    return df

def extract_qa_info(qa_dict, context_id):
    question = qa_dict['question']
    is_impossible = qa_dict['is_impossible']
    if is_impossible:
        return {}
    answers = qa_dict['answers'][0]
    answer_text = answers['text']
    answer_start = answers['answer_start']
    answer_end = answer_start + len(answer_text)

    qa_info_dict = {
        'question': question,
        'context_id': context_id,
        'answer_start': answer_start,
        'answer_end': answer_end
    }

    return qa_info_dict

def extract_qa_data(data_dict):
    question_records = []
    context_records = []
    qa_records = []

    context_id = 0
    for sample in data_dict['data']:
        for para in sample['paragraphs']:
            context = para['context']
            context_records.append({
                'context_id': context_id,
                'context': context
            })
            for qa in para['qas']:
                qa_dict = extract_qa_info(qa, context_id)
                if qa_dict:
                    qa_records.append(qa_dict)
                    question_records.append({
                        'context_id': context_id,
                        'question': qa_dict['question']
                    })
            context_id += 1
    
    return question_records, context_records, qa_records

def create_csv_data(records, save_path):
    if not os.path.exists(save_path):
        save_to_csv(records, save_path)

ROOT_DIR = 'dataset'
DATASET_NAME = 'squad2.0'
SAVE_PATH = os.path.join(ROOT_DIR, DATASET_NAME)
os.makedirs(SAVE_PATH, exist_ok=True)

train_set_path = os.path.join(SAVE_PATH, 'train-v2.0.json')
dev_set_path = os.path.join(SAVE_PATH, 'dev-v2.0.json')

qa_records_path = 'squad2.0/qa.csv'
context_records_path = 'squad2.0/context.csv'
train_set_dict = load_json(train_set_path)

question_records, context_records, qa_records = extract_qa_data(train_set_dict)
create_csv_data(qa_records, qa_records_path)
create_csv_data(context_records, context_records_path)

