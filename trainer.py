from datasets import load_dataset, DatasetDict
from sentence_transformers import SentenceTransformer, models
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import datetime
import random
import numpy as np
import pandas as pd
from models import *
from utils import *
import json
import pickle

def train(model, epochs, train_dataloader, dev_dataloader, optimizer, scheduler, device):
    seed_val = 42
    criterion = CosineSimilarityLoss()
    criterion = criterion.cuda()
    random.seed(seed_val)
    torch.manual_seed(seed_val)

    training_stats = []
    total_t0 = time.time()
    for epoch_i in range(0, epochs):
        t0 = time.time()
        total_train_loss = 0
        model.train()

        for train_data, train_label in tqdm(train_dataloader):
            train_data['input_ids'] = train_data['input_ids'].to(device)
            train_data['attention_mask'] = train_data['attention_mask'].to(device)
            train_data = concat_features(train_data)
            model.zero_grad()
            output = [model(feature).to(torch.float) for feature in train_data]
            loss = criterion(output, train_label.to(device))
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = (time.time() - t0)
        t0 = time.time()
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for dev_data, dev_label in tqdm(dev_dataloader):
            dev_data['input_ids'] = dev_data['input_ids'].to(device)
            dev_data['attention_mask'] = dev_data['attention_mask'].to(device)
            dev_data = concat_features(dev_data)
            with torch.no_grad():
                output = [model(feature).to(torch.float) for feature in dev_data]
            loss = criterion(output, dev_label.to(device))
            total_eval_loss += loss.item()

        avg_dev_loss = total_eval_loss / len(dev_dataloader)

        dev_time = (time.time() - t0)

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Eval Loss': avg_dev_loss,
                'Training Time': training_time,
                'Dev Time': dev_time
            }
        )
    return model, training_stats

def test(model_path, tokenizer, device):
    print('Running test')

    semantic_bert = SemanticSimilarityBERT()
    semantic_bert.load_state_dict(torch.load(model_path))
    semantic_bert.to(device)
    semantic_bert.eval()

    print('Model loaded')

    with open('test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
        print('Test data loaded')

    positive = 0
    negative = 0
    for it in test_dataset:
        label = it['score']

        if int(label) == 1:
            positive = positive + 1
        else:
            negative = negative + 1

    print('Positive labels: {}, negative labels: {}, total: {}'.format(positive, negative, positive + negative))

    accurate_bert = 0
    accurate_bm25 = 0
    total = len(test_dataset)

    for it in test_dataset:
        sent1 = it['title']
        sent2 = it['body']
        label = it['score']

        score = semantic_smilarity_bert([sent1, sent2], semantic_bert, tokenizer, device)

        if int(label) == 1:
            if score > 0.5:
                accurate_bert = accurate_bert + 1
        else:
            if score <= 0.5:
                accurate_bert = accurate_bert + 1

        score = semantic_similarity_bm25(sent1, sent2)

        if int(label) == 1:
            if score > 0.5:
                accurate_bm25 = accurate_bm25 + 1
        else:
            if score <= 0.5:
                accurate_bm25 = accurate_bm25 + 1

    print('Accuracy BERT: {}'.format(float(accurate_bert) / total))
    print('Accuracy BM25: {}'.format(float(accurate_bm25) / total))

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Device:', device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    dataset = load_dataset('csv', data_files='dataset/processed_data.csv')

    train_test_split = dataset['train'].train_test_split(test_size=0.2)
    test_val_split = train_test_split['test'].train_test_split(test_size=0.5)

    dataset_splits = DatasetDict({
        'train': train_test_split['train'],
        'test': test_val_split['test'],
        'dev': test_val_split['train']
    })

    with open('test_dataset.pkl', 'wb') as f:
        pickle.dump(dataset_splits['test'], f)

    print(dataset_splits)
    model = SemanticSimilarityBERT()
    model.to(device)

    train_dataset = NewsArticleDataset(dataset_splits['train'], tokenizer, 128)
    dev_dataset = NewsArticleDataset(dataset_splits['dev'], tokenizer, 128)

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, num_workers=4, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    epochs = 5
    num_training_steps = epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    model, training_stats = train(model, epochs, train_dataloader, dev_dataloader, optimizer, scheduler, device)

    torch.save(model.state_dict(), 'semantic_similarity_bert_2.pt')

    df_stats = pd.DataFrame(data=training_stats)

    df_stats = df_stats.set_index('epoch')

    print(df_stats)

    test('semantic_similarity_bert_2.pt', tokenizer, device)

    df_stats[['Training Loss', 'Eval Loss']].plot().show()



if __name__ == '__main__':
    main()
