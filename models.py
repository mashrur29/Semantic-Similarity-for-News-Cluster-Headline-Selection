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

class NewsArticleDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.scores = [it['score'] for it in dataset]
        self.body = [it['body'] for it in dataset]
        self.title = [it['title'] for it in dataset]
        self.articles = [[str(x), str(y)] for x, y in zip(self.title, self.body)]

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        yy = torch.tensor(self.scores[idx], dtype=float)
        xx = self.tokenizer(self.articles[idx], max_length=self.max_length, padding='max_length', truncation=True,
                            return_tensors="pt")
        return xx, yy


class SemanticSimilarityBERT(torch.nn.Module):
    def __init__(self):
        super(SemanticSimilarityBERT, self).__init__()
        self.bert = models.Transformer('bert-base-uncased', max_seq_length=128)
        self.pooling_layer = models.Pooling(self.bert.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[self.bert, self.pooling_layer])

    def forward(self, input_data):
        output = self.model(input_data)['sentence_embedding']
        return output


def concat_features(x):
    input_ids = x['input_ids']
    attention_masks = x['attention_mask']
    features = [{'input_ids': input_id, 'attention_mask': attention_mask}
                for input_id, attention_mask in zip(input_ids, attention_masks)]
    return features


class CosineSimilarityLoss(torch.nn.Module):

    def __init__(self, loss_fn=torch.nn.MSELoss(), transform_fn=torch.nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fn = loss_fn
        self.transform_fn = transform_fn
        self.cos_similarity = torch.nn.CosineSimilarity(dim=1)

    def forward(self, inputs, labels):
        emb_1 = torch.stack([inp[0] for inp in inputs])
        emb_2 = torch.stack([inp[1] for inp in inputs])
        outputs = self.transform_fn(self.cos_similarity(emb_1, emb_2))
        return self.loss_fn(outputs.to(torch.float), labels.squeeze().to(torch.float))