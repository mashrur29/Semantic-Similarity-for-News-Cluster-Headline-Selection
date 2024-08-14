import pickle

import torch
from transformers import BertTokenizer
from rank_bm25 import BM25Okapi

from models import SemanticSimilarityBERT, NewsArticleDataset


def semantic_similarity_bm25(sent1, sent2):
    corpus = [sent2]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = sent1.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)

    return abs(doc_scores[0])


def semantic_smilarity_bert(sentence_pair, model, tokenizer, device):
    test_input = tokenizer(sentence_pair, padding='max_length', max_length=128, truncation=True,
                           return_tensors="pt").to(device)
    test_input['input_ids'] = test_input['input_ids']
    test_input['attention_mask'] = test_input['attention_mask']
    del test_input['token_type_ids']
    output = model(test_input)
    sim = torch.nn.functional.cosine_similarity(output[0], output[1], dim=0).item()
    return sim


def test():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Device:', device)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    semantic_bert = SemanticSimilarityBERT()
    semantic_bert.load_state_dict(torch.load('semantic_similarity_bert_2.pt'))
    semantic_bert.to(device)

    with open('test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
        print('Test data loaded')

    print('Model loaded')

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


def show_test_dataset():
    with open('test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
        print('Test data loaded')

    for count, it in enumerate(test_dataset):

        print('Title: {}'.format(it['title']))
        print('Body: {}'.format(it['body']))

        if count > 10:
            break



if __name__ == '__main__':
    # test()
    show_test_dataset()