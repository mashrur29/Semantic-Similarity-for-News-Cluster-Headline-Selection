import torch
from rank_bm25 import BM25Okapi

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