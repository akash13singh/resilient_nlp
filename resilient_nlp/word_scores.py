import numpy as np
from collections import Counter
from utils import preprocess
from datasets import load_dataset, load_from_disk
import json


def compute_word_counts(data, splits):
    for split in splits:
        for label, text in zip(data[split]['label'],data[split]['text']):
            text = text.lower()
            text = preprocess(text)
            for word in text:
                if label==1:
                    positive_counts[word]+=1
                elif label==0:
                    negative_counts[word]+=1
                total_counts[word]+=1

    # pos_neg_ratios = Counter()
    #
    # for term, cnt in list(total_counts.most_common()):
    #     if (cnt > 100):
    #         pos_neg_ratio = positive_counts[term] / float(negative_counts[term] + 1)
    #         pos_neg_ratios[term] = pos_neg_ratio


def compute_conditional_probs():
    V = len(total_counts.keys())
    # get the number of unique positive and negative words
    N_pos = len(positive_counts.keys())
    N_neg = len(negative_counts.keys())

    def word_loglikelihood(w):
        if w in total_counts:
            p_w_pos = (positive_counts.get(w, 0) + 1 / (N_pos + V))
            p_w_neg = (negative_counts.get(w, 0) + 1 / (N_neg + V))
            return np.log(p_w_pos / p_w_neg)
        else:
            return (0)


    for word in total_counts.keys():
        if total_counts[word] >= 100:
            word_scores[word] = (word_loglikelihood(word))


imdb = load_from_disk("../data/imdb")
positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()
word_scores = {}
path = "../output/imdb_word_scores.json"

# compute word counts
compute_word_counts(imdb, ['train', 'test', 'dev'])
# compute conditiona probabilities
compute_conditional_probs()

print(dict(sorted(word_scores.items(), key=lambda item: item[1], reverse=True)[0:20]))
print(sorted(word_scores.items(), key=lambda item: item[1])[0:20])

with open(path, 'w') as f:
    json.dump(word_scores, fp=f)