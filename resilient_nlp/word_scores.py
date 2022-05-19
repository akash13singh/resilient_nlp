import math
import numpy as np
from collections import Counter
from utils import preprocess
from datasets import load_dataset, load_from_disk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import json
import sys


def compute_word_counts(data, splits, num_classes):
    for split in splits:
        for label, text in zip(data[split]['label'],data[split]['text']):
            text = text.lower()
            text = preprocess(text)
            for word in text:
                in_class_counts[int(label)][word]+=1
                for other_label in range(num_classes):
                    if other_label != label:
                        out_of_class_counts[other_label][word]+=1
                total_counts[word]+=1

    # pos_neg_ratios = Counter()
    #
    # for term, cnt in list(total_counts.most_common()):
    #     if (cnt > 100):
    #         pos_neg_ratio = positive_counts[term] / float(negative_counts[term] + 1)
    #         pos_neg_ratios[term] = pos_neg_ratio


def compute_conditional_probs(num_classes):
    V = len(total_counts.keys())
    # get the number of unique words in and out of class
    for label in range(num_classes):
        N_pos = len(in_class_counts[label].keys())
        N_neg = len(out_of_class_counts[label].keys())

        def word_loglikelihood(w, label):
            if w in total_counts:
                p_w_pos = (in_class_counts[label].get(w, 0) + 1 / (N_pos + V))
                p_w_neg = (out_of_class_counts[label].get(w, 0) + 1 / (N_neg + V))
                return np.log(p_w_pos / p_w_neg)
            else:
                return (0)

        for word in total_counts.keys():
            if total_counts[word] >= 100:
                word_scores[label][word] = (word_loglikelihood(word, label))


dataset_name = 'imdb'
if len(sys.argv) >= 2:
    dataset_name = sys.argv[1]

if dataset_name == 'imdb':
    dataset = load_dataset("output/huggingface/imdb")
    num_classes = 2
    path = "../output/imdb_word_scores.json"
    splits = ['train', 'test', 'dev']
elif dataset_name == 'sst':
    treebank_detok = TreebankWordDetokenizer()

    dataset = load_dataset('sst').map(
        lambda row: {
            "text": treebank_detok.detokenize(row["sentence"].split()),
            "label": min(math.floor(row["label"] / 0.2), 4.0),
        }, remove_columns=['sentence', 'tokens', 'tree']
    )
    num_classes = 5
    path = "../output/sst_word_scores.json"
    splits = ['train', 'test', 'validation']
elif dataset_name == 'sst_bin':
    treebank_detok = TreebankWordDetokenizer()

    dataset = load_dataset('sst').filter(
        lambda row: row["label"] < 0.4 or row["label"] >= 0.6
    ).map(
        lambda row: {
            "text": treebank_detok.detokenize(row["sentence"].split()),
            "label": min(math.floor(row["label"] / 0.5), 1.0),
        }, remove_columns=['sentence', 'tokens', 'tree']
    )
    num_classes = 2
    path = "../output/sst_bin_word_scores.json"
    splits = ['train', 'test', 'validation']
elif dataset_name == 'yelp_bin':
    dataset = load_dataset('yelp_polarity')
    num_classes = 2
    path = "../output/yelp_bin_word_scores.json"
    splits = ['train', 'test']
elif dataset_name == 'yelp_full':
    dataset = load_dataset('yelp_review_full')
    num_classes = 5
    path = "../output/yelp_full_word_scores.json"
    splits = ['train', 'test']

in_class_counts = [ Counter() for i in range(num_classes) ]
out_of_class_counts = [ Counter() for i in range(num_classes) ]
total_counts = Counter()
word_scores = [ {} for i in range(num_classes) ]

# compute word counts
compute_word_counts(dataset, splits, num_classes)
# compute conditiona probabilities
compute_conditional_probs(num_classes)

for label in range(num_classes):
    print(f"Class {label}")
    print(dict(sorted(word_scores[label].items(), key=lambda item: item[1], reverse=True)[0:20]))
    print(sorted(word_scores[label].items(), key=lambda item: item[1])[0:20])

with open(path, 'w') as f:
    json.dump(word_scores, fp=f)
