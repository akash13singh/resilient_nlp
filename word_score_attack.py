import sys
sys.path.append("..")
from resilient_nlp.utils import preprocess
from datasets import load_dataset
import pandas as pd
import numpy as np
from collections import Counter
from resilient_nlp.perturbers import ToyPerturber, WordScramblerPerturber
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
import torch
from tqdm import tqdm
import json


class BertWordScoreAttack:
    def __init__(self, perturber, word_scores_file, model, tokenizer,  max_sequence_length ):
        self.perturber = perturber
        self.model = model
        self.load_word_scores(word_scores_file)
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def load_word_scores(self, path):
        with open(path) as f:
            self.word_scores = json.load(fp=f)
            #print(self.word_scores)

    def bert_tokenize(self, sent):
        return self.tokenizer(sent, truncation=True, padding='max_length', max_length=self.max_sequence_length,
                              return_tensors='pt')

    def get_bert_output(self, text):
        if self.tokenizer is not None:
            tokenized = self.bert_tokenize([text])
            output = self.model(**tokenized)
        else:
            output = self.model([text])
        logits = output['logits']
        pred = torch.argmax(logits, dim=1).item()
        smax = torch.nn.Softmax(dim=1)
        probs = smax(logits)
        return pred, np.round(probs[0][pred].item(), 4)

    def perturb_word(word):
        pass

    def compute_attack_stats(self):
        attack_stats = {}
        attack_stats['total_attacks'] = len(self.results)
        attack_stats['avg_n_queries'] = np.round(self.results.num_queries.mean(), 2)
        attack_stats['successful_attacks'] = self.results.loc[self.results['attack_status'] == "Successful"].shape[0]
        attack_stats['failed_attacks'] = self.results.loc[self.results['attack_status'] == "Failed"].shape[0]
        attack_stats['skipped_attacks'] = self.results.loc[self.results['attack_status'] == "Skipped"].shape[0]
        attack_stats['attack_success_rate'] = 100 * np.round(
            attack_stats['successful_attacks'] / (attack_stats['successful_attacks'] + attack_stats['failed_attacks']), 2)
        attack_stats['orig_accuracy'] = (attack_stats['total_attacks'] - attack_stats['skipped_attacks']) * 100.0 / (
        attack_stats['total_attacks'])
        attack_stats['attack_accuracy'] = (attack_stats['failed_attacks']) * 100.0 / (attack_stats['total_attacks'])
        return attack_stats

    def attack(self, dataset, max_tokens_to_query=-1, max_tries_per_token=1, mode=0, attack_results_csv=None, logging=False,
               print_summary=True):
        """
        mode 0: Perserve best unsuccessful perturbation per token. Final attack can perturb utpo max_tokens_to_query tokens.
        mode 1: Forgets unccessful perturbations. Final Attacks perturbs only 1 token per sample.
        """
        actuals = dataset['label']
        orig_texts = dataset['text']
        n_samples = len(actuals)

        orig_preds = np.zeros(n_samples)
        attack_status = np.empty(n_samples, dtype='object')
        perturbed_texts = np.empty(n_samples, dtype='object')
        orig_tokens = np.empty(n_samples, dtype='object')
        perturbed_tokens = np.empty(n_samples, dtype='object')
        n_queries = np.empty(n_samples, dtype='object')
        perturbed_preds = np.zeros(n_samples)

        for sample_idx, (orig_text, ground_truth) in tqdm(enumerate(zip(orig_texts, actuals))):
            #print(f"------------- Sample: {sample_idx} ---------------------------------")
            # print(orig_text)
            orig_pred, orig_score = self.get_bert_output(orig_text)
            orig_preds[sample_idx] = perturbed_preds[sample_idx] = orig_pred

            if ground_truth != orig_pred:  # Model has an error. skip_attack
                # print(f'Sample {sample_idx}. Attack Skipped')
                attack_status[sample_idx] = 'Skipped'
                continue

            orig_text = orig_text.lower()
            tokens = preprocess(orig_text)
            token_scores = {token: self.word_scores[token] if token in self.word_scores else 0 for token in tokens}

            if max_tokens_to_query == -1:
                max_tokens_to_query = len(token_scores)
            else:
                max_tokens_to_query = min(max_tokens_to_query, len(token_scores))

            if orig_pred == 0:  # fetch -ve sentiment tokens
                attack_tokens = sorted(token_scores.items(), key=lambda item: item[1])[:max_tokens_to_query]
            else:  # fetch +ve sentiment tokens
                attack_tokens = sorted(token_scores.items(), key=lambda item: item[1], reverse=True)[:max_tokens_to_query]

            attack_passed = False
            token_idx = 0
            sample_query_counter = 0
            text = orig_text
            worst_score = orig_score
            worst_text = orig_text

            while token_idx < max_tokens_to_query and not attack_passed:
                # print(f"----- token_idx: {token_idx} --------------")
                # token_idx = np.random.choice(top_n_tokens)
                attack_token = attack_tokens[token_idx][0]
                token_tries_counter = 0

                candidates = []

                for n_try in range(max_tries_per_token):
                    perturbed_token = self.perturber.perturb([attack_token])[0][0]
                    perturbed_text = text.replace(attack_token, perturbed_token, 1)
                    perturbed_pred, perturbed_score = self.get_bert_output(perturbed_text)

                    # print(f"----- n_try: {n_try}----")
                    if logging:
                        print(f'sample# sample_query_counter token_query_counter attack_token perturbed_token orig_pred perturbed_pred worst_score perturbed_score')
                        print(sample_idx, sample_query_counter, token_tries_counter,
                              attack_token, perturbed_token, orig_pred, perturbed_pred,
                              worst_score, perturbed_score)
                        print(perturbed_text)

                    sample_query_counter += 1  # increment sample_query_counter
                    token_tries_counter += 1  ## increment token_tries_counter

                    if perturbed_pred != orig_pred:  # success
                        attack_passed = True
                        attack_status[sample_idx] = 'Successful'
                        perturbed_texts[sample_idx] = perturbed_text
                        orig_tokens[sample_idx] = attack_token
                        perturbed_tokens[sample_idx] = perturbed_token
                        perturbed_preds[sample_idx] = perturbed_pred
                        break

                        # track best attack (worse_score/worse_text) so far.
                    if perturbed_score < worst_score:
                        worst_score = perturbed_score
                        worst_text = perturbed_text

                if mode == 0:  ## if tries exhausted, update text to worst text. Worst perturbation per toekn are maintained.
                    text = worst_text
                token_idx += 1  ## move to next token

            n_queries[sample_idx] = sample_query_counter

            if attack_passed == False:  # attack failed
                # print(f'Sample {sample_idx}. Max tries exhausted')
                attack_status[sample_idx] = 'Failed'

        status_counts = Counter(attack_status)
        if print_summary:
            print(classification_report(actuals, orig_preds))
            print(status_counts)

        results = {'attack_status': attack_status,
                   'ground_truth': actuals,
                   'orig_prediction': orig_preds,
                   'attacked_token': orig_tokens,
                   'perturbed_token': perturbed_tokens,
                   'num_queries': n_queries,
                   'original_text': orig_texts,
                   'perturbed_text': perturbed_texts,
                   'perturbed_preds': perturbed_preds,
                   }

        self.results = pd.DataFrame.from_dict(results)

        if attack_results_csv:
            self.results.to_csv(attack_results_csv, index=False)

        success_rate = np.round(100 * status_counts['Successful'] / (status_counts['Successful'] + status_counts['Failed']),
                                2)
        if print_summary:
            print(f'Success Rate {success_rate}')
            print(f'Avg Queries: {self.results.num_queries.mean()}')
        return self.results

if __name__ == '__main__':
    checkpoint_finetuned = "artemis13fowl/bert-base-uncased-imdb"
    model = BertForSequenceClassification.from_pretrained(checkpoint_finetuned)
    tokenizer_checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    word_scores_file = "output/imdb_word_scores.json"
    max_sequence_length = 128
    batch_size = 32
    eval_steps = 100
    wsp = WordScramblerPerturber(perturb_prob=1, weight_add=1, weight_drop=1, weight_swap=1,
                                                weight_split_word=1,weight_merge_words=1)
    dataset = load_dataset("artemis13fowl/imdb", split="attack_eval_truncated")

    # attack!!
    attacker = BertWordScoreAttack(wsp, word_scores_file, model, tokenizer,  max_sequence_length)
    results = attacker.attack(dataset[:10], max_tokens_to_query=20, max_tries_per_token=3, mode=0, attack_results_csv="output/test_word_attack.csv", logging=True)
    attacker.compute_attack_stats()