import sys
sys.path.append("..")
from resilient_nlp.utils import preprocess
from datasets import load_dataset
import pandas as pd
import numpy as np
from collections import Counter
import random
from resilient_nlp.perturbers import ToyPerturber, WordScramblerPerturber
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
import torch
from tqdm import tqdm
import json
import datetime


class BertWordScoreAttack:
    def __init__(self, perturber, word_scores_file, model, tokenizer,  max_sequence_length, attack_whitespace=True):
        self.perturber = perturber
        self.model = model
        self.load_word_scores(word_scores_file)
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.attack_whitespace = attack_whitespace

    def load_word_scores(self, path):
        with open(path) as f:
            self.word_scores = json.load(fp=f)
            #print(self.word_scores)

    def bert_tokenize(self, sent):
        return self.tokenizer(sent, truncation=True, padding='max_length', max_length=self.max_sequence_length,
                              return_tensors='pt')

    def get_bert_output(self, sentences):
        if self.tokenizer is not None:
            tokenized = self.bert_tokenize(sentences)
            output = self.model(**tokenized)
        else:
            output = self.model(sentences)
        logits = output['logits']
        preds = torch.argmax(logits, dim=1)
        smax = torch.nn.Softmax(dim=1)
        probs = smax(logits)
        return preds, torch.round(probs[range(len(sentences)), preds], decimals=4)

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

    def attempt_word_merge(self, text, attack_word):
        start_from = 0

        #print(f"** Attempting word merge, attacked word: {attack_word}")
        #print(f"   orig text: {text}")

        while True:
            pos = text.find(attack_word, start_from)
            end_pos = pos + len(attack_word)
            if pos == -1:
                #print("   merge failed")
                return text
            start_from = pos + 1

            possible_actions = []

            if pos >= 2 and text[pos - 1].isspace() and not text[pos - 2].isspace():
                possible_actions.append('front')
            if end_pos + 1 < len(text) and text[end_pos].isspace() and not text[end_pos + 1].isspace():
                possible_actions.append('back')

            if not possible_actions:
                # This is most likely because the word appeared as a substring
                continue
            action = random.choice(possible_actions)

            if action == 'front':
                result = text[:pos-1] + text[pos:]
            else:
                result = text[:end_pos] + text[end_pos+1:]

            #print(f"   result:    {result}")
            return result

    def attack_single(self, sample_idx, orig_text, ground_truth, max_tokens_to_perturb, max_tries_per_token,
                      mode, logging, orig_preds, attack_status, perturbed_texts,
                      orig_tokens, perturbed_tokens, n_queries, perturbed_preds):

            #print(f"------------- Sample: {sample_idx} ---------------------------------")
            # print(orig_text)
            query_and_response = [ orig_text, None, None ]
            yield query_and_response
            orig_pred = query_and_response[1]
            orig_score = query_and_response[2]

            orig_preds[sample_idx] = perturbed_preds[sample_idx] = orig_pred

            if ground_truth != orig_pred:  # Model has an error. skip_attack
                # print(f'Sample {sample_idx}. Attack Skipped')
                attack_status[sample_idx] = 'Skipped'
                yield None

            orig_text = orig_text.lower()
            tokens = preprocess(orig_text)
            token_scores = {token: self.word_scores[int(ground_truth)].get(token, 0) for token in tokens}

            attack_tokens = sorted(token_scores.items(), key=lambda item: item[1], reverse=True)

            attack_passed = False
            token_idx = 0
            sample_query_counter = 0
            text = orig_text
            worst_score = orig_score
            worst_text = orig_text

            # Since we're preparing multiple calls to the model in order to submit
            # a batch request, make sure that the perturbations are reproducible. To
            # do this each attacked sentence will have its own random state.
            random.seed(orig_text)

            while token_idx < len(attack_tokens) and token_idx < max_tokens_to_perturb and not attack_passed:
                # print(f"----- token_idx: {token_idx} --------------")
                # token_idx = np.random.choice(top_n_tokens)
                attack_token = attack_tokens[token_idx][0]
                token_tries_counter = 0

                candidates = []

                for n_try in range(max_tries_per_token):
                    perturbed_text = text
                    attempted_word_merge = False
                    # Note: word merging needs to be handled separately, since we only ever
                    #       pass a single token to the perturber. So with some probability
                    #       we will just handle merging ourselves here
                    if self.attack_whitespace and random.random() < 0.2:
                        perturbed_text = self.attempt_word_merge(text, attack_token)
                        perturbed_token = attack_token
                        attempted_word_merge = True
                    if perturbed_text == text:
                        perturbed_token = self.perturber.perturb([attack_token])[0][0]
                        #TODO what if attack/perturbed token is part of another bigger token
                        perturbed_text = text.replace(attack_token, perturbed_token, 1)
                    if self.attack_whitespace and perturbed_text == text and not attempted_word_merge:
                        perturbed_text = self.attempt_word_merge(text, attack_token)
                        perturbed_token = attack_token
                        attempted_word_merge = True

                    # We need a model prediction to proceed. So we return a 'query' and expect
                    # our caller to provide the prediction and score after the model call is
                    # done. Also we're saving the random state since we're yielding control and
                    # another cooperative 'thread' might change it.
                    random_state = random.getstate()
                    query_and_response = [ perturbed_text, None, None ]
                    yield query_and_response
                    perturbed_pred = query_and_response[1]
                    perturbed_score = query_and_response[2]
                    random.setstate(random_state)

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

            yield None

    def attack(self, dataset,max_tokens_to_perturb=-1, max_tries_per_token=1, mode=0, attack_results_csv=None, logging=False,
               print_summary=True, eval_batch_size=32):
        """
        mode 0: Preserve best unsuccessful perturbation per token. Final attack can perturb up to max_tokens_to_query tokens.
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

        generators = []

        for sample_idx, (orig_text, ground_truth) in enumerate(zip(orig_texts, actuals)):
            generators.append(self.attack_single(sample_idx, orig_text, ground_truth,
                max_tokens_to_perturb, max_tries_per_token,
                mode, logging, orig_preds, attack_status, perturbed_texts,
                orig_tokens, perturbed_tokens, n_queries, perturbed_preds))
        generators.reverse()

        cur_gens = []

        progress_bar = tqdm(total=len(generators))

        while len(cur_gens) > 0 or len(generators) > 0:
            # Fill in generators
            while len(cur_gens) < eval_batch_size and len(generators) > 0:
                cur_gens.append(generators.pop())

            batch = []
            new_generators = []
            for g in cur_gens:
                query = next(g)
                if query is not None:
                    new_generators.append(g)
                    batch.append(query)
                else:
                    progress_bar.update()

            if len(batch) > 0:
                sentences = [ item[0] for item in batch ]
                #print(f"submitting sentences:\n{sentences}")
                preds, scores = self.get_bert_output(sentences)
                #print(f"obtained preds\n{preds}\nand scores\n{scores}")
                for i in range(len(sentences)):
                    batch[i][1] = preds[i]
                    batch[i][2] = scores[i]

            cur_gens = new_generators
        progress_bar.close()

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

    # weight_merge_words is set to 0 since there will always be 1 word at a time
    # (so nothing to possibly merge)
    wsp = WordScramblerPerturber(perturb_prob=1, weight_add=1, weight_drop=1, weight_swap=1,
                                                weight_split_word=1,weight_merge_words=0)
    dataset = load_dataset("artemis13fowl/imdb", split="attack_eval_truncated")


    attack_settings = [


    ]


    attacker = BertWordScoreAttack(wsp, word_scores_file, model, tokenizer,  max_sequence_length)

    # set attack parameters
    max_tokens_to_perturb =40
    max_tries_per_token = 4
    mode = 1
    attack_name_string = f'_{max_tokens_to_perturb}_{max_tries_per_token}_{mode}_{datetime.datetime.now().isoformat(" ", "seconds")}'
    attack_data_file = f'output/word_score_attack_data_{attack_name_string}.csv'
    attack_results_file = f'output/word_score_attack_results_{attack_name_string}.json'

    #attack!
    attack_results = attacker.attack(dataset,
                                  max_tokens_to_perturb=max_tokens_to_perturb,
                                  max_tries_per_token=max_tries_per_token,
                                  mode=mode,
                                  attack_results_csv=attack_data_file,
                                  logging=False,
                                  print_summary=True,
                                  eval_batch_size=1)
    #
    attack_stats = attacker.compute_attack_stats()
    print(attack_stats)
    with open(attack_results_file, 'w') as f:
        json.dump(attack_stats, fp=f)

