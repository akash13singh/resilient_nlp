# This code has been minimally adapted by [redacted] from the MIT licensed
# repository:
#   https://github.com/ejones313/roben
# with the aim of fitting in a single module and avoiding pulling in lots of
# libraries. All my changes are in the public domain.

# MIT License
#
# Copyright (c) 2020 Erik Jones
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import reduce
import itertools
import json
import numpy as np
import os
import pickle
import random
import torch
from tqdm import tqdm

OOV_CLUSTER = -1
OOV_TOKEN = '<UNK>'

def get_sorted_word(word):
    if len(word) < 3:
        sorted_word = word
    else:
        sorted_word = '{}{}{}'.format(word[0], ''.join(sorted(word[1:-1])), word[-1])
    return sorted_word

class Clustering(object):
    """Object representing an assignment of words to clusters.

    Provides some utilities for dealing with words, typos, and clusters.
    """
    def __init__(self, clusterer_dict, max_num_possibilities=None, passthrough=False):
        self.cluster2elements = clusterer_dict['cluster']
        self.word2cluster = clusterer_dict['word2cluster']
        self.cluster2representative = clusterer_dict['cluster2representative']
        self.word2freq = clusterer_dict['word2freq']
        self.typo2cluster = clusterer_dict['typo2cluster']
        if max_num_possibilities:
            self.cluster2elements = self.filter_possibilities(max_num_possibilities)

    def filter_possibilities(self, max_num_possibilities):
        filtered_cluster2elements = {}
        for cluster in self.cluster2elements:
            elements = self.cluster2elements[cluster]
            frequency_list = [(elem, self.word2freq[elem]) for elem in elements]
            frequency_list.sort(key = lambda x: x[1], reverse = True)
            filtered_elements = [pair[0] for pair in frequency_list[:max_num_possibilities]]
            filtered_cluster2elements[cluster] = filtered_elements
        return filtered_cluster2elements

    @classmethod
    def from_pickle(cls, path, **kwargs):
        with open(path, 'rb') as f:
            clusterer_dict = pickle.load(f)
        return cls(clusterer_dict, **kwargs)

    def get_words(self, cluster):
        if cluster == OOV_CLUSTER:
            return [OOV_TOKEN]
        return self.cluster2elements[cluster]

    def in_vocab(self, word):
        return word in self.word2cluster

    def get_cluster(self, word):
        """Get cluster of a word, or OOV_CLUSTER if out of vocabulary."""
        word = word.lower()
        if word in self.word2cluster:
            return self.word2cluster[word]
        return OOV_CLUSTER

    def get_rep(self, cluster):
        """Get representative for a cluster."""
        if cluster == OOV_CLUSTER:
            return OOV_TOKEN
        return self.cluster2representative[cluster]

    def get_freq(self, word):
        return self.word2freq[word]

    def map_token(self, token, remap_vocab=True, passthrough = False):
        """Map a token (possibly a typo) to a cluster.

        Args:
            token: a token, possibly a typo
            remap_vocab: if False, always map vocab words to themselves,
                because perturbing vocab words has been disallowed.
            passthrough: Allow OOV to go to downstream model...
        """
        token = token.lower()
        if token in self.word2cluster and not remap_vocab:
            return self.get_cluster(token)
        if token in self.typo2cluster:
            return self.typo2cluster[token]
        if passthrough:
            return token
        return OOV_CLUSTER


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label



class Recoverer(object):
    """Clean up a possibly typo-ed string."""
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.cache = {}
        self.name = None  # Subclasses should set this

    def _cache_path(self):
        return os.path.join(self.cache_dir, 'recoveryCache-{}.json'.format(self.name))

    def load_cache(self):
        path = self._cache_path()
        if os.path.exists(path):
            with open(self._cache_path()) as f:
                self.cache = json.load(f)
            print('Recoverer: loaded {} values from cache.'.format(len(self.cache)))
        else:
            print('Recoverer: no cache at {}.'.format(path))

    def save_cache(self, save = False):
        if save:
            cache_path = self._cache_path()
            print('Recoverer: saving {} cached values to {} .'.format(len(self.cache), cache_path))
            with open(cache_path, 'w') as f:
                json.dump(self.cache, f)


    def recover(self, text):
        """Recover |text| to a new string.
        
        Used at test time to preprocess possibly typo-ed input. 
        """
        if text in self.cache:
            return self.cache[text]
        recovered = self._recover(text)
        self.cache[text] = recovered
        return recovered

    def _recover(self, text):
        """Actually do the recovery for self.recover()."""
        raise NotImplementedError

    def get_possible_recoveries(self, text, attack_surface, max_num, analyze_res_attacks = False, ret_ball_stats = False):
        """For a clean string, return list of possible recovered strings, or None if too many.
        
        Used at certification time to exactly compute robust accuracy.

        Returns tuple (list_of_possibilities, num_possibilities)
        where list_of_possibilities is None if num_possibilities > max_num.
        """
        pass

    def recover_example(self, example):
        """Recover an InputExample |example| to a new InputExample.
        
        Used at test time to preprocess possibly typo-ed input. 
        """
        tokens = example.text_a.split()
        a_len = len(tokens)
        if example.text_b:
            tokens.extend(example.text_b.split())
        recovered_tokens = self.recover(' '.join(tokens)).split()
        a_new = ' '.join(recovered_tokens[:a_len])
        if example.text_b:
            b_new = ' '.join(recovered_tokens[a_len:])
        else:
            b_new = None
        return InputExample(example.guid, a_new, b_new, example.label)

    def get_possible_examples(self, example, attack_surface, max_num, analyze_res_attacks = False):
        """For a clean InputExample, return list of InputExample's you could recover to.
        
        Used at certification time to exactly compute robust accuracy.
        """
        tokens = example.text_a.split()
        a_len = len(tokens)
        if example.text_b:
            tokens.extend(example.text_b.split())
        possibilities, num_poss, perturb_counts = self.get_possible_recoveries(' '.join(tokens), attack_surface, max_num,
                                    analyze_res_attacks = analyze_res_attacks)
        if perturb_counts is not None:
            assert len(perturb_counts) == len(possibilities)
        if not possibilities:
            return (None, num_poss)
        out = []
        example_num = 0
        for i in range(len(possibilities)):
            poss = possibilities[i]
            poss_tokens = poss.split()
            a = ' '.join(poss_tokens[:a_len])
            if example.text_b:
                b = ' '.join(poss_tokens[a_len:])
            else:
                b = None
            if not analyze_res_attacks:
                poss_guid = '{}-{}'.format(example.guid, example_num)
            else:
                poss_guid = '{}-{}-{}'.format(example.guid, example_num, perturb_counts[i])
            out.append(InputExample('{}-{}'.format(poss_guid, example_num), a, b, example.label))
            example_num += 1
        return (out, len(out))


class ClusterRecoverer(Recoverer):
    def __init__(self, cache_dir, clustering):
        super(ClusterRecoverer, self).__init__(cache_dir)
        self.clustering = clustering
        self.passthrough = False

    def get_possible_recoveries(self, text, attack_surface, max_num, analyze_res_attacks = False, ret_ball_stats = False):
        tokens = text.split()
        possibilities = []
        perturb_counts = []
        standard_clustering = np.array([self.clustering.map_token(token) for token in tokens])
        for token in tokens:
            cur_perturb = attack_surface.get_perturbations(token)
            perturb_counts.append(len(cur_perturb))
            poss_clusters = set()
            for pert in cur_perturb:
                clust_id = self.clustering.map_token(pert)
                poss_clusters.add(clust_id)
            possibilities.append(sorted(poss_clusters, key=str))  # sort for deterministic order
        if ret_ball_stats:
            return [len(pos_clusters) for pos_clusters in possibilities], perturb_counts
        num_pos = reduce(lambda x, y: x * y, [len(x) for x in possibilities])
        if num_pos > max_num:
            return (None, num_pos, None)
        poss_recoveries = []
        perturb_counts = None
        if analyze_res_attacks:
            perturb_counts = []
        num_zero = 0
        for clust_seq in itertools.product(*possibilities):
            if analyze_res_attacks:
                #print("Stand: ", standard_clustering)
                #print("Seq: ", clust_seq)
                #print("Lengths: {}, {}".format(len(standard_clustering), len(clust_seq)))
                #print("Types: {}, {}".format(type(np.array(clust_seq)[0]), type(standard_clustering[0])))
                #print("Comparison: ", np.array(clust_seq) != standard_clustering)
                #print("Inv comparison: ", np.array(clust_seq) == standard_clustering)
                num_different = (np.array(clust_seq) != standard_clustering).sum()
                if num_different == 0:
                    num_zero += 1
                #print(num_different)
                perturb_counts.append(num_different)
            poss_recoveries.append(self._recover_from_clusters(clust_seq))
        assert num_zero == 1 or not analyze_res_attacks
        return (poss_recoveries, len(poss_recoveries), perturb_counts)

    def _recover(self, text):
        tokens = text.split()
        clust_ids = [self.clustering.map_token(w, passthrough = self.passthrough) for w in tokens]
        return self._recover_from_clusters(clust_ids)

    def _recover_from_clusters(self, clust_ids):
        raise NotImplementedError


class ClusterRepRecoverer(ClusterRecoverer):
    def _recover_from_clusters(self, clust_ids):
        tokens = []
        for c in clust_ids:
            if c == OOV_CLUSTER:
                tokens.append('[MASK]')
            else:
                tokens.append(self.clustering.get_rep(c))
                print(c)
        return ' '.join(tokens)

# Added by [redacted]. It looks like for general inputs, we end up with a *lot*
# of [MASK] tokens. This is a slight modification of the recoverer that just
# leaves the OOV tokens unchanged.

# All of [redacted]'s modifications are in the public domain.

class ClusterRecovererWithPassthrough(ClusterRecoverer):
    def __init__(self, cache_dir, clustering):
        super(ClusterRecovererWithPassthrough, self).__init__(cache_dir, clustering)
        self.passthrough = True

    def _recover_from_clusters(self, clust_ids):
        tokens = []
        for c in clust_ids:
            if c == OOV_CLUSTER:
                tokens.append('[MASK]')
            elif isinstance(c, str):
                tokens.append(c)
            else:
                tokens.append(self.clustering.get_rep(c))
        return ' '.join(tokens)
