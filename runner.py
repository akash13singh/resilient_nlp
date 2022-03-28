import json
import math
import os
import random
import re
import torch
import torch.nn as nn
import torch.optim as optim
import sys

from resilient_nlp.char_tokenizer import CharTokenizer
from resilient_nlp.corpora import BookCorpus
from resilient_nlp.embedders import BertEmbedder
from resilient_nlp.models import LSTMModel, CNNModel
from resilient_nlp.perturbers import NullPerturber, ToyPerturber, \
                                     WordScramblerPerturber

NUM_EPOCHS = 2
BATCH_SIZE = 64

# NOTE: This is the character vocab size
NUM_TOKENS = 1000
NUM_SENTENCES = 64000
CHAR_EMB_SIZE = 768
HIDDEN_SIZE = 768
NUM_LAYERS = 3
CNN_KERNEL_SIZE = 11

# FIXME - this is hardcoded for bert-base
WORD_EMB_SIZE = 768

class ExperimentRunner:
    def __init__(self,
                 device,
                 model_filename=None,
                 model_class=None,
                 model_params=None,
                 objective_model_name='bert-base-uncased',
                 objective_tokenizer_name=None):
        self.device = device

        char_vocab = [ '<unk>', '<s>', '</s>' ]

        if model_filename is not None:
            save_state = torch.load(model_filename, map_location='cpu')
            model_state_dict = save_state['model_state_dict']
            model_class = save_state['model_class']
            model_params = save_state['model_params']
            char_vocab = save_state['char_vocab']

        cls = globals()[model_class]
        self.model = cls(**model_params).to(device)
        self.model_class = model_class

        if model_filename is not None:
            self.model.load_state_dict(model_state_dict)

        self.char_tokenizer = CharTokenizer(
            max_vocab=NUM_TOKENS, initial_vocab=char_vocab,
            start_index=1, end_index=2)
        self.embedder = BertEmbedder(
            model_name=objective_model_name,
            tokenizer_name=objective_tokenizer_name,
            per_character_embedding=True,
            add_special_tokens=False,
            start_char_present=True,
            end_char_present=True)

    def train(self, num_sentences, num_eval_sentences=0, print_batch_stats=False):
        corpus = BookCorpus()
        # perturber = NullPerturber()
        # perturber = ToyPerturber(start_char_present=True, end_char_present=True)
        perturber = WordScramblerPerturber(start_char_present=True, end_char_present=True)

        # Ensure consistent sample
        random.seed(11)
        all_sentences = corpus.sample_items(num_sentences + num_eval_sentences)
        sentences = all_sentences[:num_sentences]
        eval_sentences = all_sentences[num_sentences:]

        loss = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        num_batches = math.ceil(num_sentences / BATCH_SIZE)

        for epoch in range(NUM_EPOCHS):
            epoch_correct = 0
            epoch_total = 0
            epoch_emb_loss = 0.0
            epoch_emb_variance = 0.0
            epoch_mask_loss = 0.0
            epoch_mask_variance = 0.0
            for i in range(num_batches):
                bs = i * BATCH_SIZE
                be = bs + BATCH_SIZE
                num_examples_in_batch = min(BATCH_SIZE, num_sentences - bs)

                bert_embeddings = self.embedder.embed(sentences[bs:be])

                batch_Y = bert_embeddings['embeddings']
                batch_Y_masks = bert_embeddings['masks']

                perturbed_sentences, batch_Y_masks, batch_Y = perturber.perturb(
                    sentences[bs:be],
                    batch_Y_masks,
                    batch_Y)

                sentence_tokens = self.char_tokenizer.tokenize(perturbed_sentences)

                batch_lengths = [ len(tokens) for tokens in sentence_tokens ]
                max_length = max(batch_lengths)
                batch_X = torch.zeros((num_examples_in_batch, max_length),
                    dtype=torch.int)

                for j, tokens in enumerate(sentence_tokens):
                    # There HAS to be a nicer way to do this... :(
                    batch_X[j,:len(tokens)] = torch.IntTensor(tokens)

                batch_X = batch_X.to(self.device)

                batch_Y = batch_Y.to(device)
                batch_Y_masks = batch_Y_masks.to(self.device)

                self.model.zero_grad()
                self.model.train()

                batch_preds, batch_pred_masks = self.model(batch_X, batch_lengths)
                batch_emb_loss = num_examples_in_batch * \
                    loss(batch_preds * batch_Y_masks.unsqueeze(2), batch_Y)
                batch_emb_variance = num_examples_in_batch * \
                    float(torch.mean(torch.var(batch_Y, (0, 1), unbiased=False)))
                batch_mask_loss = num_examples_in_batch * loss(batch_pred_masks,  batch_Y_masks)
                batch_mask_variance = num_examples_in_batch * \
                    float(torch.var(batch_Y_masks, unbiased=False))

                epoch_emb_loss += float(batch_emb_loss)
                epoch_emb_variance += batch_emb_variance
                epoch_mask_loss += float(batch_mask_loss)
                epoch_mask_variance += batch_mask_variance

                loss_multiplier = float(batch_mask_loss) / float(batch_emb_loss)
                batch_combined_loss = (batch_emb_loss * loss_multiplier + batch_mask_loss) / num_examples_in_batch

                batch_correct = 0
                batch_total = 0
                batch_positive = 0

                for idx, length in enumerate(batch_lengths):
                    batch_total += length
                    batch_correct += int(sum(torch.isclose(torch.round(batch_pred_masks[idx][:length]), batch_Y_masks[idx][:length])))
                    batch_positive += int(sum(batch_Y_masks[idx][:length]))

                epoch_correct += batch_correct
                epoch_total += batch_total

                mask_accuracy = batch_correct / batch_total
                epoch_mask_accuracy = epoch_correct / epoch_total

                batch_combined_loss.backward()
                optimizer.step()

                if print_batch_stats:
                    print("%04d-%04d (batch): norm. embedding loss: %f (absolute: %f), norm. mask loss: %f (absolute: %f), mask accuracy: %f" %
                        (epoch, i, batch_emb_loss / batch_emb_variance, batch_emb_loss,
                            batch_mask_loss / batch_mask_variance, batch_mask_loss, mask_accuracy))
                print("%04d-%04d (epoch): norm. embedding loss: %f (absolute: %f), norm. mask loss: %f (absolute: %f), mask accuracy: %f" %
                    (epoch, i, epoch_emb_loss / epoch_emb_variance, epoch_emb_loss,
                        epoch_mask_loss / epoch_mask_variance, epoch_mask_loss, epoch_mask_accuracy))

    def embed(self,
              sentences=None,
              sentence_tokens=None,
              start_token=None,
              end_token=None,
              pad_token=None,
              max_tokens=None):
        self.model.eval()

        assert(sentences is not None or sentence_tokens is not None)

        if sentences is not None:
            sentence_tokens = self.char_tokenizer.tokenize(sentences, extend_vocab=False)

        lengths = [ len(sentence) for sentence in sentence_tokens ]
        max_length = max(lengths)
        emb_lengths = []

        num_batches = math.ceil(len(sentence_tokens) / BATCH_SIZE)
        batch_embeddings = []
        for i in range(num_batches):
            bs = i * BATCH_SIZE
            be = min(bs + BATCH_SIZE, len(sentence_tokens))
            num_examples_in_batch = be - bs

            X = torch.zeros((num_examples_in_batch, max_length), dtype=torch.int)
            for j, tokens in enumerate(sentence_tokens[bs:be]):
                X[j,:len(tokens)] = torch.IntTensor(tokens)
            X = X.to(self.device)

            batch_embedding, batch_emb_lengths = self.model.predict_embeddings(X, lengths[bs:be],
                start_token=start_token, end_token=end_token, pad_token=pad_token,
                max_tokens=max_tokens)
            batch_embeddings.append(batch_embedding)
            emb_lengths += batch_emb_lengths.tolist()

        max_emb_length = max(emb_lengths)
        embeddings = torch.zeros((len(sentence_tokens), max_emb_length, WORD_EMB_SIZE), dtype=torch.float)

        for i in range(num_batches):
            bs = i * BATCH_SIZE
            be = min(bs + BATCH_SIZE, len(sentence_tokens))
            num_examples_in_batch = be - bs
            batch_embedding = batch_embeddings[i]

            embeddings[bs:be,:batch_embedding.shape[1]] = batch_embedding

        attention_mask = torch.IntTensor(
            [
                [1] * length + [0] * (max_emb_length - length)
                for length in emb_lengths
            ]
        )

        return {
            'inputs_embeds': embeddings,
            'attention_mask': attention_mask,
        }

    def inverse_embed(self, embedded):
        bert_embedding = self.embedder.model.embeddings.word_embeddings

        # cosine distance
        #res = torch.matmul(embedded['inputs_embeds'], bert_embedding.weight.data.T)

        #norm_factor = torch.sum(bert_embedding.weight.data ** 2, dim=1).view(1, 1, -1)
        #res = res / norm_factor

        # res_token_list = torch.argmax(res, dim=2).cpu().tolist()

        # euclidean distance
        embedding_weights = bert_embedding.weight.data
        expanded_weights = embedding_weights.view(
            1, embedding_weights.shape[0], embedding_weights.shape[1])
        expanded_weights = expanded_weights.expand(
            embedded['inputs_embeds'].shape[0], embedding_weights.shape[0], embedding_weights.shape[1])

        res = torch.cdist(embedded['inputs_embeds'], expanded_weights)

        res_token_list = torch.argmin(res, dim=2).cpu().tolist()

        res_sentences = []

        for i, item in enumerate(res_token_list):
            res_sentences.append(
                re.sub(" ##", "", " ".join(self.embedder.tokenizer.convert_ids_to_tokens(item))))

        return res_sentences, res_token_list

    def sanitize(self, sentences):
        return self.inverse_embed(self.embed(sentences))[0]


    def save(self, path):
        save_state = {}
        save_state['model_state_dict'] = self.model.state_dict()
        save_state['model_class'] = self.model_class
        save_state['model_params'] = self.model.params
        save_state['char_vocab'] = self.char_tokenizer.vocab
        torch.save(save_state, path)


DEFAULT_LSTM_PARAMS = {
    'word_emb_size': WORD_EMB_SIZE,
    'char_emb_size': CHAR_EMB_SIZE,
    'num_tokens': NUM_TOKENS,
    'hidden_size': HIDDEN_SIZE,
    'num_layers': NUM_LAYERS,
}

DEFAULT_CNN_PARAMS = {
    'word_emb_size': WORD_EMB_SIZE,
    'char_emb_size': CHAR_EMB_SIZE,
    'num_tokens': NUM_TOKENS,
    'hidden_size': HIDDEN_SIZE,
    'num_layers': NUM_LAYERS,
    'kernel_size': CNN_KERNEL_SIZE,
}

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    runner = ExperimentRunner(device,
        objective_model_name='artemis13fowl/bert-base-uncased-imdb',
        objective_tokenizer_name='bert-base-uncased',
        model_class='LSTMModel',
        model_params=DEFAULT_LSTM_PARAMS)
    runner.train(NUM_SENTENCES)

    test_sentences = [
      "my hovercraft is full of eels!",
      "common sense is the least common of all the senses",
      "common sense is the least common of all the senses ",
      " c0mmon s3nse 1s the l3@st comm0n of a|| th3 sens3s ",
      "common sense is the least com mon of all the senses ",
      "my hovercra ft is full of eels! ",
    ]

    sanitized = runner.sanitize(test_sentences)

    for i, item in enumerate(sanitized):
        print("Original sentence: {}".format(test_sentences[i]))
        print("Reconstructed    : {}".format(item))
