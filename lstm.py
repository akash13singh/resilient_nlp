import json
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import sys

from resilient_nlp.char_tokenizer import CharTokenizer
from resilient_nlp.corpora import BookCorpus
from resilient_nlp.embedders import BertEmbedder

# TODO: These shouldn't be consumed directly by LSTMModel
NUM_TOKENS = 1000
NUM_SENTENCES = 2000
CHAR_EMB_SIZE = 768
HIDDEN_SIZE = 768
NUM_LSTM_LAYERS = 2

class LSTMModel(nn.Module):
    def __init__(self, word_emb_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(NUM_TOKENS, CHAR_EMB_SIZE)
        self.lstm = nn.LSTM(input_size=CHAR_EMB_SIZE, hidden_size=HIDDEN_SIZE,
                            batch_first=True, bidirectional=True,
                            num_layers=NUM_LSTM_LAYERS)
        self.dense = nn.Linear(2 * HIDDEN_SIZE, word_emb_size)
        self.gate = nn.Linear(2 * HIDDEN_SIZE, 1)
        self.gate_activation = nn.Sigmoid()

    def forward(self, X, lengths):
        embedded = self.embedding(X)
        max_length = X.shape[1]
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths,
            batch_first=True, enforce_sorted=False)
        lstm_hidden_packed, lstm_cell_packed = self.lstm(packed)
        lstm_hidden, _ = nn.utils.rnn.pad_packed_sequence(lstm_hidden_packed,
            batch_first=True, total_length=max_length)
        dense_result = self.dense(lstm_hidden)
        gate_result = self.gate_activation(self.gate(lstm_hidden))

        return dense_result, gate_result.squeeze(2)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    corpus = BookCorpus()
    embedder = BertEmbedder(per_character_embedding=True)
    char_tokenizer = CharTokenizer(max_vocab=NUM_TOKENS)

    # Ensure consistent sample
    random.seed(11)
    sentences = corpus.sample_items(NUM_SENTENCES)
    sentence_tokens = char_tokenizer.tokenize(sentences)

    num_samples = len(sentences)

    model = LSTMModel(768).to(device)
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 64
    num_batches = math.ceil(len(sentences) / batch_size)
    num_epochs = 500

    for epoch in range(num_epochs):
        for i in range(num_batches):
            bs = i*batch_size
            be = bs + batch_size
            num_examples_in_batch = min(batch_size, num_samples - bs)

            max_length = max([len(tokens) for tokens in sentence_tokens[bs:be]])
            batch_X = torch.zeros((num_examples_in_batch, max_length),
                dtype=torch.int)

            for j, tokens in enumerate(sentence_tokens[bs:be]):
                # There HAS to be a nicer way to do this... :(
                batch_X[j,:len(tokens)] = torch.IntTensor(tokens)

            batch_X = batch_X.to(device)

            bert_embeddings = embedder.embed(sentences[bs:be])
            batch_Y = bert_embeddings['embeddings'].to(device)
            batch_Y_masks = bert_embeddings['masks'].to(device)

            batch_lengths = bert_embeddings['num_chars']
            model.zero_grad()
            model.train()

            batch_preds, batch_pred_masks = model(batch_X, batch_lengths)
            batch_emb_loss = \
                loss(batch_preds * batch_Y_masks.unsqueeze(2), batch_Y)
            batch_emb_variance = \
                float(torch.mean(torch.std(batch_Y, (0, 1), unbiased=False) ** 2))
            batch_mask_loss = loss(batch_pred_masks,  batch_Y_masks)
            batch_mask_variance = \
                float(torch.std(batch_Y_masks, unbiased=False) ** 2)

            loss_multiplier = float(batch_mask_loss) / float(batch_emb_loss)
            batch_combined_loss = batch_emb_loss * loss_multiplier + batch_mask_loss

            batch_correct = 0
            batch_total = 0
            batch_positive = 0

            for idx, length in enumerate(batch_lengths):
                batch_total += length
                batch_correct += int(sum(torch.isclose(torch.round(batch_pred_masks[idx][:length]), batch_Y_masks[idx][:length])))
                batch_positive += int(sum(batch_Y_masks[idx][:length]))

            mask_accuracy = batch_correct / batch_total

            batch_combined_loss.backward()
            optimizer.step()

            print("%04d-%04d: embedding loss: %f, mask loss: %f, mask accuracy: %f (%f positive examples in batch)" %
                (epoch, i, batch_emb_loss / batch_emb_variance, batch_mask_loss / batch_mask_variance, mask_accuracy, batch_positive / batch_total))
