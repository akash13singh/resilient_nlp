import torch
import torch.nn as nn
from transformers import BertModel
from transformers.modeling_outputs import TokenClassifierOutput

class BaseEmbeddingModel(nn.Module):
    def __init__(self, num_tokens):
        super(BaseEmbeddingModel, self).__init__()
        self.num_tokens = num_tokens

    def _copy_params(self, actual_locals):
        self.params = {
            k: v for k, v in actual_locals.items() if k not in ['self', '__class__' ]
        }

    def forward(self, X, lengths):
        raise NotImplementedError

    def predict_embeddings(self, X, lengths, start_token=None, end_token=None,
            pad_token=None, max_tokens=None):
        dense_result, gate_result = self.forward(X, lengths)

        gate_result = torch.round(gate_result).int().unsqueeze(2)

        leading_offset = int(start_token is not None)
        trailing_offset = int(end_token is not None)

        token_nums = torch.sum(gate_result, dim=(1, 2)) + leading_offset + trailing_offset
        max_actual_tokens = int(torch.max(token_nums))
        if max_tokens is None:
            max_tokens = max_actual_tokens
        else:
            max_tokens = min(max_tokens, max_actual_tokens)
            token_nums = torch.clamp(token_nums, max=max_tokens)

        result = torch.zeros((dense_result.shape[0], max_tokens, dense_result.shape[2]),
            dtype=torch.float, device=dense_result.device)

        for i in range(dense_result.shape[0]):
            token_idx = 0
            if start_token is not None:
                result[i][token_idx] = start_token
                token_idx += 1

            for j in range(dense_result.shape[1]):
                if gate_result[i][j][0] == 1.0:
                    if token_idx < max_tokens - trailing_offset:
                        result[i][token_idx] = dense_result[i][j]
                        token_idx += 1
            if end_token is not None:
                assert(token_idx < max_tokens)
                result[i][token_idx] = end_token
                token_idx += 1
        while pad_token is not None and token_idx < max_tokens:
            result[i][token_idx] = pad_token
            token_idx += 1

        return result.detach(), token_nums.detach()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)))


class LSTMModel(BaseEmbeddingModel):
    def __init__(self,
                 word_emb_size,
                 char_emb_size,
                 num_tokens,
                 hidden_size,
                 num_layers):
        self._copy_params(locals())
        super(LSTMModel, self).__init__(num_tokens)
        self.word_emb_size = word_emb_size
        self.char_emb_size = char_emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_tokens, char_emb_size)
        self.lstm = nn.LSTM(input_size=char_emb_size, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True,
                            num_layers=num_layers)
        self.dense = nn.Linear(2 * hidden_size, word_emb_size)
        self.gate = nn.Linear(2 * hidden_size, 1)
        self.gate_activation = nn.Sigmoid()

    def forward(self, X, lengths):
        embedded = self.embedding(X)
        max_length = X.shape[1]
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths,
            batch_first=True, enforce_sorted=False)
        lstm_hidden_packed, lstm_cell_packed = self.lstm(packed)
        lstm_hidden, _ = nn.utils.rnn.pad_packed_sequence(lstm_hidden_packed,
            batch_first=True, total_length=max_length)
        dense_result = nn.Tanh()(self.dense(lstm_hidden))
        gate_result = self.gate_activation(self.gate(lstm_hidden))

        return dense_result, gate_result.squeeze(2)


class CNNModel(BaseEmbeddingModel):
    def __init__(self,
                 word_emb_size,
                 char_emb_size,
                 num_tokens,
                 hidden_size,
                 kernel_size,
                 num_layers):
        self._copy_params(locals())
        super(CNNModel, self).__init__(num_tokens)
        self.word_emb_size = word_emb_size
        self.char_emb_size = char_emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        self.embedding = nn.Embedding(num_tokens, char_emb_size)
        self.cnns = nn.ModuleList()
        self.update_gates = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = char_emb_size if i == 0 else hidden_size
            self.cnns.append(
                nn.Conv1d(in_channels=in_dim, out_channels=hidden_size,
                          kernel_size=kernel_size, padding=int((kernel_size - 1) / 2)))
            self.update_gates.append(
                nn.Linear(hidden_size, hidden_size))

        self.dense = nn.Linear(hidden_size, word_emb_size)
        self.gate = nn.Linear(hidden_size, 1)
        self.gate_activation = nn.Sigmoid()

    def forward(self, X, lengths):
        embedded = self.embedding(X).permute((0, 2, 1))
        cnn_out = embedded
        for i, cnn in enumerate(self.cnns):
            #cnn_out = cnn_out + nn.Tanh()(cnn(cnn_out))
            tmp = nn.Tanh()(cnn(cnn_out))
            update = nn.Sigmoid()(self.update_gates[i](tmp.permute((0, 2, 1)))).permute((0, 2, 1))
            cnn_out = update * tmp + (1 - update) * cnn_out
        cnn_out = cnn_out.permute((0, 2, 1))
        dense_result = nn.Tanh()(self.dense(cnn_out))
        gate_result = self.gate_activation(self.gate(cnn_out))

        return dense_result, gate_result.squeeze(2)


class BertClassifier(nn.Module):
    def __init__(self, checkpoint, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.model = BertModel.from_pretrained(checkpoint, num_labels=self.n_classes)
        self.dropout = nn.Dropout(0.1)
        self.hidden_dim = self.model.embeddings.word_embeddings.embedding_dim
        self.classifier_layer = nn.Linear(self.hidden_dim, self.n_classes)

    def forward(self, input_ids=None, attention_mask=None, labels=None, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, )
        cls_rep = outputs['last_hidden_state'][:, 0, :]
        cls_rep = self.dropout(cls_rep)
        logits = self.classifier_layer(cls_rep)

        loss = None
        if labels is not None:
            # print(logits.shape, labels.shape)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                     attentions=outputs.attentions)
