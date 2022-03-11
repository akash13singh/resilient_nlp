import torch
from transformers import BertTokenizerFast, BertModel

class BertEmbedder:
    def __init__(self,
                 model_name='bert-base-cased',
                 add_special_tokens=True,
                 per_character_embedding=False):
        self.model_name = model_name
        self.add_special_tokens = add_special_tokens
        self.per_character_embedding = per_character_embedding
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name,
            add_special_tokens=add_special_tokens)
        self.model = BertModel.from_pretrained(model_name)

    def embed(self, inputs):
        tokenized = self.tokenizer(inputs, return_tensors='pt',
            return_offsets_mapping=True, padding=True, return_length=True)
        embedded = self.model.embeddings.word_embeddings(tokenized['input_ids'])
        embedding_dim = embedded.shape[2]

        input_lengths = [ len(input) for input in inputs ]

        lengths = torch.sum(tokenized['attention_mask'], dim=1).detach()

        if self.per_character_embedding:
            max_length = max(input_lengths)
            result = torch.zeros(
                (len(inputs), max_length, embedding_dim), dtype=torch.float)
            masks = torch.zeros((len(inputs), max_length), dtype=torch.float)

            subword_boundaries = tokenized['offset_mapping'][:,:,1]

            # TODO: maybe vectorize?
            for word_idx, boundaries in enumerate(subword_boundaries):
                for token_idx, char_offset in enumerate(boundaries):
                    if char_offset > 0:
                        result[word_idx][char_offset - 1] = \
                            embedded[word_idx][token_idx]
                        masks[word_idx][char_offset - 1] = 1.0
            return {
                'embeddings': result.detach(),
                'masks': masks.detach(),
                'num_tokens': lengths,
                'num_chars': input_lengths,
            }
        else:
            return {
                'embeddings': embedded.detach(),
                'num_tokens': lengths,
                'num_chars': input_lengths,
            }
