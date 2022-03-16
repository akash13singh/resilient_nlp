import random
import torch

class NullPerturber:
    """A "perturber" that does nothing :)"""
    def perturb(self, inputs, masks, embeddings):
        return inputs, masks, embeddings


class ToyPerturber:
    """A limited perturber that can split long words and replace some chars.

    Mainly useful for testing, not meant as a real adversarial generator.
    """

    def __init__(self,
                 split_long_words_prob=0.20,
                 replace_letters_prob=0.20,
                 start_char_present=False,
                 end_char_present=False):
        self.split_long_words_prob = split_long_words_prob
        self.replace_letters_prob = replace_letters_prob
        self.start_char_present = start_char_present
        self.end_char_present = end_char_present

    def perturb(self, inputs, masks, embeddings):
        new_inputs = []
        new_masks = []
        new_embeddings = []

        leading_offset = int(self.start_char_present)

        for i in range(len(inputs)):
            # add whitespace to long words
            temp_output = ""
            cur_word = ""
            cur_word_embedding = []
            cur_word_mask = []
            input = inputs[i] + " "
            temp_mask = []
            temp_embedding = []

            if self.start_char_present:
                temp_mask.append(masks[i][0])
                temp_embedding.append(embeddings[i][0])

            for j, char in enumerate(input):
                if char.isspace():
                    if len(cur_word) >= 7 and \
                            random.random() < self.split_long_words_prob:
                        split = random.randint(3, len(cur_word) - 3)
                        temp_output += cur_word[:split]
                        temp_output += " "
                        temp_output += cur_word[split:]

                        for k in range(split):
                            temp_mask.append(cur_word_mask[k])
                            temp_embedding.append(cur_word_embedding[k])

                        temp_mask.append(0.0)
                        temp_embedding.append(
                            torch.zeros((embeddings.shape[2]), dtype=torch.float))

                        for k in range(split, len(cur_word)):
                            temp_mask.append(cur_word_mask[k])
                            temp_embedding.append(cur_word_embedding[k])
                    else:
                        temp_output += cur_word

                        for k in range(len(cur_word)):
                            temp_mask.append(cur_word_mask[k])
                            temp_embedding.append(cur_word_embedding[k])

                    cur_word = ""
                    cur_word_mask = []
                    cur_word_embedding = []

                    if j < len(input) - 1:
                        temp_output += char
                        temp_mask.append(masks[i][leading_offset + j])
                        temp_embedding.append(embeddings[i][leading_offset + j])
                else:
                    cur_word += char
                    cur_word_mask.append(masks[i][leading_offset + j])
                    cur_word_embedding.append(embeddings[i][leading_offset + j])

            if self.end_char_present:
                temp_mask.append(masks[i][-1])
                temp_embedding.append(embeddings[i][-1])

            input = temp_output

            substs = {
                'a': [ '@' ],
                'l': [ 'I', '1', '|' ],
                'o': [ '0' ],
                'O': [ '0' ],
                'i': [ 'l', '1' ],
                'e': [ '3' ],
                ' ': [ '\t', '_' ],
            }

            temp_output = ""
            for char in input:
                if char in substs and \
                        random.random() < self.replace_letters_prob:
                    temp_output += random.choice(substs[char])
                else:
                    temp_output += char

            new_inputs.append(temp_output)
            new_masks.append(temp_mask)
            new_embeddings.append(temp_embedding)

        max_length = max([len(mask) for mask in new_masks])
        out_masks = torch.zeros((len(new_inputs), max_length), dtype=torch.float)
        out_embeddings = torch.zeros(
            (len(new_inputs), max_length, embeddings.shape[2]), dtype=torch.float)

        for batch in range(len(new_masks)):
            for idx in range(len(new_masks[batch])):
                out_masks[batch][idx] = new_masks[batch][idx]
                out_embeddings[batch][idx] = new_embeddings[batch][idx]

        return new_inputs, out_masks, out_embeddings
