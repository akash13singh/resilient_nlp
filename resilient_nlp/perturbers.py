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

    def perturb(self, inputs, masks=None, embeddings=None):
        new_inputs = []
        new_masks = []
        new_embeddings = []

        leading_offset = int(self.start_char_present)

        for i in range(len(inputs)):
            # add whitespace to long words
            temp_output = ""
            cur_word = ""
            input = inputs[i] + " "
            if masks is not None:
                cur_word_mask = []
                temp_mask = []
            cur_word_embedding = []
            temp_embedding = []

            if self.start_char_present:
                if masks is not None:
                    temp_mask.append(masks[i][0])
                if embeddings is not None:
                    temp_embedding.append(embeddings[i][0])

            for j, char in enumerate(input):
                if not char.isalpha():
                    if len(cur_word) >= 7 and \
                            random.random() < self.split_long_words_prob:
                        split = random.randint(3, len(cur_word) - 3)
                        temp_output += cur_word[:split]
                        temp_output += " "
                        temp_output += cur_word[split:]

                        for k in range(split):
                            if masks is not None:
                                temp_mask.append(cur_word_mask[k])
                            if embeddings is not None:
                                temp_embedding.append(cur_word_embedding[k])

                        if masks is not None:
                            temp_mask.append(0.0)
                        if embeddings is not None:
                            temp_embedding.append(
                                torch.zeros((embeddings.shape[2]), dtype=torch.float))

                        for k in range(split, len(cur_word)):
                            if masks is not None:
                                temp_mask.append(cur_word_mask[k])
                            if embeddings is not None:
                                temp_embedding.append(cur_word_embedding[k])
                    else:
                        temp_output += cur_word

                        for k in range(len(cur_word)):
                            if masks is not None:
                                temp_mask.append(cur_word_mask[k])
                            if embeddings is not None:
                                temp_embedding.append(cur_word_embedding[k])

                    cur_word = ""
                    if masks is not None:
                        cur_word_mask = []
                    if embeddings is not None:
                        cur_word_embedding = []

                    if j < len(input) - 1:
                        temp_output += char
                        if masks is not None:
                            temp_mask.append(masks[i][leading_offset + j])
                        if embeddings is not None:
                            temp_embedding.append(embeddings[i][leading_offset + j])
                else:
                    cur_word += char
                    if masks is not None:
                        cur_word_mask.append(masks[i][leading_offset + j])
                    if embeddings is not None:
                        cur_word_embedding.append(embeddings[i][leading_offset + j])

            if self.end_char_present:
                if masks is not None:
                    temp_mask.append(masks[i][-1])
                if embeddings is not None:
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
            if masks is not None:
                new_masks.append(temp_mask)
            if embeddings is not None:
                new_embeddings.append(temp_embedding)

        result = [ new_inputs ]

        if masks is not None:
            max_length = max([len(mask) for mask in new_masks])
            out_masks = torch.zeros((len(new_inputs), max_length), dtype=torch.float)
            if embeddings is not None:
                out_embeddings = torch.zeros(
                    (len(new_inputs), max_length, embeddings.shape[2]), dtype=torch.float)

            for batch in range(len(new_masks)):
                for idx in range(len(new_masks[batch])):
                    out_masks[batch][idx] = new_masks[batch][idx]
                    if embeddings is not None:
                        out_embeddings[batch][idx] = new_embeddings[batch][idx]
            result.append(out_masks)
            if embeddings is not None:
                result.append(out_embeddings)

        return tuple(result)


class WordScramblerPerturber:
    """A perturber that can add, delete or swap internal letters."""

    def __init__(self,
                 perturb_prob=0.20,
                 weight_add=1.0,
                 weight_drop=1.0,
                 weight_swap=1.0,
                 weight_split_word=1.0,
                 weight_merge_words=1.0,
                 start_char_present=False,
                 end_char_present=False):
        self.perturb_prob = perturb_prob
        self.start_char_present = start_char_present
        self.end_char_present = end_char_present
        perturb_map = {
            'add': weight_add,
            'drop': weight_drop,
            'swap': weight_swap,
            'split': weight_split_word,
            'merge': weight_merge_words,
        }

        self.perturb_options = [ k for k, v in perturb_map.items() if v > 0.0 ]
        self.perturb_weights = [ v for k, v in perturb_map.items() if v > 0.0 ]

    def perturb(self, inputs, masks=None, embeddings=None):
        new_inputs = []
        new_masks = []
        new_embeddings = []

        leading_offset = int(self.start_char_present)

        for i in range(len(inputs)):
            temp_output = ""
            cur_word = ""
            input = inputs[i] + " "
            if masks is not None:
                cur_word_mask = []
                temp_mask = []
            if embeddings is not None:
                cur_word_embedding = []
                temp_embedding = []

            if self.start_char_present:
                if masks is not None:
                    temp_mask.append(masks[i][0])
                if embeddings is not None:
                    temp_embedding.append(embeddings[i][0])

            for j, char in enumerate(input):
                if not char.isalpha():
                    should_perturb = random.random() < self.perturb_prob

                    perturbation_applied = False
                    ignore_char = False

                    order = random.choices(self.perturb_options,
                        weights=self.perturb_weights,
                        k=len(self.perturb_options)) if should_perturb else []

                    for op in order:
                        if op == 'swap' and len(cur_word) >= 4:
                            # Swap chars
                            pos = random.randint(1, len(cur_word) - 3)

                            temp_output += cur_word[:pos]
                            for k in range(pos):
                                if masks is not None:
                                    temp_mask.append(cur_word_mask[k])
                                if embeddings is not None:
                                    temp_embedding.append(cur_word_embedding[k])

                            # Revisit - where should the subword boundary be
                            # exactly? :(
                            temp_output += cur_word[pos+1]
                            if masks is not None:
                                temp_mask.append(cur_word_mask[pos])
                            if embeddings is not None:
                                temp_embedding.append(cur_word_embedding[pos])
                            temp_output += cur_word[pos]
                            if masks is not None:
                                temp_mask.append(cur_word_mask[pos+1])
                            if embeddings is not None:
                                temp_embedding.append(cur_word_embedding[pos+1])

                            temp_output += cur_word[pos+2:]
                            for k in range(pos+2, len(cur_word)):
                                if masks is not None:
                                    temp_mask.append(cur_word_mask[k])
                                if embeddings is not None:
                                    temp_embedding.append(cur_word_embedding[k])

                            perturbation_applied = True
                            break

                        elif op == 'drop' and len(cur_word) >= 3:
                            # Remove char
                            pos = random.randint(1, len(cur_word) - 2)

                            temp_output += cur_word[:pos]
                            for k in range(pos):
                                if masks is not None:
                                    temp_mask.append(cur_word_mask[k])
                                if embeddings is not None:
                                    temp_embedding.append(cur_word_embedding[k])

                            temp_output += cur_word[pos+1]
                            if masks is not None:
                                if cur_word_mask[pos+1] == 1.0:
                                    if cur_word_mask[pos] == 1.0:
                                        # This means we have 2 consecutive letters that
                                        # are boundaries and we're deleting the first
                                        # one.
                                        #print('mask conflict')
                                        pass
                                    temp_mask.append(cur_word_mask[pos+1])
                                    if embeddings is not None:
                                        temp_embedding.append(cur_word_embedding[pos+1])
                                else:
                                    temp_mask.append(cur_word_mask[pos])
                                    if embeddings is not None:
                                        temp_embedding.append(cur_word_embedding[pos])

                            temp_output += cur_word[pos+2:]
                            for k in range(pos+2, len(cur_word)):
                                if masks is not None:
                                    temp_mask.append(cur_word_mask[k])
                                if embeddings is not None:
                                    temp_embedding.append(cur_word_embedding[k])

                            perturbation_applied = True
                            break

                        elif op == 'add' and len(cur_word) >= 2:
                            # Insert char
                            pos = random.randint(1, len(cur_word) - 1)
                            new_char = chr(ord('a') + random.randint(0, 25))

                            temp_output += cur_word[:pos]
                            for k in range(pos):
                                if masks is not None:
                                    temp_mask.append(cur_word_mask[k])
                                if embeddings is not None:
                                    temp_embedding.append(cur_word_embedding[k])

                            temp_output += new_char
                            if masks is not None:
                                temp_mask.append(0.0)
                            if embeddings is not None:
                                temp_embedding.append(
                                    torch.zeros((embeddings.shape[2]), dtype=torch.float))

                            temp_output += cur_word[pos:]
                            for k in range(pos, len(cur_word)):
                                if masks is not None:
                                    temp_mask.append(cur_word_mask[k])
                                if embeddings is not None:
                                    temp_embedding.append(cur_word_embedding[k])

                            perturbation_applied = True
                            break

                        elif op == 'split' and len(cur_word) >= 6:
                            split = random.randint(3, len(cur_word) - 3)
                            temp_output += cur_word[:split]

                            for k in range(split):
                                if masks is not None:
                                    temp_mask.append(cur_word_mask[k])
                                if embeddings is not None:
                                    temp_embedding.append(cur_word_embedding[k])

                            temp_output += " "

                            if masks is not None:
                                temp_mask.append(0.0)
                            if embeddings is not None:
                                temp_embedding.append(
                                    torch.zeros((embeddings.shape[2]), dtype=torch.float))

                            temp_output += cur_word[split:]

                            for k in range(split, len(cur_word)):
                                if masks is not None:
                                    temp_mask.append(cur_word_mask[k])
                                if embeddings is not None:
                                    temp_embedding.append(cur_word_embedding[k])

                            perturbation_applied = True
                            break

                        elif op == 'merge':
                            if (j >= 1 and
                                j < len(input) - 1 and
                                input[j].isspace() and
                                input[j-1].isalpha() and
                                input[j+1].isalpha()):
                                # We can merge the words by ignoring the char
                                ignore_char = True
                                perturbation_applied = True
                                break

                    if not perturbation_applied:
                        temp_output += cur_word

                        for k in range(len(cur_word)):
                            if masks is not None:
                                temp_mask.append(cur_word_mask[k])
                            if embeddings is not None:
                                temp_embedding.append(cur_word_embedding[k])

                    if not ignore_char:
                        cur_word = ""
                        if masks is not None:
                            cur_word_mask = []
                        if embeddings is not None:
                            cur_word_embedding = []

                        if j < len(input) - 1:
                            temp_output += char
                            if masks is not None:
                                temp_mask.append(masks[i][leading_offset + j])
                            if embeddings is not None:
                                temp_embedding.append(embeddings[i][leading_offset + j])
                else:
                    cur_word += char
                    if masks is not None:
                        cur_word_mask.append(masks[i][leading_offset + j])
                    if embeddings is not None:
                        cur_word_embedding.append(embeddings[i][leading_offset + j])

            if self.end_char_present:
                if masks is not None:
                    temp_mask.append(masks[i][-1])
                if embeddings is not None:
                    temp_embedding.append(embeddings[i][-1])

            new_inputs.append(temp_output)
            if masks is not None:
                new_masks.append(temp_mask)
            if embeddings is not None:
                new_embeddings.append(temp_embedding)

        result = [ new_inputs ]

        if masks is not None:
            max_length = max([len(mask) for mask in new_masks])
            out_masks = torch.zeros((len(new_inputs), max_length), dtype=torch.float)
            if embeddings is not None:
                out_embeddings = torch.zeros(
                    (len(new_inputs), max_length, embeddings.shape[2]), dtype=torch.float)

            for batch in range(len(new_masks)):
                for idx in range(len(new_masks[batch])):
                    out_masks[batch][idx] = new_masks[batch][idx]
                    if embeddings is not None:
                        out_embeddings[batch][idx] = new_embeddings[batch][idx]
            result.append(out_masks)
            if embeddings is not None:
                result.append(out_embeddings)

        return tuple(result)
