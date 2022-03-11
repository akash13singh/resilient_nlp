
class CharTokenizer:
    def __init__(self,
                 initial_vocab=[ 'UNK' ],
                 extend_vocab=True,
                 max_vocab=None,
                 unk_index=0):

        self.vocab = initial_vocab[:]
        self.extend_vocab = extend_vocab
        self.max_vocab = max_vocab
        self.unk_index = unk_index
        self.vocab_map = { c: index for index, c in enumerate(self.vocab) }

    def tokenize(self, inputs):
        result = []
        for input in inputs:
            tokens = []
            for c in input:
                if c in self.vocab_map:
                    tokens.append(self.vocab_map[c])
                elif self.extend_vocab and \
                        (self.max_vocab is None or
                        len(self.vocab) < self.max_vocab):
                    # extend vocabulary
                    tokens.append(len(self.vocab))
                    self.vocab_map[c] = len(self.vocab)
                    self.vocab.append(c)
                else:
                    tokens.append(unk_index)
            result.append(tokens)

        return result
