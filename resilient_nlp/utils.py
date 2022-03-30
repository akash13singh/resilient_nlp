import string                              # for string operations
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk import tokenize   # module for tokenizing strings


def preprocess(sent):
    stopwords_english = stopwords.words('english')
    clean_sent = []
    tokenizer = tokenize.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    stemmer = PorterStemmer()
    sent = tokenizer.tokenize(sent, )
    # sent = tokenize.wordpunct_tokenize(sent.lower())

    for word in sent:
        if (word not in stopwords_english and word not in string.punctuation):  # remove stopwords, punctuation
            # word = stemmer.stem(word)
            clean_sent.append(word)
    return clean_sent