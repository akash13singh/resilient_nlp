import datasets
import random
import re

class BookCorpus:
   def __init__(self, postprocess=True):
       dataset_name = 'bookcorpus'
       self.dataset = datasets.load_dataset(dataset_name)
       self.postprocess = postprocess

   def all_items(self):
       return self.get_items(range(len(dataset['train'])))

   def sample_items(self, num_samples):
       return self.get_items(
           random.sample(range(len(self.dataset['train'])), num_samples))

   def get_items(self, indices):
       if self.postprocess:
           return [
               self.apply_postprocessing(self.dataset['train'][i]['text'])
               for i in indices
           ]
       else:
           return [ self.dataset['train'][i]['text'] for i in indices ]

   def apply_postprocessing(self, text):
       """A few heuristics for bookcorpus"""

       # Contractions; bookcorpus inserts a space before it
       text = re.sub(" n't ", "n't ", text)
       text = re.sub(" 's ", "'s ", text)
       text = re.sub(" 'll ", "'ll ", text)
       text = re.sub(" 'd ", "'d ", text)
       text = re.sub(" 'm ", "'m ", text)
       text = re.sub(" 're ", "'re ", text)
       text = re.sub(" 've ", "'ve ", text)

       # spaces before punctuation
       text = re.sub(" \\.", ".", text)
       text = re.sub(" \\?", "?", text)
       text = re.sub(" \\!", "!", text)
       text = re.sub(" , ", ", ", text)
       text = re.sub(" : ", ": ", text)
       text = re.sub(" ; ", "; ", text)

       # double quotes
       text = re.sub("`` ", "“", text)
       text = re.sub(" ''", "”", text)

       return text
