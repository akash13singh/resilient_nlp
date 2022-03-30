import json
import sys
import torch

from runner import ExperimentRunner

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert(len(sys.argv) >= 3)

    config_name = sys.argv[1]
    output_name = sys.argv[2]

    config = json.load(open(config_name))

    runner = ExperimentRunner(device,
        objective_model_name=config['objective_model_name'],
        objective_tokenizer_name=config['objective_tokenizer_name'],
        model_class=config['model_class'],
        model_params=config['model_params'],
        perturber_class=config['perturber_class'],
        perturber_params=config['perturber_params'])
    runner.train(config['num_epochs'], config['num_train_sentences'],
                 num_eval_sentences=config['num_eval_sentences'], lr=config['lr'])

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

    runner.save(output_name)
