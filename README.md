# resilient_nlp

## Instructions for re-creating finetuned BERT models

The finetuned transformer models are available in the output/huggingface directory.
 
To re-train them locally, execute the following notebooks (available in the notebooks folder):
- imdb_finetune.ipynb (IMDb BERT)
- imdb_finetune_roberta.ipynb (IMDb RoBERTa)
- imdb_finetune_all_pert.ipynb (IMDb BERT, data augmentation)
- sst5_finetune_bert.ipynb (SST-5 BERT)
- sst5_finetune_roberta.ipynb (SST-5 RoBERTa)
- multi_model_finetune.ipynb (Lyft-2 and Lyft-5 for BERT and RoBERTa)

## Dataset

The IMDb dataset split used for finetuning and evaluating IMDb is available in output/huggingface/imdb . For other tasks we are using default HuggingFace datasets.

## MockingBERT models

The trained MockingBERT models can be found here:
```
https://resilient-nlp.s3.us-west-2.amazonaws.com/bert_2m_lstm_all_pert_finetuned_imdb.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/bert_32k_lstm_all_pert_finetuned_imdb_100ep.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/bert_64k_cnn_no_whitespace_pert_finetuned_imdb.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/bert_64k_lstm_all_pert_finetuned_imdb.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/bert_64k_lstm_all_pert_finetuned_sst_bin.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/bert_64k_lstm_all_pert_finetuned_sst.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/bert_64k_lstm_all_pert_finetuned_yelp_bin.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/bert_64k_lstm_all_pert_finetuned_yelp_full.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/bert_64k_lstm_all_pert_vanilla.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/bert_64k_lstm_clean_finetuned_imdb.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/bert_64k_lstm_clean_vanilla.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/bert_64k_lstm_no_whitespace_pert_finetuned_imdb.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/bert_64k_lstm_no_whitespace_pert_vanilla.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/bert_64k_rnn_all_pert_finetuned_imdb.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/roberta_64k_lstm_all_pert_finetuned_imdb.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/roberta_64k_lstm_all_pert_finetuned_sst_bin.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/roberta_64k_lstm_all_pert_finetuned_sst.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/roberta_64k_lstm_all_pert_finetuned_yelp_bin.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/roberta_64k_lstm_all_pert_finetuned_yelp_full.pth
https://resilient-nlp.s3.us-west-2.amazonaws.com/roberta_64k_lstm_all_pert_vanilla.pth
```

For evaluation, they need to be placed in the 'output' directory of the 'resilient_nlp' repository.
To re-train, execute 'train_all_configs.sh' (this can take a few hours on a GPU machine).

## RobEn models

To run the evaluation, you will need first download the files used by RobEn: 'vocab100000_ed1.pkl' and 'vocab100000_ed1_gamma0.3.pkl'. They can be found in this worksheet: https://worksheets.codalab.org/worksheets/0x8fc01c7fc2b742fdb29c05669f0ad7d2

The direct links are:
https://worksheets.codalab.org/rest/bundles/0xd5f5ee6dcd314e6d82b828c2b318f71e/contents/blob/clusterers/vocab100000_ed1.pkl
https://worksheets.codalab.org/rest/bundles/0x6ba29d12a7f9450283b6584f818cd9ee/contents/blob/clusterers/vocab100000_ed1_gamma0.3.pkl

These files need to be placed in the root directory of the repository.

## Evaluation

To execute the model evaluation, run the 'multi_model_eval.ipybp' notebook. It may take more than 10 hours on an NVidia Tesla V100. Notebook evaluation results are available in the 'output' directory in this repository.

