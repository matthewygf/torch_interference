import torch

from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper

from language_models import lstm_tagger

seq2seq_models = {
  'lstm': True
}

# NOTE: Necessary because different dataset has different dataset reader,
# i am too lazy to change the default dataset reader
dataset_model = {
  'debug': lstm_tagger.SimpleLstmPosTagger,
  'ud-eng': lstm_tagger.WordsLstmPosTagger
}

# NOTE: Necessary because different dataset has different dataset reader,
# i am too lazy to change the default dataset reader
output_feature_keys = {
  'debug': 'labels',
  'ud-eng': 'pos'
}

def get_model_fn(model_name, embeddings, vocab, input_dims=128, hidden_dims=128, dataset_name='debug', **kwargs):
  is_seq2seq = seq2seq_models.get(model_name, False)
  
  wrapped = None
  if is_seq2seq:
    # default use LSTM
    wrapped = PytorchSeq2SeqWrapper(torch.nn.LSTM(input_dims, hidden_dims, **kwargs))
  
  model_args = {
    'word_embeddings': embeddings,
    'encoder': wrapped,
    'vocab': vocab,
    'output_feature_key': output_feature_keys[dataset_name]
  }

  model = dataset_model[dataset_name](**model_args)
  return output_feature_keys[dataset_name], model