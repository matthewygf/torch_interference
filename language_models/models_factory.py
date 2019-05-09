import torch
from torch.nn import LSTM
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention import DotProductAttention

from language_models import lstm_tagger

seq2seq_models = {
  'lstm': True,
  'transformer': True
}

seq2seq_model_fn = {
  'lstm': LSTM,
  'transformer': StackedSelfAttentionEncoder
}

seq2seq_wrapped = {
  'lstm': PytorchSeq2SeqWrapper,
  'transformer': None
} 

# NOTE: Necessary because different dataset has different dataset reader,
# i am too lazy to change the default dataset reader
dataset_model = {
  'debug': lstm_tagger.SimpleLstmPosTagger,
  'ud-eng': lstm_tagger.WordsLstmPosTagger,
  'nc_zhen': SimpleSeq2Seq
}

# NOTE: Necessary because different dataset has different dataset reader,
# i am too lazy to change the default dataset reader
output_feature_keys = {
  'debug': 'labels',
  'ud-eng': 'pos',
  'nc_zhen': 'target_tokens'
}

def get_model_fn(model_name, embeddings, vocab, input_dims=128, hidden_dims=128, dataset_name='debug', **kwargs):
  is_seq2seq = seq2seq_models.get(model_name, False)
  
  wrapped = None

  # TODO: Factory or something else.
  encoder_args = {
    'lstm':
      [
        input_dims,
        hidden_dims,
      ],
      'transformer':
      [
        input_dims,
        hidden_dims,
        input_dims,
        input_dims,
        1,
        3,
        # TODO: add more if you need.
      ]
  }

  encoder_arg = encoder_args[model_name]

  if is_seq2seq:
    # default use LSTM
    wrapped_fn = seq2seq_wrapped[model_name]
    seq_model_fn = seq2seq_model_fn[model_name]
    
    if wrapped_fn is not None:
      wrapped = wrapped_fn(seq_model_fn(*encoder_arg, **kwargs))
    else:
      wrapped = seq_model_fn(*encoder_arg)
  
  # TODO: get a factory or something
  if 'nc_zhen' not in dataset_name:
    model_args = {
      'word_embeddings': embeddings,
      'encoder': wrapped,
      'vocab': vocab,
      'output_feature_key': output_feature_keys[dataset_name]
    }
  else:
    model_args = {
      'vocab': vocab,
      'source_embedder': embeddings,
      'encoder': wrapped,
      'max_decoding_steps': 20, # arbitrary
      'attention': DotProductAttention(), # arbitrary
      'beam_size': 8, #arbitrary
    }

  model = dataset_model[dataset_name](**model_args)
  return output_feature_keys[dataset_name], model