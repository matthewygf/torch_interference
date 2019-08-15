from absl import app
from absl import flags

# torch 
import torch
import torch.optim as optim

# allenNLP
from allennlp.common.file_utils import cached_path

# TODO: I am not sure how to use this PennTreeBank yet :/
from allennlp.data.dataset_readers import PennTreeBankConstituencySpanDatasetReader, UniversalDependenciesDatasetReader, Seq2SeqDatasetReader, LanguageModelingReader
from languages_data.wikitext_dataset_reader import WikiTextDatasetReader
from allennlp.data.vocabulary import Vocabulary

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.models import Model

from allennlp.training.trainer import Trainer

from languages_data import pos_data_reader, embeddings_factory, iterators_factory, datasets_factory, preprocessing_factory
from language_models import models_factory, datareader_cfg_factory
from languages_predictors import predictors_factory
from ops_profiler.flop_counter import *

import time
import utils as U
import numpy as np
import os

from urllib.parse import urlparse

import copy

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)

FLAGS = flags.FLAGS
flags.DEFINE_string('model', None, 'The model architecture you want to test')
flags.DEFINE_string('run_name', None, 'The name you want to give to this run')
flags.DEFINE_string('task', None, 'The task the model is trained to do')
flags.DEFINE_string('dataset_dir', 'data/', 'Dataset directory')
flags.DEFINE_string('dataset', 'debug', 'Dataset to use')
flags.DEFINE_string('embeddings', 'basic', 'Embeddings to use')
flags.DEFINE_integer('embeddings_dim', 128, 'Embedding dimension to use')
flags.DEFINE_integer('hiddens_dim', 128, 'Embedding dimension to use')
flags.DEFINE_boolean('use_cuda', False, 'whether to use GPU')
flags.DEFINE_integer('log_interval', 10, 'Batch intervals to log')
flags.DEFINE_integer('batch_size', 16, 'Batch intervals to log')
flags.DEFINE_integer('max_epochs', 1, 'max epoch number to run')
flags.DEFINE_integer('max_vocabs', 100000, 'Maximum number of vocabulary')
flags.DEFINE_string('optimizer', 'adam', 'Gradient descent optimizer')
flags.DEFINE_float('drop_out', 0., 'dropout rate, if it is RNN base: for outputs of each RNN layer except the last layer')
flags.DEFINE_boolean('bidirectional', False, 'if it is RNNbase, whether it becomes bidirectional RNN')
flags.DEFINE_integer('max_len', 40, 'maximum length to generate tokens')
flags.DEFINE_integer('num_layers', 1, 'number of layers of recurrent models')
flags.DEFINE_interger('max_length_sentence', 200, 'maxium length per sentence for the encoder')
flags.DEFINE_bool('profile_only', False, 'Profile the model and exit.')

#TODO: DATA PARALLEL / MODEL PARALLEL

flags.mark_flag_as_required('run_name')
flags.mark_flag_as_required('model')
flags.mark_flag_as_required('task')
flags.mark_flag_as_required('dataset_dir')
flags.mark_flag_as_required('dataset')

data_reader_factory = {
  'debug': pos_data_reader.TaskDataReader,
  'ptb_tree': PennTreeBankConstituencySpanDatasetReader,
  'ud-eng': UniversalDependenciesDatasetReader,
  'nc_zhen': Seq2SeqDatasetReader,
  'wikitext': WikiTextDatasetReader
}

test_sentences = {
  'debug' : 'I am your father',
  'nc_zhen' : 'I am your father',
  'ud-eng': ['I', 'am', 'your', 'father']
}

optimizers_factory = {
  'adam': optim.Adam,
  'sgd': optim.SGD,
  'rmsprop': optim.RMSprop
}

def main(argv):
  del argv

  logger = U.get_logger(__name__+FLAGS.run_name)
  logger.info("run: %s, specified model: %s, dataset: %s", FLAGS.run_name, FLAGS.model, FLAGS.dataset)
  _cudart = U.get_cudart()
  device = torch.device("cuda" if FLAGS.use_cuda else "cpu")
  if _cudart is None:
    logger.warning("No cudart, probably means you do not have cuda on this machine.")

  cfgs = datareader_cfg_factory.get_datareader_configs(FLAGS.dataset)
  if cfgs is not None:
    cfgs.update({'max_length_sentence': FLAGS.max_length_sentence})
    reader = data_reader_factory[FLAGS.dataset](**cfgs)
  else:
    reader = data_reader_factory[FLAGS.dataset]()

  dataset_paths = datasets_factory.get_dataset_paths(FLAGS.dataset)
  # NOTE: check whether we need preprocessing, i.e. machine translation datasets
  train_dataset = None
  validation_dataset = None
  if dataset_paths['train']['preprocess']:
    # TODO: not yet ready. .___.
    preprocessor = preprocessing_factory.get_preprocessor(FLAGS.dataset)
  
  cache_dataset_dir = os.path.join(FLAGS.dataset_dir, FLAGS.dataset)
  # TODO: If there is multiple paths to make one huge dataset, we should do it with the preprocessor
  train_dataset = reader.read(cached_path(dataset_paths['train']['paths'][0], cache_dataset_dir))
  validation_dataset = None
  if dataset_paths['val'] is not None:
    validation_dataset = reader.read(cached_path(dataset_paths['val']['paths'][0], cache_dataset_dir))
  
  vocab = Vocabulary.from_instances(
                train_dataset + validation_dataset, max_vocab_size=FLAGS.max_vocabs)

  #print(vocab.print_statistics())

  embeddings = embeddings_factory.get_embeddings(FLAGS.embeddings, vocab, embedding_dim=FLAGS.embeddings_dim)

  models_args = {
    'model_name': FLAGS.model,
    'embeddings': embeddings,
    'vocab': vocab,
    'input_dims': FLAGS.embeddings_dim,
    'hidden_dims': FLAGS.hiddens_dim,
    'batch_first': True,
    'dataset_name': FLAGS.dataset,
    'dropout': FLAGS.drop_out,
    'bidirectional': FLAGS.bidirectional,
    'max_len': FLAGS.max_len,
    'num_layers': FLAGS.num_layers
  }

  out_feature_key, model = models_factory.get_model_fn(**models_args)
  if FLAGS.profile_only:
    # language
    torch.save(model, FLAGS.run_name+"model.pth")
    return

  model = model.to(device)

  optimizer = optimizers_factory[FLAGS.optimizer](model.parameters(), lr=0.001)

  iterator = iterators_factory.get_iterator(FLAGS.dataset, FLAGS.batch_size)

  iterator.index_with(vocab)
  cuda_device = 0 if FLAGS.use_cuda else -1 # TODO: multi GPU
  trainer = Trainer(model=model,
                    optimizer=optimizer,
                    iterator=iterator,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    num_epochs=FLAGS.max_epochs,
                    log_batch_size_period = 10,
                    cuda_device=cuda_device)

  start_time = time.time()
  try:
    status = None
    if _cudart:
      status =  _cudart.cudaProfilerStart()
    trainer.train()
  finally:
    if status == 0:
      _cudart.cudaProfilerStop()
  final_time = time.time() - start_time
  logger.info("Finished training: ran for %d secs", final_time)

  # TODO: VERY ROUGH.
  if FLAGS.task == 'lm':
    for _ in range(50):
      tokens, _ = model.generate(device, FLAGS.num_layers)
    logger.info("GENERATED WORDS:")
    logger.info(''.join(token.text for token in tokens))
  else:
    predictor = predictors_factory.get_predictors(FLAGS.dataset, model, reader)
    test_tokens_or_sentence = test_sentences[FLAGS.dataset]
    pred_logits = predictor.predict(test_tokens_or_sentence)
    pred_logits_key = predictors_factory.get_logits_key(FLAGS.task)
    if pred_logits_key is not None:
     pred_logits = pred_logits[pred_logits_key]

    if FLAGS.task == 'pos':
      top_ids = np.argmax(pred_logits, axis=-1)
      print([model.vocab.get_token_from_index(i, out_feature_key) for i in top_ids])
    else:
      print(pred_logits)

  final_time = time.time() - start_time
  logger.info("Finished application: ran for %d secs", final_time)
  
if __name__ == "__main__":
  app.run(main)
