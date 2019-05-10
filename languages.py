from absl import app
from absl import flags

# torch 
import torch
import torch.optim as optim

# allenNLP
from allennlp.common.file_utils import cached_path

# TODO: I am not sure how to use this PennTreeBank yet :/
from allennlp.data.dataset_readers import PennTreeBankConstituencySpanDatasetReader, UniversalDependenciesDatasetReader, Seq2SeqDatasetReader
from allennlp.data.vocabulary import Vocabulary

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.models import Model

from allennlp.training.trainer import Trainer

from languages_data import pos_data_reader, embeddings_factory, iterators_factory, datasets_factory, preprocessing_factory
from language_models import models_factory, datareader_cfg_factory
from languages_predictors import predictors_factory

import time
import utils as U
import numpy as np
import os

from urllib.parse import urlparse

import copy

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
  'nc_zhen': Seq2SeqDatasetReader
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
    reader = data_reader_factory[FLAGS.dataset](cfgs)
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

  embeddings = embeddings_factory.get_embeddings(FLAGS.embeddings, vocab, embedding_dim=FLAGS.embeddings_dim)

  models_args = {
    'model_name': FLAGS.model,
    'embeddings': embeddings,
    'vocab': vocab,
    'input_dims': FLAGS.embeddings_dim,
    'hidden_dims': FLAGS.hiddens_dim,
    'batch_first': True,
    'dataset_name': FLAGS.dataset,
  }

  out_feature_key, model = models_factory.get_model_fn(**models_args)

  model = model.to(device)

  optimizer = optimizers_factory[FLAGS.optimizer](model.parameters(), lr=0.001)

  # print(vocab.print_statistics())

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
  try:
    status = None
    if _cudart:
      status =  _cudart.cudaProfilerStart()
    trainer.train()
  finally:
    if status == 0:
      _cudart.cudaProfilerStop()
  
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

if __name__ == "__main__":
  app.run(main)
