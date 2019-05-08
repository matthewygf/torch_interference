from allennlp.data.iterators import BucketIterator

iterator_keys = {
  # this is implemented in pos_data_reader
  'debug': ('sentence', 'num_tokens'),
  # this is implemented in UniversalDependenciesDatasetReader
  'ud-eng': ('words', 'num_tokens')
}

iterator_type = {
  'ud-eng': BucketIterator
}

def get_iterator(dataset_name, batch_size=16):
  keys = iterator_keys[dataset_name]
  iterator_obj = iterator_type[dataset_name](batch_size=batch_size, sorting_keys=[keys])
  return iterator_obj

