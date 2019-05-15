from allennlp.data.iterators import BucketIterator, BasicIterator

iterator_keys = {
  # this is implemented in pos_data_reader
  'debug': ('sentence', 'num_tokens'),
  # this is implemented in UniversalDependenciesDatasetReader
  'ud-eng': ('words', 'num_tokens'),
  'nc_zhen': ('source_tokens', 'num_tokens'),
  'wikitext': None
}

iterator_type = {
  'ud-eng': BucketIterator,
  'nc_zhen': BucketIterator,
  'wikitext': BasicIterator
}

def get_iterator(dataset_name, batch_size=16):
  keys = iterator_keys[dataset_name]
  if keys is not None:
    iterator_obj = iterator_type[dataset_name](batch_size=batch_size, sorting_keys=[keys])
  else:
    iterator_obj = iterator_type[dataset_name](batch_size=batch_size)
  return iterator_obj

