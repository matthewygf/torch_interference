from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer


data_reader_configs = {
  'debug': None,
  'ud-eng': None,
  'nc_zhen': {
    'source_tokenizer': WordTokenizer(),
    'target_tokenizer': CharacterTokenizer(),
    'source_token_indexers': {'tokens': SingleIdTokenIndexer()},
    'target_token_indexers': {'tokens': SingleIdTokenIndexer(namespace='target_tokens')}
  },
  'wikitext': {}
}

def get_datareader_configs(dataset_name):
  return data_reader_configs.get(dataset_name)