
datasets_links = {
  'debug': {
    'train': 'https://raw.githubusercontent.com/allenai/allennlp/master/tutorials/tagger/training.txt',
    'val': 'https://raw.githubusercontent.com/allenai/allennlp/master/tutorials/tagger/validation.txt'
  },
  'ud-eng': {
    'train': 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.3/en_ewt-ud-train.conllu',
    'val': 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu'
  }
}

def get_dataset_paths(dataset_name):
  return datasets_links.get(dataset_name)