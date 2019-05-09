
datasets_links = {
  'debug': {
    'train': {
      'preprocess': False,
      'paths': [
        'https://raw.githubusercontent.com/allenai/allennlp/master/tutorials/tagger/training.txt',
      ]
    },
    'val': {
      'preprocess': False,
      'paths': [
        'https://raw.githubusercontent.com/allenai/allennlp/master/tutorials/tagger/validation.txt',
      ]
    }
  },
  'ud-eng': {
    'train': {
      'preprocess': False,
      'paths': [
        'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.3/en_ewt-ud-train.conllu'
      ]
    },
    'val': {
      'preprocess': False,
      'paths': [
        'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu',
      ]
    }
  },
  'nc_zhen': {
    'train': {
      'preprocess': False,
      'paths': [
        'https://s3.eu-west-2.amazonaws.com/mattzooey/training-parallel-nv-v12/training/news-commentary-v14.en-zh.train.tsv',
        ]
      },
    'val': {
      'preprocess': False,
      'paths': [
        'https://s3.eu-west-2.amazonaws.com/mattzooey/training-parallel-nv-v12/training/news-commentary-v14.en-zh.val.tsv'
      ]
    }
  }
}

def get_dataset_paths(dataset_name):
  return datasets_links.get(dataset_name)