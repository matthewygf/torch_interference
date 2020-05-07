
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
        'https://mattyzap.s3-ap-southeast-1.amazonaws.com/training-parallel-nv-v12/training/news-commentary-v14.en-zh.train.tsv',
        ]
      },
    'val': {
      'preprocess': False,
      'paths': [
        'https://mattyzap.s3-ap-southeast-1.amazonaws.com/training-parallel-nv-v12/training/news-commentary-v14.en-zh.val.tsv',
      ]
    }
  },
  'wikitext':{
    'train': {
      'preprocess': False,
      'paths':[
        'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt'
      ]
    },
    'val': {
      'preprocess': False,
      'paths': [
        'https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/valid.txt'
      ]
    }
  }
}

def get_dataset_paths(dataset_name):
  return datasets_links.get(dataset_name)