from typing import Iterator, Dict, List, Optional
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

class TaskDataReader(DatasetReader):
  """
  DatasetReader for PoS tagging data, one sentence per line,
    e.g. 'The###DET dog###NN ate###V the###DET apple###NN'
  """
  def __init__(self, 
               token_indexers: Dict[str, TokenIndexer] = None) -> None:
    super().__init__(lazy=False)
    # map word into index, single index for each word(token)
    self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

  def text_to_instance(self, 
                       tokens: List[Token], 
                       tags: List[str] = None) -> Instance:
    sentence_field = TextField(tokens, self.token_indexers)
    fields = {"sentence": sentence_field}

    # tags were optional because if we were in inference phrases
    # we do not have labels 
    if tags:
      label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
      fields["labels"] = label_field
    return Instance(fields)

  def _read(self, file_path: str) -> Iterator[Instance]:
    with open(file_path) as f:
      for line in f:
        pairs = line.strip().split()
        sentence, tags = zip(*(pair.split("###") for pair in pairs))
        yield self.text_to_instance([Token(word) for word in sentence], tags)


