from typing import Dict
from allennlp.data import Instance
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers import LanguageModelingReader
from allennlp.common.file_utils import cached_path
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from overrides import overrides
import logging
from allennlp.common.tqdm import Tqdm
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("wiki_text")
class WikiTextDatasetReader(LanguageModelingReader):
  """
  DatasetReader for WikiText
  """

  def __init__(self, 
               tokens_per_instance: int = None, 
               tokenizer: Tokenizer = None, 
               token_indexers: Dict[str, TokenIndexer] = None, 
               lazy: bool = False, 
               max_length_sentence: int = None):

    self.max_length_sentence = max_length_sentence
    super().__init__(tokens_per_instance=tokens_per_instance, tokenizer=tokenizer, token_indexers=token_indexers, lazy=lazy)

  @overrides
  def _read(self, file_path: str):
      # if `file_path` is a URL, redirect to the cache
      file_path = cached_path(file_path)

      with open(file_path, "r", encoding="utf8") as text_file:
          instance_strings = text_file.readlines()

      if self._tokens_per_instance is not None:
          all_text = " ".join([x.replace("\n", " ").strip() for x in instance_strings])
          tokenized_text = self._tokenizer.tokenize(all_text)
          num_tokens = self._tokens_per_instance + 1
          tokenized_strings = []
          logger.info("Creating dataset from all text in file: %s", file_path)
          for index in Tqdm.tqdm(range(0, len(tokenized_text) - num_tokens, num_tokens - 1)):
              tokenized_strings.append(tokenized_text[index:(index + num_tokens)])
      else:
          tokenized_strings = [self._tokenizer.tokenize(s) for s in instance_strings]

      for tokenized_string in tokenized_strings:
          if len(tokenized_string) <= 6: continue # skip short sentence
          tokenized_string = list(tokenized_string)
          length = min(len(tokenized_string), self.max_length_sentence)
          tokenized_string = tokenized_string[:length-1]
          tokenized_string.insert(0, Token(START_SYMBOL))
          tokenized_string.append(Token(END_SYMBOL))
          input_field = TextField(tokenized_string[:-1], self._token_indexers)
          output_field = TextField(tokenized_string[1:], self._output_indexer)
          yield Instance({'input_tokens': input_field,
                          'output_tokens': output_field})