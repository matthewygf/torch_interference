import csv
from typing import Dict
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("maxlenseq2seq")
class MaxLengthSeq2SeqReader(Seq2SeqDatasetReader):
  def __init__(self, 
               source_tokenizer=None, 
               target_tokenizer=None, 
               source_token_indexers=None, 
               target_token_indexers=None, 
               source_add_start_token=True, 
               delimiter='\t', 
               lazy=False,
               max_sentence_length=None):
    self.max_sentence_length = max_sentence_length
    super().__init__(source_tokenizer=source_tokenizer, 
                     target_tokenizer=target_tokenizer, 
                     source_token_indexers=source_token_indexers, 
                     target_token_indexers=target_token_indexers, 
                     source_add_start_token=source_add_start_token, 
                     delimiter=delimiter, 
                     lazy=lazy)
  
  @overrides
  def text_to_instance(self, source_string: str, target_string: str = None) -> Instance:  # type: ignore
    # pylint: disable=arguments-differ
    tokenized_source = self._source_tokenizer.tokenize(source_string)
    length = min(len(tokenized_source), self.max_sentence_length)
    tokenized_source = tokenized_source[:length-2]

    if self._source_add_start_token:
        tokenized_source.insert(0, Token(START_SYMBOL))
    tokenized_source.append(Token(END_SYMBOL))
    source_field = TextField(tokenized_source, self._source_token_indexers)
    if target_string is not None:
        tokenized_target = self._target_tokenizer.tokenize(target_string)
        length = min(len(tokenized_target), self.max_sentence_length)
        tokenized_target = tokenized_target[:length-2]
        tokenized_target.insert(0, Token(START_SYMBOL))
        tokenized_target.append(Token(END_SYMBOL))
        target_field = TextField(tokenized_target, self._target_token_indexers)
        return Instance({"source_tokens": source_field, "target_tokens": target_field})
    else:
        return Instance({'source_tokens': source_field})
  
  @overrides
  def _read(self, file_path):
    with open(cached_path(file_path), "r", encoding="utf-8") as data_file:
        logger.info("Reading instances from lines in file at: %s", file_path)
        for line_num, row in enumerate(csv.reader(data_file, delimiter=self._delimiter)):
            if len(row) != 2:
                raise ConfigurationError("Invalid line format: %s (line number %d)" % (row, line_num + 1))
            source_sequence, target_sequence = row
            yield self.text_to_instance(source_sequence, target_sequence)