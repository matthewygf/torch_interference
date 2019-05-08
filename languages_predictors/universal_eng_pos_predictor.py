# Borrowed from: https://github.com/mhagiwara/realworldnlp/blob/master/realworldnlp/predictors.py
# Note: we need our own predictor because the Vocab instance created by the  UniversalDependenciesDatasetReader
#       has made that our list of str is in key of 'words' instead of 'sentences' ._.

from typing import List

from allennlp.common.util import JsonDict

from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.data import DatasetReader, Instance

from overrides import overrides

@Predictor.register('universal_eng_pos_predictor')
class UniversalEngPosPredictor(Predictor):
  def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
    super().__init__(model, dataset_reader)
  
  def predict(self, words: List[str]) -> JsonDict:
    return self.predict_json({'words': words})

  @overrides
  def _json_to_instance(self, json_dict: JsonDict) -> Instance:
    words = json_dict['words']
    # we do not need 'upostag' like UniversalDependenciesDatasetReader specified in prediction
    # so we can just put words again as it doesnt use it anyway
    return self._dataset_reader.text_to_instance(words, words)