import torch
import logging
from typing import Dict, List, Optional, Iterator

from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import CategoricalAccuracy

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
logger = logging.getLogger(__name__)

class WordsLstmPosTagger(Model):
  def __init__(self, 
               word_embeddings: TextFieldEmbedder,
               encoder: Seq2SeqEncoder,
               vocab: Vocabulary,
               output_feature_key: str = 'pos',
               **kwargs) -> None:
    super().__init__(vocab)
    self.word_embeddings = word_embeddings
    self.encoder = encoder
    self.output_size = vocab.get_vocab_size(output_feature_key)
    logger.info(f"Key {output_feature_key} : feature size {self.output_size}")
    self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=self.output_size)
    self.accuracy = CategoricalAccuracy()

  def forward(self, 
              words: Dict[str, torch.Tensor],
              pos_tags: torch.Tensor = None,
              **args) -> Dict[str, torch.Tensor]:
    mask = get_text_field_mask(words)
    embeddings = self.word_embeddings(words)
    encoder_out = self.encoder(embeddings, mask)

    tag_logits = self.hidden2tag(encoder_out)
    outputs = {'tag_logits': tag_logits}

    if pos_tags is not None:
      self.accuracy(tag_logits, pos_tags, mask)
      outputs['loss'] = sequence_cross_entropy_with_logits(tag_logits, pos_tags, mask)
    return outputs

  def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    return {'accuracy': self.accuracy.get_metric(reset)}

class SimpleLstmPosTagger(Model):
  def __init__(self, 
               word_embeddings: TextFieldEmbedder,
               encoder: Seq2SeqEncoder,
               vocab: Vocabulary,
               output_feature_key: str = 'labels',
               **kwargs) -> None:
    super().__init__(vocab)
    self.word_embeddings = word_embeddings
    self.encoder = encoder
    self.output_feature_key = output_feature_key
    logger.info(f"Key {output_feature_key} : feature size {self.output_size}")
    self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size(self.output_feature_key))
    self.accuracy = CategoricalAccuracy()

  def forward(self, 
              sentence: Dict[str, torch.Tensor],
              labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    mask = get_text_field_mask(sentence)
    embeddings = self.word_embeddings(sentence)
    encoder_out = self.encoder(embeddings, mask)

    tag_logits = self.hidden2tag(encoder_out)
    outputs = {'tag_logits': tag_logits}

    if labels is not None:
      self.accuracy(tag_logits, labels, mask)
      outputs['loss'] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)
    return outputs

  def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    return {'accuracy': self.accuracy.get_metric(reset)}