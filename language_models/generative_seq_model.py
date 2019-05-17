# https://github.com/mhagiwara/realworldnlp/blob/master/examples/generation/lm.py

import torch
from typing import Tuple, List
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules import TextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary, DEFAULT_PADDING_TOKEN
from allennlp.data.tokenizers import Token
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.common.util import START_SYMBOL, END_SYMBOL

class GenerativeSeqModel(Model):
  def __init__(self,
               word_embeddings: TextFieldEmbedder, 
               hidden_size: int, 
               max_len: int,
               vocab: Vocabulary,
               encoder: Seq2SeqEncoder = None,
               **kwargs) -> None:
    super().__init__(vocab)
    self.embeddings = word_embeddings
    self.encoder = encoder
    self.hidden2out = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('tokens'))
    self.hidden_size = hidden_size
    self.max_len = max_len
    self.vocab = vocab
  
  def forward(self, input_tokens, output_tokens):
    mask = get_text_field_mask(input_tokens)
    embeddings = self.embeddings(input_tokens)
    hidden_states = self.encoder(embeddings, mask)
    out_logits = self.hidden2out(hidden_states)
    loss = sequence_cross_entropy_with_logits(out_logits, output_tokens['tokens'], mask)
    return {'loss': loss}
  
  def generate(self, device, num_layers) -> Tuple[List[Token], torch.Tensor]:
    start_symbol_index = self.vocab.get_token_index(START_SYMBOL, 'tokens')
    end_symbol_index = self.vocab.get_token_index(END_SYMBOL, 'tokens')
    padding_symbol_index = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN, 'tokens')

    word_idx = start_symbol_index
    # TODO: many ways to init state.
    state = (torch.zeros(2*num_layers, 1, self.hidden_size).to(device),
             torch.zeros(2*num_layers, 1, self.hidden_size).to(device))
    
    log_likihood = 0.
    words = []
    for i in range(self.max_len):
      tokens = torch.tensor([[word_idx]]).to(device)
      embeddings = self.embeddings({'tokens': tokens})
      output, state = self.encoder._module(embeddings, state)
      output = self.hidden2out(output)

      log_prob = torch.log_softmax(output[0,0], dim=0)

      dist = torch.exp(log_prob)

      word_idx = start_symbol_index
      while word_idx in {start_symbol_index, padding_symbol_index}:
        word_idx = torch.multinomial(dist, num_samples=1, replacement=False).item()
      
      log_likihood += log_prob[word_idx]
      if word_idx == end_symbol_index:
        break
      
      token = Token(text=self.vocab.get_token_from_index(word_idx, 'tokens'))
      words.append(token)
    return words, log_likihood