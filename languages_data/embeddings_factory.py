
import torch
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

embeddings_weight_factory = {
  'basic': None,
}

embeddings_factory = {
  'basic': BasicTextFieldEmbedder,
}

def get_embeddings(embeddings_name, vocab, embedding_dim=128):
  token_embeddings = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                               embedding_dim=embedding_dim)
  weights = embeddings_weight_factory[embeddings_name]
  if weights is not None:
    if isinstance(weights, torch.Tensor):
      token_embeddings.weight = weights
  embedder = embeddings_factory[embeddings_name]({'tokens': token_embeddings})
  return embedder

