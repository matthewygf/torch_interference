"""
an experimental implementation for distributed training
"""
import logging
from typing import Dict, List, Union, Any

from allennlp.common import Params, Registrable
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.models.model import Model
import torch

logger = logging.getLogger(__name__)

class DistributedTrainerBase(Registrable):
  default_implementation = "distribute"
  """
   we leverage n process n gpus, hence, we assume that when code path hits this.
   we are getting a gpu only because when multiprocess spawns, assume we have n distributed trainer.
   if 
  """
  def __init__(self,
               rank: int,
               worldsize: int,
               ngpus_per_node: int, 
               cuda_device: Union[int, List],
               serialization_dir: str = None) -> None:
    
    # we set the serialization directory, however will only get used for loading ckpt
    # only rank = 0 will be allowed to save the model.
    self._serialization_dir = serialization_dir
    self._rank = rank
    self._worldsize = worldsize
    self._ngpus_per_node = ngpus_per_node

    self._is_chief = self._rank % self._ngpus_per_node == 0
    
    # NOTE: assume users did its own cuda check. 
    # user should realize that if there is no need for distributed, just use normal trainer base.
    assert len(cuda_device) == 1, "Expect 1 gpu device only for n process n gpu pattern, but got %s" % str(cuda_device)
    self._cuda_device = cuda_device
  
  def _move_to_gpu(self, model:Model) -> Model:
    return model.cuda(self._cuda_device[0])
  
  def train(self) -> Dict[str, Any]:
    raise NotImplementedError
