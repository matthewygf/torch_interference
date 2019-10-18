import logging
import math
import os
import time
import datetime
import traceback
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any

import torch
import torch.optim.lr_scheduler

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, parse_cuda_device
from allennlp.common.util import (dump_metrics, gpu_memory_mb, peak_memory_mb,
                                  lazy_groups_of)
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
from train_utils.distributed_trainer_base import DistributedTrainerBase
from train_utils.distribute import get_metrics
from allennlp.training import util as training_util
from allennlp.training.moving_average import MovingAverage

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DistributedTrainerBase.register("distribute")
class DistributeTrainer(DistributedTrainerBase):
  """
    NOTE: only work in nprocess_ngpus
  """
  def __init__(self,
               rank: int,
               worldsize: int,
               ngpus_per_node: int,
               cuda_device: Union[int, List],
               model: Model,
               optimizer: torch.optim.Optimizer,
               iterator: DataIterator,
               train_dataset: Iterable[Instance],
               validation_dataset: Optional[Iterable[Instance]] = None,
               patience: Optional[int] = None,
               validation_metric: str = "-loss",
               validation_iterator: DataIterator = None,
               shuffle: bool = True,
               num_epochs: int = 20,
               serialization_dir: Optional[str] = None,
               num_serialized_models_to_keep: int = 20,
               keep_serialized_model_every_num_seconds: int = None,
               checkpointer: Checkpointer = None,
               model_save_interval: float = None,
               grad_norm: Optional[float] = None,
               grad_clipping: Optional[float] = None,
               learning_rate_scheduler: Optional[LearningRateScheduler] = None,
               momentum_scheduler: Optional[MomentumScheduler] = None,
               summary_interval: int = 100,
               histogram_interval: int = None,
               should_log_parameter_statistics: bool = True,
               should_log_learning_rate: bool = False,
               log_batch_size_period: Optional[int] = None,
               moving_average: Optional[MovingAverage] = None) -> None:

    super().__init__(rank, worldsize, ngpus_per_node, cuda_device, serialization_dir)

    self.model = model
    self.iterator = iterator
    self._validation_iterator = validation_iterator
    self.shuffle = shuffle
    self.optimizer = optimizer
    self.train_data = train_dataset
    self._validation_data = validation_dataset

    self._metric_tracker = MetricTracker(patience, validation_metric)
    self._validation_metric = validation_metric[1:]
    self._num_epochs = num_epochs

    # NOTE: although We have ckpter for everyone, only rank 0 of each node should be able to ckpt
    if checkpointer is not None:
      self._checkpointer = checkpointer
    else:
      self._checkpointer = Checkpointer(serialization_dir,
                                        keep_serialized_model_every_num_seconds,
                                        num_serialized_models_to_keep)

    self._model_save_interval = model_save_interval
    
    self._grad_norm = grad_norm
    self._grad_clipping = grad_clipping
    self._learning_rate_scheduler = learning_rate_scheduler
    self._momentum_scheduler = momentum_scheduler
    self._moving_average = moving_average

    # We keep the total batch number as an instance variable because it
    # is used inside a closure for the hook which logs activations in
    # ``_enable_activation_logging``.
    self._batch_num_total = 0

    # NOTE: log.
    serialization_dir = os.path.join(serialization_dir, str(rank))
    self._tensorboard = TensorboardWriter(
                get_batch_num_total=lambda: self._batch_num_total,
                serialization_dir=serialization_dir,
                summary_interval=summary_interval,
                histogram_interval=histogram_interval,
                should_log_parameter_statistics=should_log_parameter_statistics,
                should_log_learning_rate=should_log_learning_rate)
    
    self._log_batch_size_period = log_batch_size_period

    self._last_log = 0.0  # time of last logging
    
    # Enable activation logging.
    if histogram_interval is not None:
      self._tensorboard.enable_activation_logging(self.model)

  def rescale_gradients(self) -> Optional[float]:
    return training_util.rescale_gradients(self.model, self._grad_norm)
  
  def batch_loss(self, batch_group: List[TensorDict], for_training: bool) -> torch.Tensor:
    assert len(batch_group) == 1
    batch = batch_group[0]
    batch = nn_util.move_to_device(batch, self._cuda_device[0])
    output_dict = self.model(**batch)

    try:
      loss = output_dict["loss"]
      if for_training:
        loss += self.model.get_regularization_penalty()
    except KeyError:
      if for_training:
        raise RuntimeError("The model you are trying to optimize does not contain a"
                                  " 'loss' key in the output of model.forward(inputs).")
        loss = None

    return loss
  
  def _train_epoch(self, epoch: int) -> Dict[str, float]:
    """ 
    Trains one epoch and returns metrics. 
    only report system utils when we are local rank 0 at each machine. 
    """
    logger.info("Rank %d: Epoch %d/%d", self._rank, epoch, self._num_epochs - 1)
    peak_cpu_usage = peak_memory_mb()
    if self._is_chief:
      logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
    
    train_loss = 0.0
    # Set the model to "train" mode.
    self.model.train()

    # should be 1 anyway, because we are only dealing with nprocess_with_ngpus
    num_gpus = len(self._cuda_device)

    # TODO: Implementation of whether the generator should take into account of worldsize.
    # Get tqdm for the training batches
    raw_train_generator = self.iterator(self.train_data,
                                        num_epochs=1,
                                        shuffle=self.shuffle)
    train_generator = lazy_groups_of(raw_train_generator, num_gpus)
    num_training_batches = math.ceil(self.iterator.get_num_batches(self.train_data)/num_gpus)
    self._last_log = time.time()
    last_save_time = time.time()

    batches_this_epoch = 0
    if self._batch_num_total is None:
      self._batch_num_total = 0

    histogram_parameters = set(self.model.get_parameters_for_histogram_tensorboard_logging())

    logger.info("Training")
    train_generator_tqdm = Tqdm.tqdm(train_generator,
                                      total=num_training_batches)
    cumulative_batch_size = 0
    # NOTE: only work in nprocess_ngpus
    device = torch.device("cuda:%d" % self._cuda_device[0])
    for batch_group in train_generator_tqdm:
      batches_this_epoch += 1
      self._batch_num_total += 1
      batch_num_total = self._batch_num_total

      self.optimizer.zero_grad()

      loss = self.batch_loss(batch_group, for_training=True)

      if torch.isnan(loss):
        raise ValueError("nan loss encountered")
      
      loss.backward()
      train_loss += loss.item()
      batch_grad_norm = self.rescale_gradients()

      # This does nothing if batch_num_total is None or you are using a
      # scheduler which doesn't update per batch.
      if self._learning_rate_scheduler:
        self._learning_rate_scheduler.step_batch(batch_num_total)
      if self._momentum_scheduler:
        self._momentum_scheduler.step_batch(batch_num_total)
      
      if self._is_chief:
        # only chief do tensorboard
        if self._tensorboard.should_log_histograms_this_batch():
          # get the magnitude of parameter updates for logging
          # We need a copy of current parameters to compute magnitude of updates,
          # and copy them to CPU so large models won't go OOM on the GPU.
          param_updates = {name: param.detach().cpu().clone()
                            for name, param in self.model.named_parameters()}
          self.optimizer.step()
          for name, param in self.model.named_parameters():
              param_updates[name].sub_(param.detach().cpu())
              update_norm = torch.norm(param_updates[name].view(-1, ))
              param_norm = torch.norm(param.view(-1, )).cpu()
              self._tensorboard.add_train_scalar("gradient_update/" + name,
                                                  update_norm / (param_norm + 1e-7))
        else:
          self.optimizer.step()
      else:
        self.optimizer.step()

      # Update moving averages
      # NOTE: not sure whether this need to be average
      if self._moving_average is not None:
        self._moving_average.apply(batch_num_total)

      metrics = get_metrics(self.model, device, self._worldsize, train_loss, batches_this_epoch)

      description = training_util.description_from_metrics(metrics)
      train_generator_tqdm.set_description(("Rank %d: " % self._rank) + description, refresh=False)

      if self._is_chief:
        # Log parameter values to Tensorboard
        if self._tensorboard.should_log_this_batch():
          self._tensorboard.log_parameter_and_gradient_statistics(self.model, batch_grad_norm)
          self._tensorboard.log_learning_rates(self.model, self.optimizer)

          self._tensorboard.add_train_scalar("loss/loss_train", metrics["loss"])
          self._tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in metrics.items()})

        if self._tensorboard.should_log_histograms_this_batch():
          self._tensorboard.log_histograms(self.model, histogram_parameters)

      if self._log_batch_size_period:
        cur_batch = sum([training_util.get_batch_size(batch) for batch in batch_group])
        cumulative_batch_size += cur_batch
        if (batches_this_epoch - 1) % self._log_batch_size_period == 0:
          average = cumulative_batch_size/batches_this_epoch
          logger.info(f"rank {self._rank}, current batch size: {cur_batch} mean batch size: {average}")
          if self._is_chief:
            self._tensorboard.add_train_scalar("current_batch_size", cur_batch)
            self._tensorboard.add_train_scalar("mean_batch_size", average)
      
      if self._is_chief:
        # Save model if needed.
        if self._model_save_interval is not None and (
                time.time() - last_save_time > self._model_save_interval
        ):
          last_save_time = time.time()
          self._save_checkpoint(
                  '{0}.{1}'.format(epoch, training_util.time_to_str(int(last_save_time)))
          )
      
      metrics = get_metrics(self.model, device, self._worldsize, train_loss, batches_this_epoch)
      metrics['cpu_memory_MB'] = peak_cpu_usage
      return metrics

  def _validation_loss(self) -> Tuple[float, int]:
    """
    Computes the validation loss. Returns it and the number of batches.
    """
    logger.info("Rank %d Validating", self._rank)

    self.model.eval()

    # Replace parameter values with the shadow values from the moving averages.
    if self._moving_average is not None:
        self._moving_average.assign_average_value()

    if self._validation_iterator is not None:
        val_iterator = self._validation_iterator
    else:
        val_iterator = self.iterator

    num_gpus = len(self._cuda_device)

    raw_val_generator = val_iterator(self._validation_data,
                                      num_epochs=1,
                                      shuffle=False)
    val_generator = lazy_groups_of(raw_val_generator, num_gpus)
    num_validation_batches = math.ceil(val_iterator.get_num_batches(self._validation_data)/num_gpus)
    val_generator_tqdm = Tqdm.tqdm(val_generator,
                                    total=num_validation_batches)
    batches_this_epoch = 0
    val_loss = 0
    for batch_group in val_generator_tqdm:

        loss = self.batch_loss(batch_group, for_training=False)
        if loss is not None:
            # You shouldn't necessarily have to compute a loss for validation, so we allow for
            # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
            # currently only used as the divisor for the loss function, so we can safely only
            # count those batches for which we actually have a loss.  If this variable ever
            # gets used for something else, we might need to change things around a bit.
            batches_this_epoch += 1
            val_loss += loss.detach().cpu().numpy()

        # Update the description with the latest metrics
        val_metrics = training_util.get_metrics(self.model, val_loss, batches_this_epoch)
        description = training_util.description_from_metrics(val_metrics)
        val_generator_tqdm.set_description(description, refresh=False)

    # Now restore the original parameter values.
    if self._moving_average is not None:
        self._moving_average.restore()

    return val_loss, batches_this_epoch

  def train(self) -> Dict[str, Any]:
    """
    Trains the supplied model with the supplied parameters.
    """
    try:
        epoch_counter = self._restore_checkpoint()
    except RuntimeError:
        traceback.print_exc()
        raise ConfigurationError("Could not recover training from the checkpoint.  Did you mean to output to "
                                  "a different serialization directory or delete the existing serialization "
                                  "directory?")

    training_util.enable_gradient_clipping(self.model, self._grad_clipping)

    logger.info("Rank %d Beginning training.", self._rank)

    train_metrics: Dict[str, float] = {}
    val_metrics: Dict[str, float] = {}
    this_epoch_val_metric: float = None
    metrics: Dict[str, Any] = {}
    epochs_trained = 0
    training_start_time = time.time()

    metrics['best_epoch'] = self._metric_tracker.best_epoch
    for key, value in self._metric_tracker.best_epoch_metrics.items():
      metrics["best_validation_" + key] = value

    for epoch in range(epoch_counter, self._num_epochs):
      epoch_start_time = time.time()
      train_metrics = self._train_epoch(epoch)

      # get peak of memory usage
      if self._is_chief:
        if 'cpu_memory_MB' in train_metrics:
          metrics['peak_cpu_memory_MB'] = max(metrics.get('peak_cpu_memory_MB', 0),
                                              train_metrics['cpu_memory_MB'])

        if self._validation_data is not None:
          with torch.no_grad():
            # We have a validation set, so compute all the metrics on it.
            val_loss, num_batches = self._validation_loss()
            val_metrics = training_util.get_metrics(self.model, val_loss, num_batches, reset=True)

            # Check validation metric for early stopping
            this_epoch_val_metric = val_metrics[self._validation_metric]
            self._metric_tracker.add_metric(this_epoch_val_metric)

            if self._metric_tracker.should_stop_early():
              logger.info("Ran out of patience.  Stopping training.")
              break
        if self._is_chief:
          self._tensorboard.log_metrics(train_metrics,
                                        val_metrics=val_metrics,
                                        log_to_console=True,
                                        epoch=epoch + 1)  # +1 because tensorboard doesn't like 0

        # Create overall metrics dict
        training_elapsed_time = time.time() - training_start_time
        metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
        metrics["training_start_epoch"] = epoch_counter
        metrics["training_epochs"] = epochs_trained
        metrics["epoch"] = epoch

        for key, value in train_metrics.items():
          metrics["training_" + key] = value
        for key, value in val_metrics.items():
          metrics["validation_" + key] = value

        if self._metric_tracker.is_best_so_far():
          # Update all the best_ metrics.
          # (Otherwise they just stay the same as they were.)
          metrics['best_epoch'] = epoch
          for key, value in val_metrics.items():
            metrics["best_validation_" + key] = value

          self._metric_tracker.best_epoch_metrics = val_metrics

        if self._serialization_dir and self._is_chief:
          dump_metrics(os.path.join(self._serialization_dir, f'metrics_epoch_{epoch}.json'), metrics)

        # The Scheduler API is agnostic to whether your schedule requires a validation metric -
        # if it doesn't, the validation metric passed here is ignored.
        if self._learning_rate_scheduler:
          self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)
        if self._momentum_scheduler:
          self._momentum_scheduler.step(this_epoch_val_metric, epoch)

        if self._is_chief:
          self._save_checkpoint(epoch)

        epoch_elapsed_time = time.time() - epoch_start_time
        logger.info("Rank %d Epoch duration: %s", self._rank, datetime.timedelta(seconds=epoch_elapsed_time))

        if epoch < self._num_epochs - 1:
          training_elapsed_time = time.time() - training_start_time
          estimated_time_remaining = training_elapsed_time * \
              ((self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1)
          formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
          logger.info("Rank %d, Estimated training time remaining: %s", self._rank, formatted_time)

        epochs_trained += 1

    # make sure pending events are flushed to disk and files are closed properly
    self._tensorboard.close()

    # Load the best model state before returning
    best_model_state = self._checkpointer.best_model_state()
    if best_model_state:
        self.model.load_state_dict(best_model_state)

    return metrics

  def _save_checkpoint(self, epoch: Union[int, str]) -> None:
    """
    Saves a checkpoint of the model to self._serialization_dir.
    Is a no-op if self._serialization_dir is None.

    Parameters
    ----------
    epoch : Union[int, str], required.
        The epoch of training.  If the checkpoint is saved in the middle
        of an epoch, the parameter is a string with the epoch and timestamp.
    """
    # If moving averages are used for parameters, we save
    # the moving average values into checkpoint, instead of the current values.
    if self._moving_average is not None:
      self._moving_average.assign_average_value()

    # These are the training states we need to persist.
    training_states = {
            "metric_tracker": self._metric_tracker.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "batch_num_total": self._batch_num_total
    }

    # If we have a learning rate or momentum scheduler, we should persist them too.
    if self._learning_rate_scheduler is not None:
        training_states["learning_rate_scheduler"] = self._learning_rate_scheduler.state_dict()
    if self._momentum_scheduler is not None:
        training_states["momentum_scheduler"] = self._momentum_scheduler.state_dict()

    self._checkpointer.save_checkpoint(
            model_state=self.model.state_dict(),
            epoch=epoch,
            training_states=training_states,
            is_best_so_far=self._metric_tracker.is_best_so_far())

    # Restore the original values for parameters so that training will not be affected.
    if self._moving_average is not None:
      self._moving_average.restore()
  
  
  def _restore_checkpoint(self) -> int:
    """
    Restores the model and training state from the last saved checkpoint.
    This includes an epoch count and optimizer state, which is serialized separately
    from model parameters. This function should only be used to continue training -
    if you wish to load a model for inference/load parts of a model into a new
    computation graph, you should use the native Pytorch functions:
    `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

    If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
    this function will do nothing and return 0.

    Returns
    -------
    epoch: int
        The epoch at which to resume training, which should be one after the epoch
        in the saved training state.
    """
    model_state, training_state = self._checkpointer.restore_checkpoint()

    if not training_state:
        # No checkpoint to restore, start at 0
        return 0

    self.model.load_state_dict(model_state)
    self.optimizer.load_state_dict(training_state["optimizer"])
    if self._learning_rate_scheduler is not None and "learning_rate_scheduler" in training_state:
        self._learning_rate_scheduler.load_state_dict(training_state["learning_rate_scheduler"])
    if self._momentum_scheduler is not None and "momentum_scheduler" in training_state:
        self._momentum_scheduler.load_state_dict(training_state["momentum_scheduler"])
    training_util.move_optimizer_to_cuda(self.optimizer)

    # Currently the ``training_state`` contains a serialized ``MetricTracker``.
    if "metric_tracker" in training_state:
        self._metric_tracker.load_state_dict(training_state["metric_tracker"])
    # It used to be the case that we tracked ``val_metric_per_epoch``.
    elif "val_metric_per_epoch" in training_state:
        self._metric_tracker.clear()
        self._metric_tracker.add_metrics(training_state["val_metric_per_epoch"])
    # And before that we didn't track anything.
    else:
        self._metric_tracker.clear()

    if isinstance(training_state["epoch"], int):
        epoch_to_return = training_state["epoch"] + 1
    else:
        epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1

    # For older checkpoints with batch_num_total missing, default to old behavior where
    # it is unchanged.
    batch_num_total = training_state.get('batch_num_total')
    if batch_num_total is not None:
        self._batch_num_total = batch_num_total

    return epoch_to_return