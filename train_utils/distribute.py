import torch
import torch.utils.data.distributed as dist

from allennlp.training import util as allen_training_util

def distributed_dataloader(dataset, threads=2, batch_size=32):
  train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=threads, sampler=train_sampler)
  return train_sampler, dataloader


def get_metrics(model: Model, device: torch.device, world_size: int, total_loss: float, num_batches: int, reset: bool = False) -> Dict[str, float]:
  """
  Gets the metrics but sets ``"loss"`` to
  the total loss divided by the ``num_batches`` so that
  the ``"loss"`` metric is "average loss per batch".
  https://github.com/scarecrow1123/allennlp/commit/65036e7b7c4c169a04446237ebef87ece6ca8bfe#diff-7ad763815a56f9a0fa02c60ab6fabccb
  """
  metrics = model.get_metrics(reset=reset)
  metrics["loss"] = float(total_loss / num_batches) if num_batches > 0 else 0.0
  aggregated_metrics = {}
  for metric_name, metric_val in metrics.items():
    metric_tensor = torch.tensor(metric_val).to(device)
    metric_gathered = [torch.zeros_like(metric_tensor) for _ in range(world_size)]
  
    dist.all_gather(metric_gathered, metric_tensor)
    metric_gathered = torch.tensor(metric_gathered)
    aggregated_metrics[metric_name] = metric_gathered.mean().item()
  return aggregated_metrics
