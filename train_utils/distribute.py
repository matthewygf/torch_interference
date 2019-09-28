import torch
import torch.utils.data.distributed


def distributed_dataloader(dataset, threads=2, batch_size=32):
  train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=threads, sampler=train_sampler)
  return train_sampler, dataloader