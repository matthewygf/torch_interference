import torch
import torch.utils.data.distributed

#https://pytorch.org/docs/1.1.0/nn.html#torch.nn.parallel.DistributedDataParallel
def distributed_model(model, gpu_index):
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_index], output_device=gpu_index)
  return model

def distributed_dataloader(dataset, gpus, rank, threads=2, batch_size=32):
  train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,gpus,rank)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=threads, sampler=train_sampler)
  return dataloader