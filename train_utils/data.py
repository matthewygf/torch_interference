from torchvision import transforms
from torch.utils.data import DataLoader
import train_utils.distribute as distribute_utils

def get_dataset(dataset_fn, dataset_dir, download=False):
  # TODO: currently this is the only transforms.
  compose_trans = transforms.Compose([
      transforms.RandomVerticalFlip(),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor()
    ])
  
  train_dataset = dataset_fn(dataset_dir, transform=compose_trans, train=True, download=download)
  val_dataset = dataset_fn(dataset_dir, transform=compose_trans, train=False, download=download)
  return train_dataset, val_dataset

def get_standard_dataloader(dataset_fn, dataset_dir, batch_size, threadiness=2, shuffle=True, download=True):
  train_dataset, val_dataset = get_dataset(dataset_fn, dataset_dir, download=download)
  train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            num_workers=threadiness)
  val_loader = DataLoader(val_dataset, 
                          batch_size=batch_size, 
                          shuffle=shuffle, 
                          num_workers=threadiness)
  return train_loader, val_loader

def get_distribute_dataloader(dataset_fn, dataset_dir, batch_size, threadiness, download):
  train_dataset, val_dataset = get_dataset(dataset_fn, dataset_dir, download=download)
  sampler, dist_train_loader = distribute_utils.distributed_dataloader(train_dataset, threadiness, batch_size)
  val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=threadiness)
  return sampler, dist_train_loader, val_loader
