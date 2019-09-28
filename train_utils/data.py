from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataset(dataset_fn, dataset_dir):
  # TODO: currently this is the only transforms.
  compose_trans = transforms.Compose([
      transforms.RandomVerticalFlip(),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor()
    ])
    
  train_dataset = dataset_fn(dataset_dir, transform=compose_trans, train=True, download=True)
  val_dataset = dataset_fn(dataset_dir, transform=compose_trans, train=False, download=True)
  return train_dataset, val_dataset

def get_standard_dataloader(dataset_fn, dataset_dir, batch_size, threadiness=2, shuffle=True):
  train_dataset, val_dataset = get_dataset(dataset_fn, dataset_dir)
  train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            num_workers=threadiness)
  val_loader = DataLoader(val_dataset, 
                          batch_size=batch_size, 
                          shuffle=shuffle, 
                          num_workers=threadiness)
  return train_loader, val_loader