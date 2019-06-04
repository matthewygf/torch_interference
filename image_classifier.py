from absl import app
from absl import flags
# using predefined set of models
import torchvision.models as models
import torchvision.datasets as predefined_datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from image_models.model import EfficientNet
import torch
import torch.optim as optim
import time
import ctypes
import csv
import datetime
import utils as U

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""
  NOTE: Only using 1 GPU
  TODO: 4 GPUs job packing
"""

FLAGS = flags.FLAGS

flags.DEFINE_string('run_name', None, 'The name you want to give to this run')
flags.DEFINE_string('model', None, 'The model you want to test')
flags.DEFINE_string('dataset_dir', None, 'Dataset directory')
flags.DEFINE_integer('batch_size', 64, 'Batch size of the model training')
flags.DEFINE_string('dataset', 'cifar10', 'The dataset to train with')
flags.DEFINE_boolean('use_cuda', False, 'whether to use GPU')
flags.DEFINE_integer('log_interval', 10, 'Batch intervals to log')
flags.DEFINE_integer('max_epochs', 5, 'maximum number of epochs to run')

flags.mark_flag_as_required('run_name')
flags.mark_flag_as_required('model')
flags.mark_flag_as_required('dataset_dir')

models_factory = {
  'googlenet': models.googlenet,
  'mobilenet': models.mobilenet_v2,
  'mobilenet_large': models.mobilenet_v2,
  'resnet': models.resnet50,
  'vgg19': models.vgg19,
  'densenet121': models.densenet121,
  'densenet169': models.densenet169,
  'efficientnet-b0': EfficientNet.from_name,
  'efficientnet-b1': EfficientNet.from_name,
  'efficientnet-b2': EfficientNet.from_name,
  'efficientnet-b3': EfficientNet.from_name,
  'efficientnet-b4': EfficientNet.from_name,
}

datasets_factory = {
  'cifar10': predefined_datasets.CIFAR10
}

datasets_sizes = {
  'cifar10': 10,
  'imagenet': 100
}

def train(logger, model, device, train_loader, optimizer, epoch, loss_op):
  model.train()
  epoch_start = time.time()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    start_time = time.time()
    pred = model(data)
    loss = loss_op(pred, target)
    loss.backward()
    optimizer.step()
    time_elapsed = time.time() - start_time
    if batch_idx == 0:
      logger.info("First step of this epoch: %s", str(datetime.datetime.utcnow()))
    
    if batch_idx == len(train_loader) - 1:
      epoch_elapsed = time.time() - epoch_start
      logger.info("Last step of this epoch: %s, ran for %.4f", str(datetime.datetime.utcnow()), epoch_elapsed)

    if batch_idx % FLAGS.log_interval == 0:
      logger.info("Epoch %d: %d/%d [Loss: %.4f] (%.4f sec/step)", 
                  epoch, batch_idx*len(data), 
                  len(train_loader.dataset), loss.item(), time_elapsed)

def main(argv):
  del argv
  
  logger = U.get_logger(__name__+FLAGS.run_name)
  logger.info("run: %s, specified model: %s, dataset: %s", FLAGS.run_name, FLAGS.model, FLAGS.dataset)
  _cudart = U.get_cudart()
  if _cudart is None:
    logger.warning("No cudart, probably means you do not have cuda on this machine.")

  model_fn = models_factory[FLAGS.model]
  dataset_fn = datasets_factory[FLAGS.dataset]
  dataset_classes = datasets_sizes[FLAGS.dataset]
  # TODO: Really need to start to change this better soon :/
  if 'google' in FLAGS.model: 
    model = model_fn(pretrained=False, transform_input=False, aux_logits=False, num_classes=dataset_classes)
  elif 'mobilenet_large' in FLAGS.model:
    inverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [10, 24, 2, 2],
        [10, 32, 3, 2],
        [10, 64, 4, 2],
        [10, 96, 3, 1],
        [10, 160, 3, 2],
        [10, 320, 1, 1],
    ]
    model = model_fn(pretrained=False, num_classes=dataset_classes, inverted_residual_setting=inverted_residual_setting)
  elif 'efficientnet' in FLAGS.model:
    model = model_fn(FLAGS.model, {'num_classes': dataset_classes})
  else:
    model = model_fn(pretrained=False, num_classes=dataset_classes)
  
  compose_trans = transforms.Compose([
    transforms.ToTensor()
  ])
  train_dataset = dataset_fn(FLAGS.dataset_dir, transform=compose_trans, train=True, download=True)
  val_dataset = dataset_fn(FLAGS.dataset_dir, transform=compose_trans, train=False, download=True)

  train_loader = DataLoader(train_dataset, 
                            batch_size=FLAGS.batch_size, 
                            shuffle=True, 
                            num_workers=2)
  device = torch.device("cuda" if FLAGS.use_cuda else "cpu")

  model = model.to(device)
  optimizer = optim.Adam(model.parameters(), lr=0.0001)
  loss_op = torch.nn.CrossEntropyLoss()
  start_time = time.time()
  try:
    if _cudart is not None:
      status = _cudart.cudaProfilerStart()
    else:
      status = None
    for epoch in range(1, FLAGS.max_epochs+1):
      train(logger, model, device, train_loader, optimizer, epoch, loss_op)
  finally:
    if status == 0:
      _cudart.cudaProfilerStop()
  final_time = time.time() - start_time
  logger.info("Finished: ran for %d secs", final_time)
if __name__ == "__main__":
  app.run(main)

