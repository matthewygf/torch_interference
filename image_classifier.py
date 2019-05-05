from absl import app
from absl import flags
import logging
import sys
# using predefined set of models
import torchvision.models as models
import torchvision.datasets as predefined_datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import time
import ctypes
    
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

flags.mark_flag_as_required('run_name')
flags.mark_flag_as_required('model')
flags.mark_flag_as_required('dataset_dir')

models_factory = {
  'googlenet': models.googlenet,
  'mobilenet': models.mobilenet_v2,
  'resnet': models.resnet50,
  'vgg19': models.vgg19
}

datasets_factory = {
  'cifar10': predefined_datasets.CIFAR10
}

def train(logger, model, device, train_loader, optimizer, epoch, loss_op):
  model.train()

  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    start_time = time.time()
    pred = model(data)
    loss = loss_op(pred, target)
    loss.backward()
    optimizer.step()
    time_elapsed = time.time() - start_time

    if batch_idx % FLAGS.log_interval == 0:
      logger.info("Epoch %d: %d/%d [Loss: %.4f] (%.4f sec/step)", 
                  epoch, batch_idx*len(data), 
                  len(train_loader.dataset), loss.item(), time_elapsed)


def main(argv):
  del argv
  
  logger = logging.getLogger(__name__+FLAGS.run_name)

  formatter = logging.Formatter(
    "%(asctime)s:%(filename)s:%(funcName)s:%(lineno)d-%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
  )

  handler = logging.StreamHandler(sys.stdout)
  handler.setFormatter(formatter)
  
  if (logger.hasHandlers()):
    logger.handlers.clear()
  
  logger.setLevel(logging.INFO)
  logger.addHandler(handler)
  logger.propagate = False
  logger.info("run: %s, specified model: %s, dataset: %s", FLAGS.run_name, FLAGS.model, FLAGS.dataset)

  try:
    _cudart = ctypes.CDLL('libcudart.so')
  except:
    _cudart = None
    logger.warning("No cudart, probably means you do not have cuda on this machine.")

  model_fn = models_factory[FLAGS.model]
  dataset_fn = datasets_factory[FLAGS.dataset]
  if 'google' in FLAGS.model: 
    model = model_fn(pretrained=False, transform_input=False, aux_logits=False, num_classes=10)
  else:
    model = model_fn(pretrained=False, num_classes=10)

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

  try:
    if _cudart is not None:
      status = _cudart.cudaProfilerStart()
    else:
      status = None
    
    for epoch in range(1, 6):
      train(logger, model, device, train_loader, optimizer, epoch, loss_op)
  finally:
    if status == 0:
      _cudart.cudaProfilerStop()

if __name__ == "__main__":
  app.run(main)

