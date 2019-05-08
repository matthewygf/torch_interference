import logging
import sys
import ctypes

def get_logger(name):
  logger = logging.getLogger(name)

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
  return logger

def get_cudart():
  try:
    _cudart = ctypes.CDLL('libcudart.so')
  except:
    _cudart = None
  return _cudart