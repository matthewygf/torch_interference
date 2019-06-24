import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app
from absl import flags
from tf_image_models.densenet_tf import *
from tf_image_models.vgg import *
from tf_image_models.resnet import *

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cifar10', 'The dataset to train with')
flags.DEFINE_string('dataset_dir', 'data', 'Dataset directory')
flags.DEFINE_integer('batch_size', 64, 'Batch size of the model training')
flags.DEFINE_integer('max_epochs', 5, 'maximum number of epochs to run')
flags.DEFINE_string('model', None, 'The model you want to test')

flags.mark_flag_as_required('dataset')
flags.mark_flag_as_required('dataset_dir')

models_factory = {
  # TODO: OOM
  'densenet121': densenet121,
  # TODO: BROKEN :/
  'densenet40': densenet40,
  'vgg19': vgg19,
  'resnet18': resnet18,
  'resnet50': resnet50
}

# NOTE: KERAS has to use tuple dataset.
def one_hot(data, num_classes):
  images, labels = data['image'], data['label']
  labels = tf.keras.backend.one_hot(labels, num_classes)
  return (images, labels)

def _transpose_data(images, labels):
  images = tf.transpose(images, perm=[0,3,2,1])
  return (images, labels)

def main(_):
  tf.keras.backend.clear_session()

  data, info = tfds.load(FLAGS.dataset, data_dir=FLAGS.dataset_dir, with_info=True)
  train_data, test_data = data['train'], data['test']
  assert isinstance(train_data, tf.data.Dataset)

  gpu_available = tf.test.is_built_with_cuda() and tf.test.is_gpu_available()

  train_data = train_data.shuffle(FLAGS.batch_size).batch(FLAGS.batch_size)
  test_data = test_data.shuffle(FLAGS.batch_size).batch(FLAGS.batch_size)

  train_data = train_data.map(lambda x : one_hot(x, info.features['label'].num_classes))
  test_data = test_data.map(lambda x : one_hot(x, info.features['label'].num_classes))

  # NOTE: CHECK FOR CHANNEL LAST
  is_channel_last = info.features['image'].shape[-1] == 3
  if gpu_available and is_channel_last:
    # then we transpose to channel first
    train_data = train_data.map(_transpose_data)
    test_data = test_data.map(_transpose_data)
  
  train_data = train_data.repeat()
  test_data = test_data.repeat()

  data_format = 'channels_first' if gpu_available else 'channels_last'
  input_shape = tf.compat.v1.data.get_output_shapes(train_data)[0][1:]
  model_args = dict(
    num_classes=info.features['label'].num_classes, 
    input_shape=input_shape,
    data_format=data_format
  )

  # NOTE: update to v1 compat get_ouput_xxxx when using v2
  model = models_factory[FLAGS.model](**model_args)
  # TODO: TF KERAS CALLBACK LEARNING RATE SCHEDULER
  model.compile(optimizer=tf.train.GradientDescentOptimizer(0.001),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.CategoricalAccuracy()]) 

  steps_per_epoch = info.splits['train'].num_examples // FLAGS.batch_size + 1
  valid_steps = info.splits['test'].num_examples // FLAGS.batch_size + 1
  
  # NOTE: KERAS has to use tuple, when feeding tf.data.dataset
  model.fit(train_data, epochs=FLAGS.max_epochs, steps_per_epoch=steps_per_epoch,
            validation_data=test_data, validation_steps=valid_steps)

  print(model.summary())

  # Clear the session explicitly to avoid session delete error
  tf.keras.backend.clear_session()


if __name__ == "__main__":
  app.run(main)