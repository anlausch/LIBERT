#! /usr/bin/env python
import os
import tensorflow as tf
import argparse
from socket import gethostname


def set_gpus_from_args(args):
  if args.cuda_visible_devices:
    cuda_devices = args.cuda_visible_devices
  else:
    return None

  gpu_fraction = 1.0
  if args.gpu_fraction:
    gpu_fraction = args.gpu_fraction

  return set_gpus(visible_devices=cuda_devices, gpu_fraction=gpu_fraction)


def disable_all_gpus():
  os.environ["CUDA_VISIBLE_DEVICES"] = ""


def set_train_gpus():
  import config as c
  return set_gpus(c.TRAIN_CUDA_VISIBLE_DEVICES, c.TRAIN_CUDA_GPU_FRAC)


def set_dev_gpus():
  import config as c
  return set_gpus(c.DEV_CUDA_VISIBLE_DEVICES, c.DEV_CUDA_GPU_FRAC)


def set_gpus(visible_devices="", gpu_fraction=1):
  os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
  perc = gpu_fraction / 1.0 * 100
  tf.logging.warning("Using these GPUs: %s with %s%%  capacity (hostname: %s)" %
                     (str(os.environ["CUDA_VISIBLE_DEVICES"]), str(perc), gethostname()))
  if gpu_fraction == 1 or gpu_fraction == 1.0:
    return None
  return tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)


def _launch_dummy_process(args):
  gpu_options = set_gpus_from_args(args)

  n = 10000 # 8192
  dtype = tf.float32
  with tf.device("/gpu:0"):
    matrix1 = tf.Variable(tf.ones((n, n), dtype=dtype))
    matrix2 = tf.Variable(tf.ones((n, n), dtype=dtype))
    product = tf.matmul(matrix1, matrix2)

  graph_options = tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0))
  if gpu_options:
    config = tf.ConfigProto(graph_options=graph_options, gpu_options = gpu_options)
  else:
    config = tf.ConfigProto(graph_options=graph_options)

  sess = tf.Session(config=config)
  sess.run(tf.global_variables_initializer())
  sess.run(product.op)
  while True:
    sess.run(product.op)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Launch this file to occupy a GPU, "
                                               "kill this process to release resource")
  parser.add_argument("-d", "--cuda_visible_devices", type=str, help="visible GPUs")
  parser.add_argument("-f", "--gpu_fraction", type=float, help="percentage of GPU memory to occupy.")
  arguments = parser.parse_args()
  _launch_dummy_process(arguments)
