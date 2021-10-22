import tensorflow as tf
import numpy as np
from tensorflow.python.lib.io import file_io
import os


mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

BUCKET_DIR = "gs://suwan_bucket"
DATASET_DIR = "mnist/rawdata"

np.save(file_io.FileIO(os.path.join(BUCKET_DIR, DATASET_DIR, "train_x.npy"), 'w'), train_x)
np.save(file_io.FileIO(os.path.join(BUCKET_DIR, DATASET_DIR, "train_y.npy"), 'w'), train_y)
np.save(file_io.FileIO(os.path.join(BUCKET_DIR, DATASET_DIR, "test_x.npy"), 'w'), test_x)
np.save(file_io.FileIO(os.path.join(BUCKET_DIR, DATASET_DIR, "test_y.npy"), 'w'), test_y)