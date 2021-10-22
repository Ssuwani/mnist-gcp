import tensorflow as tf
import numpy as np
from tensorflow.python.lib.io import file_io
import os

BUCKET_DIR = "gs://suwan_bucket"
RAW_DATASET_DIR = "mnist/rawdata"
PROC_DATASET_DIR = "mnist/proc"

train_x = np.load(file_io.FileIO(os.path.join(BUCKET_DIR, RAW_DATASET_DIR, "train_x.npy"), 'rb'))
train_y = np.load(file_io.FileIO(os.path.join(BUCKET_DIR, RAW_DATASET_DIR, "train_y.npy"), 'rb'))
test_x = np.load(file_io.FileIO(os.path.join(BUCKET_DIR, RAW_DATASET_DIR, "test_x.npy"), 'rb'))
test_y = np.load(file_io.FileIO(os.path.join(BUCKET_DIR, RAW_DATASET_DIR, "test_y.npy"), 'rb'))

train_x = train_x / 255.0
test_x = test_x / 255.0

np.save(file_io.FileIO(os.path.join(BUCKET_DIR, PROC_DATASET_DIR, "train_x.npy"), 'w'), train_x)
np.save(file_io.FileIO(os.path.join(BUCKET_DIR, PROC_DATASET_DIR, "train_y.npy"), 'w'), train_y)
np.save(file_io.FileIO(os.path.join(BUCKET_DIR, PROC_DATASET_DIR, "test_x.npy"), 'w'), test_x)
np.save(file_io.FileIO(os.path.join(BUCKET_DIR, PROC_DATASET_DIR, "test_y.npy"), 'w'), test_y)
