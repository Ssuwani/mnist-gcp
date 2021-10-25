import tensorflow as tf
import numpy as np
from tensorflow.python.lib.io import file_io
import os
from model import build_mnist_model

BUCKET_DIR = "gs://suwan_bucket"
PROC_DATASET_DIR = "mnist/proc"
MODEL_DIR = "mnist/model"

train_x = np.load(
    file_io.FileIO(os.path.join(BUCKET_DIR, PROC_DATASET_DIR, "train_x.npy"), "rb")
)
train_y = np.load(
    file_io.FileIO(os.path.join(BUCKET_DIR, PROC_DATASET_DIR, "train_y.npy"), "rb")
)

model = build_mnist_model()

model.fit(train_x, train_y, epochs=3)

model.save(os.path.join(BUCKET_DIR, MODEL_DIR))
