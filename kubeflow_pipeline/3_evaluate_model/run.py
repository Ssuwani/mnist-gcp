import tensorflow as tf
import numpy as np
from tensorflow.python.lib.io import file_io
import os

BUCKET_DIR = "gs://suwan_bucket"
PROC_DATASET_DIR = "mnist/proc"
MODEL_DIR = "mnist/model"


test_x = np.load(
    file_io.FileIO(os.path.join(BUCKET_DIR, PROC_DATASET_DIR, "test_x.npy"), "rb")
)
test_y = np.load(
    file_io.FileIO(os.path.join(BUCKET_DIR, PROC_DATASET_DIR, "test_y.npy"), "rb")
)

model = tf.keras.models.load_model(os.path.join(BUCKET_DIR, MODEL_DIR))

loss, acc = model.evaluate(test_x, test_y)

print("Model loss: {:.4f} acc: {:.4f}".format(loss, acc))
