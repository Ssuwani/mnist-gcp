def collect_data(raw_data_dir):
    import tensorflow as tf
    import numpy as np
    from tensorflow.python.lib.io import file_io
    import os
    
    mnist = tf.keras.datasets.mnist
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    
    np.save(file_io.FileIO(os.path.join(raw_data_dir, "train_x.npy"), 'w'), train_x)
    np.save(file_io.FileIO(os.path.join(raw_data_dir, "train_y.npy"), 'w'), train_y)
    np.save(file_io.FileIO(os.path.join(raw_data_dir, "test_x.npy"), 'w'), test_x)
    np.save(file_io.FileIO(os.path.join(raw_data_dir, "test_y.npy"), 'w'), test_y)
    
    print("-"*15)
    print("데이터 저장 위치: ", raw_data_dir)
    print("-"*15)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--raw_data_dir', type=str, default="gs://suwan_bucket/mnist/rawdata")
    
    args = parser.parse_args()
    
    collect_data(
        args.raw_data_dir
    )
    