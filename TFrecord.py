import os
from tqdm import tqdm
from glob import glob
import random
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(img_str, source_id, filename):
    # Create a dictionary with features that may be relevant.
    feature = {'image/source_id': _int64_feature(source_id),
               'image/filename': _bytes_feature(filename),
               'image/encoded': _bytes_feature(img_str)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(dataset_path, output_path):
    samples = []
    labels = []
    cnt = 0
    
    with open("labels.txt", "r") as file:
        labels = file.read().splitlines()
    
    print("Reading data list...")
    for id_name in tqdm(os.listdir(dataset_path)):
        img_paths = glob(os.path.join(dataset_path, id_name, '*.jpg'))
        #labels.append([cnt, id_name.split('_')[1]])
        try:
            cnt = labels.index(id_name.split('_')[1])
        except ValueError:
            print(id_name.split('_')[1] + "is not found")
        
        for img_path in img_paths:
            filename = os.path.join(id_name.split('_')[1], os.path.basename(img_path))
            samples.append((img_path, cnt, filename))
        
        #cnt += 1 
    random.shuffle(samples)

    print("Writing tfrecord file...")
    with tf.io.TFRecordWriter(output_path) as writer:
        for img_path, id_name, filename in tqdm(samples):
            tf_example = make_example(img_str=open(img_path, 'rb').read(),
                                     source_id=int(id_name),
                                     filename=str.encode(filename))
            writer.write(tf_example.SerializeToString())
    """
    with open('labels.txt', 'w') as f:
        for i in range(cnt):
            f.write(labels[i][1] + "\n") 
    """
if __name__ == "__main__":
    main("/media/sgwjl/DATA/송민석/brand/Validation/sample", "vaild.tfrecord")

