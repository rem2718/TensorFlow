'''
    this file convert csv files to tfrecord
    the csv file must have these headers: class,image_path,name,xmax,xmin,ymax,ymin
'''

import tensorflow.compat.v1 as tf
import pandas as pd
import os
from PIL import Image
import argparse
from collections import namedtuple
from object_detection.utils import dataset_util
from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

# Flags
parser = argparse.ArgumentParser(description="TFRecord Converter")
parser.add_argument("-tr", "--train_path", type=str)                # path to train.tfrecord
parser.add_argument("-te", "--test_path", type=str)                 # path to test.tfrecord
parser.add_argument("-i", "--image_dir", type=str)                  # path to images folder
parser.add_argument("-c", "--csv_path", type=str)                   # path to dataset(csv)

args = parser.parse_args()

def serialize_example(features):
    '''
        this function convert dataframe row to example and serialize it
    '''
    feature = {
        'image/height': dataset_util.int64_feature(features[0]),
        'image/width': dataset_util.int64_feature(features[1]),
        'image/filename': dataset_util.bytes_feature(features[2]),
        'image/source_id': dataset_util.bytes_feature(features[2]),
        'image/encoded': dataset_util.bytes_feature(features[3]),
        'image/format': dataset_util.bytes_feature(features[4]),
        'image/object/bbox/xmin': dataset_util.float_list_feature(features[5]),
        'image/object/bbox/xmax': dataset_util.float_list_feature(features[6]),
        'image/object/bbox/ymin': dataset_util.float_list_feature(features[7]),
        'image/object/bbox/ymax': dataset_util.float_list_feature(features[8]),
        'image/object/class/text': dataset_util.bytes_list_feature(features[9]),
        'image/object/class/label': dataset_util.int64_list_feature(features[10]),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def split(df, group):
    '''
        this function group the rows by the image_path
    '''
    data = namedtuple('data', ['image_path', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def to_list(group, path):
    '''
        this function prepare the records to be stored in the tfrecord
    '''
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.image_path)), 'rb') as fid:
        encoded_jpg = fid.read()
    img = Image.open(os.path.join(path, group.image_path))
    width, height = img.size
    filename = group.image_path[:-4].encode('utf8')
    format = b'jpg'
    xmins, xmaxs, ymins, ymaxs, names, labels = [], [], [], [], [], [] 
    for index, row in group.object.iterrows():
        xmins.append((row['xmin'] * 2) / width) # we multiply them by 2 to suits 1920x1080px
        xmaxs.append((row['xmax'] * 2) / width)
        ymins.append((row['ymin'] * 2) / height)
        ymaxs.append((row['ymax'] * 2) / height)
        names.append(row['name'].encode('utf8'))
        labels.append(int(row['class']) + 1)
    return [height, width, filename, encoded_jpg, format, xmins, xmaxs, ymins, ymaxs, names, labels]

def main(_):
    path = os.path.join(args.image_dir)
    df = pd.read_csv(args.csv_path)
    df_after_grouping = pd.DataFrame(columns = ['height', 'width', 'filename', 'encoded_jpg', 'format', 'xmins', 'xmaxs', 'ymins', 'ymaxs', 'names', 'labels'])
    
    grouped = split(df, 'image_path')
    for group in grouped:                                                   # process each image and store them in a new df
        df_after_grouping.loc[len(df_after_grouping)] = to_list(group, path)

    train, test = train_test_split(df_after_grouping, train_size= 0.8)      # split the dataset for training and evaluation
    print(f'train shape: {train.shape}\n test shape: {test.shape}')    

    writer = tf.python_io.TFRecordWriter(args.train_path)                   # write into train.tfrecord
    for index,row in train.iterrows():
        example = serialize_example(row.tolist())
        writer.write(example)
    writer.close()

    writer2 = tf.python_io.TFRecordWriter(args.test_path)                  # write into test.tfrecord
    for index,row in test.iterrows():
        example = serialize_example(row.tolist())
        writer2.write(example)
    writer2.close()

if __name__ == '__main__':
    tf.app.run()
    


