'''
    this file read test.csv(list if test images) for piture inference and store them in output.csv
'''

import time
import tensorflow as tf
import tensorflow.compat.v1 as tft
import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


# Flags
parser = argparse.ArgumentParser(description="pic_inference")
parser.add_argument("-sm", "--saved_model", type=str)               # path to saved_model folder
parser.add_argument("-ld", "--label_dir", type=str)                 # path to label_map.pbtxt
parser.add_argument("-id", "--image_dir", type=str)                  # path to images folder
parser.add_argument("-td", "--test_dir", type=str)                   # path to test.csv
parser.add_argument("-o", "--output", type=str)                     # path to output.csv

args = parser.parse_args()

def overlap(box1, box2):
    '''
        this function check if two boxes intersect or not
    '''
    if (box2[3] > box1[1] and box2[3] < box1[3]) or (box2[1] > box1[1] and box2[1] < box1[3]):
        x_match = True
    else:
        x_match = False
    if (box2[2] > box1[0] and box2[2] < box1[2]) or (box2[0] > box1[0] and box2[0] < box1[2]):
        y_match = True
    else:
        y_match = False
    if x_match and y_match:
        return True
    else:
        return False

def NMS(boxess, threshold, width, height):
    '''
        this function perform non max suppression algorithm and return the indices of the valid bounding boxes 
    '''
    boxes = np.copy(boxess)
    if boxes.shape[0] == 1:
        return [0]
    xmin = boxes[:,1] = (boxes[:,1] * width).astype(np.int64)
    ymin = boxes[:,0] = (boxes[:,0] * height).astype(np.int64)
    xmax = boxes[:,3] = (boxes[:,3] * width).astype(np.int64)
    ymax = boxes[:,2] = (boxes[:,2] * height).astype(np.int64)
    areas = (xmax - xmin) * (ymax - ymin) 
    indices = np.arange(len(xmin))
    areas = np.flip(areas,0)
    boxes = np.flip(boxes,0).astype(np.int64)                          # flip the order of the boxes since they are ordered descendingly
    
    for i,box in enumerate(boxes):                                     #check the iou between a box and all the others
        temp_indices = indices[indices!=i]
        for index in temp_indices:
            if not overlap(box, boxes[index]):
                temp_indices = np.setdiff1d(temp_indices, index)
        xx1 = np.maximum(box[1], boxes[temp_indices,1])
        yy1 = np.maximum(box[0], boxes[temp_indices,0])
        xx2 = np.minimum(box[3], boxes[temp_indices,3])
        yy2 = np.minimum(box[2], boxes[temp_indices,2])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        iou = (w * h) / (areas[temp_indices] + areas[i] - (w * h))      # intersection over union
        if len(iou[iou > threshold]) > 0:                               # if any of the boxes has high iou then delete the box
            indices = indices[indices != i]
            
    indices = len(xmin) - indices - 1 # flip the indices to the original ordering
    return indices

def main(_):
    print('Loading model...', end='')
    start_time = time.time()
    # loading the model
    detect_fn = tf.saved_model.load(args.saved_model) 
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Done! Took {elapsed_time} seconds')
    
    # loading the label map
    category_index = label_map_util.create_category_index_from_labelmap(args.label_dir,use_display_name=True)
    # reading test.csv
    test = pd.read_csv(args.test_dir)
    IMAGE_PATHS = test['image_path'].apply(lambda x: args.image_dir + x).tolist()

    submission = pd.DataFrame(columns=['class', 'image_path', 'name', 'xmax', 'xmin', 'ymax', 'ymin'])
    # size of the pictures
    width = 1920
    height = 1080
    # we tried different values for the thresholds and we found those give us the best results
    score_threshold = .05
    iou_threshold = .05
    count = 0
    for image_path in IMAGE_PATHS: # infer each image
        path = image_path.replace(args.image_dir,'')
        print(f'Running inference for {path}...',end='')
        image_np = np.array(Image.open(image_path))
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key:value[0, :num_detections].numpy()
                    for key,value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            
        labels = detections['detection_classes']
        boxes = detections['detection_boxes']
        scores = detections['detection_scores']
        # deleting the images with score below the threshold 
        ind = []
        for j in range(len(scores)):
            if scores[j] < score_threshold:
                ind.append(j)
                            
        scores = np.delete(scores, ind, axis=0)
        boxes = np.delete(boxes, ind, axis=0)
        labels = np.delete(labels, ind, axis=0)   
        ind.clear() 
        
        # perform non max suppression
        indices = NMS(boxes,iou_threshold,width,height)
        # filtering them out
        labels = labels[indices]
        boxes = boxes[indices]
        scores = scores[indices]
        
        if len(indices) == 0: # if no bounding box is detected set it to 0
            label = 0
            path = image_path.replace(args.image_dir,'')
            text = 'background'
            xmin = 0
            xmax = 0
            ymin = 0
            ymax = 0 
            submission.loc[len(submission)] = [label, path, text, xmax, xmin, ymax, ymin] 
        else:
            for i in range(len(indices)):
                label = labels[i]
                path = image_path.replace(args.image_dir,'')
                text = category_index[label]['name']
                xmin = int(boxes[i][1] * width)
                xmax = int(boxes[i][3] * width)
                ymin = int(boxes[i][0] * height)
                ymax = int(boxes[i][2] * height) 
                submission.loc[len(submission)] = [label, path, text, xmax, xmin, ymax, ymin]
        
        count +=1
        print(f'Done {count}')
    # write the output.csv    
    submission.to_csv( args.output, index=False)

if __name__ == '__main__':
    tft.app.run()
    













