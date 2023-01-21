# VISUAL POLLUTION EFFICIENTDET

A pre-trained model for object detection **efficientDet(D0)** for 11 classes of different visual pollution using **TensorFlow Object Detection API**.

## OVERVIEW

We trained [efficentDet(D0)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) on 11 classes of visual_pollution images that are taken from restricted geographic ares in KSA.

The classes:

- GRAFFITI
- FADED SIGNAGE
- POTHOLES
- GARBAGE
- CONSTRUCTION ROAD
- BROKEN_SIGNAGE
- BAD STREETLIGHT
- BAD BILLBOARD
- SAND ON ROAD
- CLUTTER_SIDEWALK
- UNKEPT_FACADE

## HOW TO RUN PICTURE INFERENCE:

1. First check out the installation of **TensorFlow object detection API** [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html), otherwise you won't able to run the code

2. Download _visual_pollution_efficientDet.zip_ file

3. You need to place _models_ folder from tensorflow API and place it inside the project folder.

4. Install all the requirements

```
    pip install -r requirements.txt
```

5. Open your terminal at _scripts_ folder directory and activate your virtual environment **(tensorflow)**

6. Run this command

   Replace the following:

| variable            | description                                                                |
| ------------------- | :------------------------------------------------------------------------- |
| PATH_TO_SAVED_MODEL | saved model folder in _exported-models\\our_efficientdet_d0_coco17_tpu-32_ |
| PATH_TO_MAP         | _label_map.pbtxt_                                                          |
| PATH_TO_IMAGES      | _images_ folder                                                            |
| PATH_TO_TEST        | _test.csv_                                                                 |
| PATH_TO_OUTPUT      | choose whatever location you want but make it ends with _.csv_             |

```
python pic_inference.py -sm PATH_TO_SAVED_MODEL -ld PATH_TO_MAP -id PATH_TO_IMAGES -td PATH_TO_TEST -o PATH_TO_OUTPUT
```

7. Done! you can find the predictions in this format **class,image_path,name,xmax,xmin,ymax,ymin** inside the _output.csv_ file

## HOW TO RUN TFRECORD CONVERTER:

We customize a converter to suits our dataset, if you wanna try it follow these steps:

1. Your dataset must have these **class,image_path,name,xmax,xmin,ymax,ymin** in a _csv_ file with _images_ folder for the whole images

2. Open your terminal at _scripts_ folder directory and activate your virtual environment

3. Run this command:

   Replace the following:

| variable       | description           |
| -------------- | :-------------------- |
| PATH_TO_TRAIN  | _train.tfrecord_ path |
| PATH_TO_TEST   | _test.tfrecord_ path  |
| PATH_TO_IMAGES | _images_ folder       |
| PATH_TO_CSV    | _dataset.csv_ path    |

```
python TFRecordConverter.py -tr PATH_TO_TRAIN -te PATH_TO_TEST -i PATH_TO_IMAGES -c PATH_TO_CSV
```

## HOW TO RUN VISUALIZE:

1. Open your terminal at _scripts_ folder directory and activate your virtual environment **(tensorflow)**

2. Run this command

   Replace the following:

| variable            | description                                                                |
| ------------------- | :------------------------------------------------------------------------- |
| PATH_TO_SAVED_MODEL | saved model folder in _exported-models\\our_efficientdet_d0_coco17_tpu-32_ |
| PATH_TO_MAP         | _label_map.pbtxt_                                                          |
| PATH_TO_IMAGES      | _images_ folder                                                            |
| PATH_TO_TEST        | _test.csv_                                                                 |
| PATH_TO_OUTPUT      | output image folder                                                        |
| NUM_OF_IMAGES       | number of output images                                                    |

```
python visualize.py -sm PATH_TO_SAVED_MODEL -ld PATH_TO_MAP -id PATH_TO_IMAGES -td PATH_TO_TEST -o PATH_TO_OUTPUT -n NUM_OF_IMAGES
```

## NOTES:

- We used _model_main_tf2.py_ to train and evaluate the model, and _exporter_main_v2_ to export the model

- _TFRecordConverter.py_ shuffle the dataset, so each time you run it, you will get different _train.tfrecord_ and _test.tfrecord_

- in _pic_inference.py_ we coded **non max suppression** algorithm, we checked the test images and we saw many overlapping, so we did it ourselves with a little help from this [medium webpage](https://towardsdatascience.com/non-maxima-suppression-139f7e00f0b5) and [stackoverflow](https://stackoverflow.com/) of course.

- mAP is low for many reasons: computing power limitation, human generated dataset, shortage in time, and we still beginners, its our first time working with CNN :)

## SOFTWARE REFERENCES:

- **TensorFlow Model Garden**

    **_Authors:_** Hongkun Yu, Chen Chen, Xianzhi Du, Yeqing Li, Abdullah Rashwan, Le Hou, Pengchong Jin, Fan Yang, Frederick Liu, Jaeyoun Kim, and Jing Li

    **_Date:_** 2020

    **_Code version:_** 2.0

    **_Availability:_** https://github.com/tensorflow/models

- **Automl EfficientDet**

    **_Authors:_** Mingxing Tan, Ruoming Pang, Quoc V. Le

    **_Date:_** 2020

    **_Code version:_** 7.0

    **_Availability:_** https://github.com/google/automl/tree/master/efficientdet

## THANK YOU!

Hope you enjoy this model :)

If you face any problems please contact me

Made by: xEra team

For any suggestions/comments/questions email us!
| | |
|---------------|----------------------|
| Reem Hejazi | 219410002@psu.edu.sa |
| Nour Alakhras | 220410351@psu.edu.sa |
| Ayat Abodayeh | 220410035@psu.edu.sa |
