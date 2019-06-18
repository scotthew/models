# Imports
from utils import visualization_utils as vis_util
from utils import label_map_util
from object_detection.utils import ops as utils_ops
import numpy as np
import os
import six.moves.urllib as urllib
import sys, traceback
import tarfile
import tensorflow as tf
import zipfile
import itertools
import time
import ntpath

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Using RC
# if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
#  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

print("TensorFlow Version: " + tf.__version__)

# Env setup - Jupyter
# This is needed to display the images.
# %matplotlib inline

# Object detection imports


# Model preparation
# What model to download.
# See tensorflow object model zoo https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

# about 3 seconds an image.  OK detection.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'

# about 6 seconds an image.  Good at bike.
#MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'

# about 17 seconds an image
#MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'

# about 26 seconds an image.  Way more detection like tree's and women.
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12'

# TODO: try face detection... facessd_mobilenet_v2_quantized_320x320_open_image_v4
# Initial Test Resulted in error
#MODEL_NAME = 'facessd_mobilenet_v2_quantized_320x320_open_image_v4'

# Oic with decent speed.  13 seconds an image. Detection is not good.
#MODEL_NAME = 'ssd_mobilenet_v2_oid_v4_2018_12_12'

# Best coco mAP.  20-30 seconds per image.  So far the best but very slow.
#MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28'

MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
# COOC Labels
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# OIC V4 Labels
PATH_TO_LABELS = os.path.join('data', 'oid_v4_label_map.pbtxt')

# Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

# Load a (frozen) Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)

# Helper code


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
#PATH_TO_TEST_IMAGES_DIR = ntpath.join('test_images','FrontCam', 'Missed')
PATH_TO_TEST_IMAGES_DIR = ntpath.join('test_images', 'FrontCam')
TEST_IMAGE_PATHS = [os.path.join(
    PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(18523, 23969)]
#PATH_TO_TEST_IMAGES_OUT_DIR = ntpath.join('test_images_out', 'FrontCam', 'Missed')
PATH_TO_TEST_IMAGES_OUT_DIR = ntpath.join('test_images_out', 'FrontCam')

# Size, in inches, of the output images.
# inches = pixel/DPI
# 1280x720 = 13.3x7.5
IMAGE_SIZE = (13.3, 7.5)


def run_inference_for_single_image(image, graph, sess):
    # with tf.compat.v1.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # Get handles to input and output tensors
    ops = tf.compat.v1.get_default_graph().get_operations()
    all_tensor_names = {
        output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                tensor_name)
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(
            tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(
            tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(
            tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [
            real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [
            real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: image})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(
        output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def output_image_to_file(image_path, image_np):
    # Convert to image and write to out file
    out_image_path = ntpath.join(
        PATH_TO_TEST_IMAGES_OUT_DIR, ntpath.basename(image_path))
    print("Out Image Name: %s" % (out_image_path))
    out_img = Image.fromarray(image_np)
    out_img.save(out_image_path, "jpeg")
    return


def calculate_image_processing_time(process_start_time, total_process_time, image_index, test_image_len):
    process_end_time = time.time()
    image_process_time = process_end_time - process_start_time
    print("Image Processing Time: %s" % (image_process_time))

    total_process_time += image_process_time
    print("Total Processing Time: %s" % (total_process_time))

    avg_processing_time = total_process_time / image_index
    print("Average Processing Time: %s" % (avg_processing_time))

    remaining_processing_time = (
        test_image_len - image_index) * avg_processing_time
    print("Remaining Processing Time: %s minutes, %s hours" % (
        (remaining_processing_time / 60), ((remaining_processing_time / 60) / 60)))
    return total_process_time

# Jupyter...
# This is needed to display the images.
# %matplotlib inline


threshold = 0.50

test_image_len = len(TEST_IMAGE_PATHS)
print("Total Images To Process: %s" % (test_image_len))

total_process_time = 0
image_index = 1

objects_detected_dict = dict()

start_time = time.time()
#print("Start Time: %s" % (start_time))

with detection_graph.as_default() as graph:
    with tf.compat.v1.Session() as sess:
        for image_path in TEST_IMAGE_PATHS:
            try:
                #print("Image Path: %s" % (image_path))
                print("Processing image #%s, Path: %s" % (image_index, image_path))

                # TODO: split this out into functions
                process_start_time = time.time()

                image = Image.open(image_path)

                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                output_dict = run_inference_for_single_image(
                    image_np_expanded, graph, sess)
                #print("Output Dict: %s" % (output_dict))

                has_detected_obj = any(
                    i >= threshold for i in output_dict['detection_scores'])
                #print("Objest Detected: %s" % (has_detected_obj))
                if has_detected_obj:
                    # Select all class ids with a score >= threshold
                    selectors = [
                        x >= threshold for x in output_dict['detection_scores']]
                    detected_class_ids = list(itertools.compress(
                        output_dict['detection_classes'], selectors))
                    #print("Class Ids Found: %s" % (detected_class_ids))

                    # Labes for mscoco_label_map
                    #filter_by_class_id = False
                    filter_by_class_id = True

                    # coco filter list
                    #label_filter_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 25, 37, 38, 42, 62, 64, 72]
                    # oid v4 filter list
                    label_filter_list = [50, 89, 99, 103, 160, 218, 241, 318, 334, 366,
                                        391, 393, 400, 404, 409, 444, 456, 462, 485, 535, 571]

                    if filter_by_class_id and set(detected_class_ids).issubset(label_filter_list):
                        print(
                            "All detected classes are ignored.  Image will not be visualized.")
                    else:
                        detected_class_names = [category_index[i]['name']
                                                for i in detected_class_ids]
                        #print("Class Names: %s" % (detected_class_names))
                        #print("Output Dict: %s" % (output_dict))
                        #print("Category Index: %s" % (category_index))

                        #print("Objest Was Detected.  Image Path: %s, Class Names: %s" % (image_path, detected_class_names))
                        objects_detected_dict[image_path] = detected_class_names

                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            output_dict['detection_boxes'],
                            output_dict['detection_classes'],
                            output_dict['detection_scores'],
                            category_index,
                            instance_masks=output_dict.get('detection_masks'),
                            use_normalized_coordinates=True,
                            line_thickness=1)
                        # line_thickness=2)
                        # thickness of 8 is too thick for distant objects.
                        # line_thickness=8)

                        # Used to visualize the image in matplotlib.
                        # plt.figure(figsize=IMAGE_SIZE)
                        # plt.imshow(image_np)

                        output_image_to_file(image_path, image_np)

                total_process_time = calculate_image_processing_time(
                    process_start_time, total_process_time, image_index, test_image_len)
                image_index += 1
            except:
                print("Failed to process image.  Image Path: %s" % (image_path))
                print("Unexpected error:", sys.exc_info()[0])
                traceback.print_exc()
                total_process_time = calculate_image_processing_time(
                    process_start_time, total_process_time, image_index, test_image_len)
                image_index += 1
                continue

# Calculate the total processing time.
end_time = time.time()
#print("End Time: %s" % (end_time))
total_processing_time = (end_time - start_time)
print("Total Processing Time: %s minutes, %s hours" %
      ((total_processing_time / 60), ((total_processing_time / 60) / 60)))

PATH_TO_OBJECT_RESULT_FILE = ntpath.join(PATH_TO_TEST_IMAGES_OUT_DIR, 'detection_results.json')
object_result_files = open(PATH_TO_OBJECT_RESULT_FILE, "w")
object_result_files.writelines(objects_detected_dict)
object_result_files.close()
print("Detected Files Writen To File: %s" % (PATH_TO_OBJECT_RESULT_FILE))
#print("Objects Detected:\n%s" % (objects_detected_dict))
