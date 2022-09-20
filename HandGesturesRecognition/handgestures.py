from re import I
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from Tensorflow.resources import constants

WORKSPACE_PATH = 'Tensorflow/resources'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
CONFIG_PATH = "/Tensorflow/resources/models/my_ssd_mobnet/pipeline.config"
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'



# labels = [{'name':'hello', 'id':1}, 
#           {'name':'thanks', 'id':2}, 
#           {'name':'yes', 'id':3},
#           {'name':'no', 'id':4},
#           {'name':'iloveyou', 'id':5},
#           {'name':'sleep', 'id':6},
#           {'name':'stop', 'id':7},
#           {'name':'sad', 'id':8},
#           {'name':'play', 'id':9},
#           {'name':'play', 'id':10},
#           ]


# Creating annotations
# with open('HandGesturesRecognition\\Tensorflow\\resources\\annotations\\maps.pbtxt','w') as file:
    
#     for label in labels:
#         file.write('item {\n')
#         file.write('\tname:\'{}\'\n'.format(label['name']))
#         file.write('\tid:\'{}\'\n'.format(label['id']))
#         file.write('}\n')
    

# configurations for transfer learning

# configuration = config_util.get_configs_from_pipeline_file(constants.CONFIGURATION_FILE_PATH)

# configuration

# pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
# with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:                                                                                                                                                                                                                     
#     proto_str = f.read()
#     text_format.Merge(proto_str, pipeline_config)  

# print(CONFIG_PATH)
# print(PRETRAINED_MODEL_PATH+'\ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8\checkpoint\ckpt-0')
# print(ANNOTATION_PATH + '\label_map.pbtxt')
# print([ANNOTATION_PATH + '\\train.record'])
# print(ANNOTATION_PATH + '\label_map.pbtxt')
# print([ANNOTATION_PATH + '\\test.record'])


# pipeline_config.model.ssd.num_classes = 2
# pipeline_config.train_config.batch_size = 4
# pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
# pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
# pipeline_config.train_input_reader.label_map_path= ANNOTATION_PATH + '/label_map.pbtxt'
# pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
# pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
# pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']




# config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
# with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:                                                                                                                                                                                                                     
#     f.write(config_text)
#     print("Written config file")
    

# print(" ")
# print(" ")

print("""python {0}/research/object_detection/model_main_tf2.py --model_dir={1}/{2} --pipeline_config_path={3}/{4}/pipeline.config --num_train_steps=15000""".format(APIMODEL_PATH, MODEL_PATH,CUSTOM_MODEL_NAME,MODEL_PATH,CUSTOM_MODEL_NAME))

print("""python {}/research/object_detection/model_main_tf2.py --model_dir={}/{} --pipeline_config_path={}/{}/pipeline.config --num_train_steps=15000""".format(APIMODEL_PATH, MODEL_PATH,CUSTOM_MODEL_NAME,MODEL_PATH,CUSTOM_MODEL_NAME))


import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

configs = config_util.get_configs_from_pipeline_file("D:\python codes\DSHack-Kiters\HandGesturesRecognition\Tensorflow\\resources\models\my_ssd_mobnet\pipeline.config")
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)


ckpt.restore(os.path.join(CHECKPOINT_PATH,'ckpt-7')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections   




import cv2
import numpy as np
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from Tensorflow.resources import constants
import tensorflow as tf



category_index = label_map_util.create_category_index_from_labelmap(constants.ANNOTATION_PATH + '/maps.pbtxt')

capture = cv2.VideoCapture(0)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


while True: 
    ret, frame = capture.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        capture.release()
        break
    
    
    
detections = detect_fn(input_tensor)

print(detections)