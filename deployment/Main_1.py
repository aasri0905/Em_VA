import keras.backend as K
import sys
sys.path.append("./")
from datetime import datetime
import cv2


from deployment.tensorflow_detector import *
from deployment.utils import label_map_util
from deployment.utils import visualization_utils_color as vis_util
from deployment.video_threading_optimization import *
import numpy as np
import matplotlib.pyplot as plt


## OPTIONS ##
path_main = "./deployment/frozen_graphs"
PATH_TO_CKPT = path_main+'/frozen_inference_graph_face.pb'
PATH_TO_CLASS = path_main+'/classificator_full_model.pb'
PATH_TO_REGRESS = path_main+'/regressor_full_model.pb'
label_map = label_map_util.load_labelmap('./deployment/protos/face_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
detector = TensorflowDetector(PATH_TO_CKPT, PATH_TO_CLASS, PATH_TO_REGRESS)



def Primary(image):

    boxes, scores, classes, num_detections, emotions_print, valence, arousal,emotions_score = detector.run(image)


    text = "classes: {}".format(emotions_print)
    cv2.putText(image, text, org=(25, 25), fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=0.35, color=(0, 255, 0))
    frame = vis_util.visualize_boxes_and_labels_on_image_array(image,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=1)

    try:
        valence = float(valence[0])
        arousal = float(arousal[0])
        emotions_print = emotions_print[0]
        emotions_score = float(emotions_score[0])
    except:
        pass
    
    if type(valence) == list:
        return 0, 0, 0, 0, 0
    else:
        return valence, arousal, emotions_print,emotions_score, boxes
