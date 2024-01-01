#!/usr/bin/python

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

window_name = "Real-time Line Chart"
cv2.namedWindow(window_name)
# Initialize chart parameters
num_points = 100  # Number of data points to display
data1 = np.zeros(num_points)
data2 = np.zeros(num_points)

history1 = []  # List to store historical data points for Line 1
history2 = []  # List to store historical data points for Line 2


# Create the chart figure and axes
fig, (ax1, ax2) = plt.subplots(2, 1)

# Create Line 1 and Line 2
line1, = ax1.plot(data1, color='red',label='Valence')
line2, = ax2.plot(data2, color='blue',label='Arousal')

ax1.legend()
ax2.legend()

# Update function for the chart
def update_chart():
    line1.set_ydata(data1)
    line2.set_ydata(data2)
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    fig.canvas.draw()

## OPTIONS ##
path_main = "/home/ronak/affnet/AffectNet/deployment/frozen_graphs"
PATH_TO_CKPT = path_main+'/frozen_inference_graph_face.pb'
PATH_TO_CLASS = path_main+'/classificator_full_model.pb'
PATH_TO_REGRESS = path_main+'/regressor_full_model.pb'
label_map = label_map_util.load_labelmap('/home/ronak/affnet/AffectNet/deployment/protos/face_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=2, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
detector = TensorflowDetector(PATH_TO_CKPT, PATH_TO_CLASS, PATH_TO_REGRESS)


# cap = cv2.VideoCapture("/home/ronak/Downloads/adi_trim.mp4")
cap = cv2.VideoCapture(0)


def track_bounding_box(frame):
    global refPt, drawing, initialized, tracker

    if not initialized:
        cv2.imshow("Frame", frame)
        return frame

    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(coord) for coord in bbox]
        x_min = x
        x_max = x + w
        y_min = y
        y_max = y + h
        cropped_frame = frame[y_min:y_max, x_min:x_max]
        # wid, hei = cropped_frame.shape[:2]
    return cropped_frame


refPt = []
drawing = False 
initialized = False
tracker = None

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global refPt, drawing, initialized, tracker

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        drawing = True
        initialized = False
        tracker = None

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        drawing = False
        initialized = True
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame, (refPt[0][0], refPt[0][1], abs(refPt[1][0] - refPt[0][0]), abs(refPt[1][1] - refPt[0][1])))
cv2.namedWindow("Frame")

cv2.setMouseCallback("Frame", mouse_callback)

# while not initialized:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     frame = track_bounding_box(frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

while True:
    ret, frame_ = cap.read()
    frame = track_bounding_box(frame_)
    cv2.imshow("choped",frame)
    [h, w] = frame.shape[:2]
    # image = frame[0:w//2, :]
    image = frame
    # image = cv2.flip(frame, 1)
    # flip_2 = cv2.flip(frame_, 1)
    cv2.imshow("Full",frame_)
    boxes, scores, classes, num_detections, emotions_print, valence, arousal = detector.run(image)
    try:
        valence = valence[0]
        arousal = arousal[0]

        data1 = np.roll(data1, -1)
        data1[-1] = valence
        data2 = np.roll(data2, -1)
        data2[-1] = arousal

        history1.append(valence)
        history2.append(arousal)


        # Trim the historical data lists to the desired length
        history1 = history1[-num_points:]
        history2 = history2[-num_points:]
        update_chart()
        chart_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        chart_image = chart_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # Display the chart image using cv2.imshow()
        cv2.imshow(window_name, chart_image)
    except:
        continue
    text = "classes: {}".format(emotions_print)
    cv2.putText(image, text, org=(25, 25), fontFace=cv2.FONT_HERSHEY_DUPLEX,fontScale=0.35, color=(0, 255, 0))
    vis_util.visualize_boxes_and_labels_on_image_array(image,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=1)
    # cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
    # canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # canvas[:640, :1280] = frame_
    # canvas[:480, 640:] = chart_image
    # canvas[:wid, :hei] = image
    cv2.imshow("Frame", image)

    k = cv2.waitKey(1) & 0xff
    if k == ord('q') or k == 27:
        break

cv2.destroyAllWindows()

# def predict_from_camera(detector):
#     print('Press q to exit')
#     vs = WebcamVideoStream(src=0).start()
#     fps = FPS().start()
#     window_not_set = True

#     while True:
#         global data1
#         global data2
#         global history1
#         global history2
#         # grab the frame from the threaded video stream
#         image = vs.read()
#         [h, w] = image.shape[:2]
#         image = cv2.flip(image, 1)

#         boxes, scores, classes, num_detections, emotions_print, valence, arousal = detector.run(image)
#         try:
#             valence = valence[0]
#             arousal = arousal[0]

#             data1 = np.roll(data1, -1)
#             data1[-1] = valence
#             data2 = np.roll(data2, -1)
#             data2[-1] = arousal

#             history1.append(valence)
#             history2.append(arousal)
        

#             # Trim the historical data lists to the desired length
#             history1 = history1[-num_points:]
#             history2 = history2[-num_points:]

#             # x_values = np.arange(len(data1))  # Generate a sequence of indices
#             # current_time = datetime.datetime.now().strftime("%H:%M:%S")  # Get the current time
#             # ax1.set_xticks(x_values)  # Set the x-axis ticks
#             # ax1.set_xticklabels([current_time] * len(data1))  # Set the x-axis tick labels
#             # ax2.set_xticks(x_values)  # Set the x-axis ticks
#             # ax2.set_xticklabels([current_time] * len(data2)) 

#             update_chart()
#             chart_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#             chart_image = chart_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#             # Display the chart image using cv2.imshow()
#             cv2.imshow(window_name, chart_image)
#         except:
#             continue


#         text = "classes: {}".format(emotions_print)
#         cv2.putText(image, text, org=(25, 25), fontFace=cv2.FONT_HERSHEY_DUPLEX,
#                     fontScale=0.35, color=(0, 255, 0))

#         vis_util.visualize_boxes_and_labels_on_image_array(
#             image,
#             np.squeeze(boxes),
#             np.squeeze(classes).astype(np.int32),
#             np.squeeze(scores),
#             category_index,
#             use_normalized_coordinates=True,
#             line_thickness=1)

#         if window_not_set is True:
#             cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)

#             window_not_set = False
#         cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
#         k = cv2.waitKey(1) & 0xff
#         if k == ord('q') or k == 27:
#             break
#         fps.update()
#     # stop the timer and display FPS information    
#     fps.stop()
#     print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#     print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#     # do a bit of cleanup
#     cv2.destroyAllWindows()
#     vs.stop()


# def predict_from_video(detector, file_path):
#     vs = FileVideoStream(file_path).start()
#     time.sleep(2.0)
#     time.sleep(2.0)
#     fps = FPS().start()
#     out = None
#     frame_count = 0

#     while vs.more():
#         # grab the frame from the threaded video stream
#         image = vs.read()
#         frame_count += 1

#         if out is None:
#             [h, w] = image.shape[:2]
#             out = cv2.VideoWriter("test_out.avi", 0, 25.0, (w, h))

#         # Check if this is the frame closest to 5 seconds
#         if frame_count == 2:
#             frame_count = 0

#             boxes, scores, classes, num_detections, emotions, valence, arousal = detector.run(image)
#             valence = valence[0]
#             arousal = arousal[0]

#             data1 = np.roll(data1, -1)
#             data1[-1] = valence
#             data2 = np.roll(data2, -1)
#             data2[-1] = arousal

#             history1.append(valence)
#             history2.append(arousal)
        

#             # Trim the historical data lists to the desired length
#             history1 = history1[-num_points:]
#             history2 = history2[-num_points:]
#             update_chart()
#             chart_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#             chart_image = chart_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#             # Display the chart image using cv2.imshow()
#             # cv2.imshow(window_name, chart_image)


#             text = "classes: {}".format(emotions)
#             cv2.putText(image, text, org=(25, 25), fontFace=cv2.FONT_HERSHEY_DUPLEX,
#                         fontScale=0.35, color=(0, 255, 0))

#             vis_util.visualize_boxes_and_labels_on_image_array(
#                 image,
#                 np.squeeze(boxes),
#                 np.squeeze(classes).astype(np.int32),
#                 np.squeeze(scores), 
#                 category_index,
#                 use_normalized_coordinates=True,
#                 line_thickness=1)
#         cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
#         k = cv2.waitKey(1) & 0xff
#         if k == ord('q') or k == 27:
#             break
#         out.write(image)
#         # preds_all = np.reshape(dd, (-1, 2))
#         # np.set_printoptions(suppress=True)
#         # np.savetxt("data.csv", preds_all, delimiter = ",", fmt='%f')
#         # fps.update()
#     # stop the timer and display FPS information
#     fps.stop()
#     print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#     print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#     # do a bit of cleanup
#     cv2.destroyAllWindows()
#     vs.stop()


# if __name__ == "__main__":

#     if sys.argv[1] == '-c':
#         K.clear_session()
#         tf.compat.v1.reset_default_graph()
#         detector = TensorflowDetector(PATH_TO_CKPT, PATH_TO_CLASS, PATH_TO_REGRESS)
#         predict_from_camera(detector)

#     elif sys.argv[1] == '-v':
#         K.clear_session()
#         tf.compat.v1.reset_default_graph()
#         detector = TensorflowDetector(PATH_TO_CKPT, PATH_TO_CLASS, PATH_TO_REGRESS)
#         predict_from_video(detector, '/home/ronak/Downloads/Elemental_oyt_trim.mp4')

#     else:
#         print('Wrong argument')