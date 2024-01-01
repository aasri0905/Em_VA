import threading

import cv2
import streamlit as st
from matplotlib import pyplot as plt

from streamlit_webrtc import webrtc_streamer,RTCConfiguration

import av
import numpy as np
import tempfile
import io
import os
import Main_1
import pandas as pd 
import datetime 
import time
import csv
import uuid




RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
lock = threading.Lock()
img_container = {"img": None}
start_time = time.time()
random_uuid = uuid.uuid4()

print("Random UUID:----------", random_uuid)



def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
    return frame

# ctx = webrtc_streamer(key="Internaltest", video_frame_callback=video_frame_callback,rtc_configuration=RTC_CONFIGURATION,media_stream_constraints={"video": True,"audio": False})

ctx = webrtc_streamer(key="Internaltest", video_frame_callback=video_frame_callback,media_stream_constraints={"video": True,"audio": False})

headers = ['Valence', 'Arousal', 'Emotion','Em_score','Timestamp']
file_name = "CSV_SAVER/"+str(random_uuid)+".csv"
directory = "CSV_SAVER"
file_path = os.path.join("CSV_SAVER", str(random_uuid)+".csv")
if not os.path.exists(directory):
    os.makedirs(directory)



while ctx.state.playing:    
    with lock:
        img = img_container["img"]
    if img is None:
        continue
    VAL, ARO, emotions_print,emotions_score_,boxes = Main_1.Primary(img)

    timestamp = time.time() - start_time

    data_insert = {"Valence":VAL,"Arousal":ARO,"Emo_Name":emotions_print,"Emotion_score":emotions_score_,"Timestamps":timestamp}

    df = pd.DataFrame(data_insert,index=[0])

    print(VAL,ARO,emotions_print,emotions_score_,timestamp)

    with open(file_name, 'a') as f:
            df.to_csv(f, header=f.tell() == 0, index=False)