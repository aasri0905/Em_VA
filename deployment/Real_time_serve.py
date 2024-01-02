import threading


import cv2
import streamlit as st
from matplotlib import pyplot as plt

from streamlit_webrtc import webrtc_streamer,RTCConfiguration

import os
import av
import numpy as np
import tempfile
import io
# import Main_1
from deployment import Main_1
import pandas as pd 
import datetime 
import time



# hashed_passwords = stauth.Hasher(['abc', 'def']).generate()


# RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
lock = threading.Lock()
img_container = {"img": None}
start_time = time.time()


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
    return frame

# ctx = webrtc_streamer(key="Internaltest", video_frame_callback=video_frame_callback,rtc_configuration=RTC_CONFIGURATION,media_stream_constraints={"video": True,"audio": False})

ctx = webrtc_streamer(key="Internaltest", video_frame_callback=video_frame_callback,media_stream_constraints={"video": True,"audio": False})


valence_data = []
arousal_data = []
timestamps = []
emotions_list = []
scores_list = [[] for _ in range(8)]


col1, col2 = st.columns([2,2],gap='medium')

with col1:
    chart_placeholder_valence = st.empty()
with col2:      
    chart_placeholder_arousal = st.empty()
chart_placeholder_emotions = st.empty()


# Fixed colors for each emotion
emotion_colors = ['#AAAAAA', '#FFD700', '#4169E1', '#FF4500', '#228B22', '#800080', '#FF0000', '#8B4513']

def update_plot(valence_data,timestamps,arousal_data,scores_list):

    D1 = pd.DataFrame({
        "timestamp": timestamps,
        "valence": valence_data
    })
    with col1:
        chart_placeholder_valence.line_chart(D1,x="timestamp")
    D2 = pd.DataFrame({
        "timestamp": timestamps,
        "arousal": arousal_data
    })
    with col2:
        chart_placeholder_arousal.line_chart(D2,x="timestamp")
    

    # Create a single plot for emotions
    data = pd.DataFrame({"timestamp": timestamps})
    for i, emotions_list in enumerate(["Neutral", 'Happy', 'Sad', 'Surprised', 'Afraid', 'Disgusted', 'Angry', 'Contemptuous']):
        print(data,"=============================")

        data[emotions_list] = scores_list[i]
    chart_placeholder_emotions.line_chart(data,x="timestamp")

while ctx.state.playing:
    with lock:
        img = img_container["img"]
    if img is None:
        continue
    VAL, ARO, emotions_print,emotions_score_,boxes = Main_1.Primary(img)

    timestamp = time.time() - start_time

    print(VAL,ARO,emotions_print,emotions_score_)

    timestamps.append(timestamp)
    valence_data.append(VAL)
    arousal_data.append(ARO)
    emotions_list.append(emotions_print)
    if str(emotions_print) == "Neutral":
        scores_list[0].append(emotions_score_)
    else:
        scores_list[0].append(0)
    if emotions_print == "Happy":
        scores_list[1].append(emotions_score_)
    else:
        scores_list[1].append(0)
    if emotions_print == "Sad":
        scores_list[2].append(emotions_score_)
    else:
        scores_list[2].append(0)        
    if emotions_print == "Surprised":
        scores_list[3].append(emotions_score_)
    else:
        scores_list[3].append(0)
    if emotions_print == "Afraid":
        scores_list[4].append(emotions_score_)
    else:
        scores_list[4].append(0)        
    if emotions_print == "Disgusted":
        scores_list[5].append(emotions_score_)
    else:
        scores_list[5].append(0)       
    if emotions_print == "Angry":
        scores_list[6].append(emotions_score_)
    else:
        scores_list[6].append(0)        
    if emotions_print == "Contemptuous":
        scores_list[7].append(emotions_score_)
    else:
        scores_list[7].append(0)

    print(scores_list)
    update_plot(valence_data,timestamps,arousal_data,scores_list)


