import threading

import os

import cv2
import streamlit as st
from matplotlib import pyplot as plt

from streamlit_webrtc import webrtc_streamer,RTCConfiguration

import av
import numpy as np
import tempfile
import io
# import Main_1
import sys
sys.path.append("./deployment/")
from deployment import Main_1
import pandas as pd 
import datetime 
import time


st.set_page_config(layout='centered',page_title="Streamlit WebRTC", page_icon="🤖")
 
task_list = ["Video Stream", "Upload Video and Process"]

with st.sidebar:
    st.title('Task Selection')
    task_name = st.selectbox("Select your tasks:", task_list)
st.title(task_name)


if task_name == task_list[0]:

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    lock = threading.Lock()
    img_container = {"img": None}
    start_time = time.time()


    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        with lock:
            img_container["img"] = img
        return frame

    ctx = webrtc_streamer(key="Internaltest", video_frame_callback=video_frame_callback,rtc_configuration=RTC_CONFIGURATION,media_stream_constraints={"video": True,"audio": False})

    # ctx = webrtc_streamer(key="Internaltest", video_frame_callback=video_frame_callback,media_stream_constraints={"video": True,"audio": False})


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
            chart_placeholder_valence.line_chart(D1,x="timestamp",y="valence")
        D2 = pd.DataFrame({
            "timestamp": timestamps,
            "arousal": arousal_data
        })
        with col2:
            chart_placeholder_arousal.line_chart(D2,x="timestamp",y="arousal")
        

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

elif task_name == task_list[1]:

    st.write("Please upload a video file.")
    st.write("Plese Note That, There should be only one person in the frame with good light conditions.")


    video_file = st.file_uploader("Upload Video", type=["mp4", "mov"])

    st.write("Upload completed")

    if st.button("Process"):
        if video_file is not None:
            g = io.BytesIO(video_file.read())  ## BytesIO Object
            temporary_location = "testout_simple.mp4"

            with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
                out.write(g.read())  ## Read bytes into file

            out.close()
        st.write("Processing the uploaded video ")

        cap = cv2.VideoCapture("testout_simple.mp4")
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        print(frame_width, frame_height)
        size = (frame_width, frame_height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        result = cv2.VideoWriter('demo_pop.mp4', fourcc,20, (frame_width,frame_height))

        col1, col2 = st.columns([2,2],gap='medium')

        with col1:
            chart_placeholder_valence = st.empty()
        with col2:      
            chart_placeholder_arousal = st.empty()
        chart_placeholder_emotions = st.empty()

        def update_plot(valence_data,timestamps,arousal_data,scores_list):

            D1 = pd.DataFrame({
                "timestamp": timestamps,
                "valence": valence_data
            })
            with col1:
                chart_placeholder_valence.line_chart(D1,x="timestamp",y="valence")
            D2 = pd.DataFrame({
                "timestamp": timestamps,
                "arousal": arousal_data
            })
            with col2:
                chart_placeholder_arousal.line_chart(D2,x="timestamp",y="arousal")
            

            # Create a single plot for emotions
            data = pd.DataFrame({"timestamp": timestamps})
            for i, emotions_list in enumerate(["Neutral", 'Happy', 'Sad', 'Surprised', 'Afraid', 'Disgusted', 'Angry', 'Contemptuous']):
                print(data,"=============================")

                data[emotions_list] = scores_list[i]
            chart_placeholder_emotions.line_chart(data,x="timestamp")


        D1 = []
        D2 = []
        D3 = []
        scores_list = [[] for _ in range(8)]
        T=0

        # emotions_print_ = ""
        start_time = time.time()


        while True:
            ret, frame = cap.read()
            video_time = cap.get(cv2.CAP_PROP_POS_MSEC)

            if ret == False:
                break      

            VAL, ARO, emotions_print,emotions_score_,boxes = Main_1.Primary(frame)

            timestamp = time.time() - start_time
            
            D1.append(VAL)
            D2.append(ARO)
            D3.append(timestamp)
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
        
            result.write(frame)
        
        result.release()

        os.system("ffmpeg -i demo_pop.mp4 -c:v libx264 -profile:v main -vf format=yuv420p -c:a aac -movflags +faststart output.mp4 -y")

        video_file = open('output.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        update_plot(D1,D3,D2,scores_list)