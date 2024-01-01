import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading
import numpy as np
import cv2
import tempfile
import io
import os
import Main_1
import pandas as pd 
import datetime 
import matplotlib.pyplot as plt
import time


RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

st.set_page_config(layout='wide',page_title="Streamlit WebRTC", page_icon="🤖")
 
task_list = ["Video Stream", "Upload Video and Process"]

with st.sidebar:
    st.title('Task Selection')
    task_name = st.selectbox("Select your tasks:", task_list)
st.title(task_name)

if task_name == task_list[0]:
    style_list = ['color', 'black and white']

    st.sidebar.header('Style Selection')
    style_selection = st.sidebar.selectbox("Choose your style:", style_list)

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model_lock = threading.Lock()
            self.style = style_list[0]

        def update_style(self, new_style):
            if self.style != new_style:
                with self.model_lock:
                    self.style = new_style

        def recv(self, frame):
            img = frame.to_image()
            if self.style == style_list[1]:
                img = img.convert("L")
            return av.VideoFrame.from_image(img)

    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

    if ctx.video_processor:

        ctx.video_transformer.update_style(style_selection)

elif task_name == task_list[1]:
    st.write("Please upload a video file.")

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
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        result = cv2.VideoWriter('demo_pop.mp4', fourcc,20, (frame_width,frame_height))

        D1 = []
        D2 = []
        D3 = []
        T=0

        emotions_print_ = ""

        while True:
            ret, frame = cap.read()
            video_time = cap.get(cv2.CAP_PROP_POS_MSEC)

            if ret == False:
                break      

            VAL, ARO, emotions_print = Main_1.Primary(frame)
            
            if emotions_print_ == emotions_print:
                print(time.time())
            else:
                print(100*"-")
            emotions_print_ = emotions_print

            print(emotions_print_)

            D1.append(VAL)
            D2.append(ARO)

            
            T=T+1
            D3.append(T)


            result.write(frame)

        D1 = [i for i in D1 if i]
        D2 = [i for i in D2 if i]
        print(D1, len(D1),len(D2),"-----------------")


        cap.release()
        result.release()


        data = pd.DataFrame()
        se = pd.Series(D1)
        data['Valence'] = se.values
        se1 = pd.Series(D2)
        data['Arousal'] = se1.values
        se2 = pd.Series(D3)
        try:
            data['FNO'] = se2.values
        except:
            pass
        data['FPS'] = int(20)
        data["X-axis"] = data['FNO']/ data['FPS']
        # 
        data = data.fillna(0)
        data.to_csv("Saver.csv",index=False)
        data_2 = pd.DataFrame()
        data_2['Time'] = data['X-axis']
        data_2['Valence'] = data['Valence']
        data_2['Arousal'] = data['Arousal']

        data_2 = data_2.rename(columns={'Time':'index'}).set_index('index')


        os.system("ffmpeg -i demo_pop.mp4 -c:v libx264 -profile:v main -vf format=yuv420p -c:a aac -movflags +faststart output.mp4 -y")

        video_file = open('output.mp4', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)
        print(data)
        st.line_chart(data_2[['Valence', 'Arousal']])

        # st.area_chart(data_2)