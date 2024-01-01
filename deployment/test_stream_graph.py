import streamlit as st
import numpy as np
import time

# Initialize empty lists for data
timestamp_list = []
valence_list = []
arousal_list = []
emotion_score_lists = [[] for _ in range(8)]

# Function to simulate real-time data updates (replace with your actual data source)
def generate_data():
    timestamp_list.append(len(timestamp_list))
    valence_list.append(np.random.rand() * 100)
    arousal_list.append(np.random.rand() * 100)
    for i in range(8):
        emotion_score_lists[i].append(np.random.rand() * 100)

# Create the chart placeholders
chart_placeholder_valence = st.empty()
chart_placeholder_arousal = st.empty()
chart_placeholder_emotions = st.empty()

chart_valence = chart_placeholder_valence.line_chart({
    "timestamp": timestamp_list,
    "valence": valence_list
},x="timestamp")

chart_arousal = chart_placeholder_arousal.line_chart({
    "timestamp": timestamp_list,
    "arousal": arousal_list
},x="timestamp")

# Fixed colors for each emotion
emotion_colors = ['#AAAAAA', '#FFD700', '#4169E1', '#FF4500', '#228B22', '#800080', '#FF0000', '#8B4513']

# Function to update the existing charts with new data
def update_plot():
    generate_data()
    chart_valence.line_chart({
        "timestamp": timestamp_list,
        "valence": valence_list
    },x="timestamp")
    chart_arousal.line_chart({
        "timestamp": timestamp_list,
        "arousal": arousal_list
    },x="timestamp")

    # Create a single plot for emotions
    data = {"timestamp": timestamp_list}
    for i, emotion in enumerate(['Neutral', 'Happy', 'Sad', 'Surprised', 'Afraid', 'Disgusted', 'Angry', 'Contemptuous']):
        data[emotion] = emotion_score_lists[i]
    
    chart_placeholder_emotions.line_chart(data,x="timestamp")


# Start the real-time plotting
if st.button("Start Plotting"):
    while True:
        update_plot()
        time.sleep(1)  # Adjust the interval as needed
