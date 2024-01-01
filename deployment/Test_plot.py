import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Initialize OpenCV window
window_name = "Real-time Line Charts"
cv2.namedWindow(window_name)

# Initialize chart parameters
num_points_per_second = 1000  # Number of data points per second
num_points_history = 1000  # Number of historical data points to display
data1 = []  # List to store data points for Line 1
data2 = []  # List to store data points for Line 2
history1 = []  # List to store historical data points for Line 1
history2 = []  # List to store historical data points for Line 2

# Create the chart figure and axes
fig, (ax1, ax2) = plt.subplots(2, 1)

# Create Line 1 and Line 2
line1, = ax1.plot([], [], color='red', label='Line 1')
line2, = ax2.plot([], [], color='blue', label='Line 2')

# Add legends to the chart
ax1.legend()
ax2.legend()

# Update function for the chart
def update_chart():
    line1.set_data(np.arange(len(data1)), data1)
    line2.set_data(np.arange(len(data2)), data2)
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    fig.canvas.draw()

# Main loop
start_time = datetime.now()
while True:
    # Generate random data points for Line 1 and Line 2
    new_data_point1 = np.random.randint(0, 100)
    new_data_point2 = np.random.randint(0, 100)

    # Append the new data points to the respective data lists
    data1.append(new_data_point1)
    data2.append(new_data_point2)

    # Append the new data points to the historical data lists
    history1.append(new_data_point1)
    history2.append(new_data_point2)

    # Trim the data and history lists to the desired length
    data1 = data1[-num_points_per_second:]
    data2 = data2[-num_points_per_second:]
    history1 = history1[-num_points_history:]
    history2 = history2[-num_points_history:]

    # Update the chart
    update_chart()

    # Convert the chart figure to an image
    chart_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    chart_image = chart_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Display the chart image using cv2.imshow()
    cv2.imshow(window_name, chart_image)

    # Check for the 'Esc' key press to exit the loop
    if cv2.waitKey(1) == 27:
        break

    # Calculate the elapsed time in seconds
    elapsed_seconds = (datetime.now() - start_time).total_seconds()

    # Check if a second has elapsed
    if elapsed_seconds >= 1.0:
        # Print the historical data points for Line 1 and Line 2
        print("Line 1 History:", history1)
        print("Line 2 History:", history2)
        print("----------------------")

        # Reset the start time
        start_time = datetime.now()

# Close OpenCV window and exit
cv2.destroyAllWindows()
