import cv2

# Global variables
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

# OpenCV video capture
video_path = "path/to/video/file.mp4"
cap = cv2.VideoCapture(0)

# Create a window and set mouse callback
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_callback)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Track the bounding box if initialized
    if initialized:
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(coord) for coord in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw the bounding box if drawing
    if drawing:
        cv2.rectangle(frame, refPt[0], (refPt[0][0], refPt[0][1]), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
