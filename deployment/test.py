import cv2
import Main_1
import time



 
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

print(frame_width, frame_height)
size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
result = cv2.VideoWriter('demo_pop.mp4', fourcc,20, (frame_width,frame_height))

emotions_print_ = ""

while True:
    ret, frame = cap.read()

    if ret == False:
        break      

    VAL, ARO, emotions_print  = Main_1.Primary(frame)

    if emotions_print_ == emotions_print:
        print(time.time())
    else:
        print(100*"-")
    emotions_print_ = emotions_print


    # Seek_in, Seek_out = 

    result.write(frame)
    cv2.imshow("frame",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
result.release()
