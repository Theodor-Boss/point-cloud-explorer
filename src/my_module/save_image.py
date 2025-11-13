import cv2
import os
print(os.getcwd())

# Åbn videoen
video = cv2.VideoCapture("src/my_module/video.mp4")

# Gå til et bestemt tidspunkt (fx 5 sekunder inde)
video.set(cv2.CAP_PROP_POS_MSEC, (9*60+22)*1000)

if not video.isOpened():
    print("Kunne ikke åbne videoen!")

# Læs næste frame
success, frame = video.read()

if success:
    cv2.imwrite("src/my_module/frame1.jpg", frame)
    print("Gemte frame som frame.jpg")
else:
    print("Kunne ikke læse frame")

video.release()
