import cv2
from dosage import Dosage
from VideoCapture import VideoCaptureThread

# Usage:
# 1. Place {your_video_name.ext} in examples/input
# 2. Set name = '{your_video_name.ext}'
# 3. Run python3 examples/real_time.py

name = '{your_video_name.ext}'  # <- Your video name goes here
src = f'examples/input/{name}'

cap_thread = VideoCaptureThread(src).start()
width = int(cap_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

dosage = Dosage(width, height)

while cap_thread.cap.isOpened():
    grabbed, frame = cap_thread.read()
    if not grabbed:
        break

    saliency_map = dosage.run(frame)
    cv2.imshow("Result", saliency_map)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_thread.stop()
cv2.destroyAllWindows()
