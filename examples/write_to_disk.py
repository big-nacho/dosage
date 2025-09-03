import cv2
import numpy as np
from dosage import Dosage

# Usage:
# 1. Place {your_video_name.ext} in examples/input
# 2. Set name = '{your_video_name.ext}'
# 3. Run python3 examples/write_to_disk.py
# 4. See examples/output/{your_video_name.mp4}


name = '{your_video_name.ext}'  # <- Your video name goes here
src = f'examples/input/{name}'
out_name = f'examples/output/{name}'

cap = cv2.VideoCapture(src)
if not cap.isOpened():
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(
    out_name,
    fourcc,
    fps,
    (width, height)
)

dosage = Dosage(width, height)

frames_processed = 0
while True:
    grabbed, frame = cap.read()
    if not grabbed:
        break

    saliency_map = dosage.run(frame)
    saliency_map = (saliency_map * 255).astype(np.uint8)
    saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_GRAY2BGR)
    out.write(saliency_map)

    frames_processed += 1
    print(f'Progress: {round(100 * (frames_processed / frame_count))}%', end='\r')

cap.release()
out.release()
cv2.destroyAllWindows()
