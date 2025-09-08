import cv2
import time
import threading


class VideoCaptureThread:
    def __init__(self, src=''):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.delay = 1.0 / self.fps if self.fps > 0 else 1.0 / 30
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(
            target=self.update,
            daemon=True
        ).start()
        return self

    def update(self):
        while not self.stopped:
            start = time.time()
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed = grabbed
                if grabbed:
                    self.frame = frame

            elapsed = time.time() - start
            sleep_time = max(0.0, self.delay - elapsed)
            time.sleep(sleep_time)

    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.grabbed else None

    def stop(self):
        self.stopped = True
        self.cap.release()
