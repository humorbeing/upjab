from threading import Thread
import cv2
import time

from collections import deque



class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, frame_queue, src=0, feed_FPS=15):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

        frame_count = self.stream.get(cv2.CAP_PROP_FRAME_COUNT)
        # Check if frame_count is a valid, non-zero number (heuristic for files)
        if frame_count > 0: # Add a reasonable upper limit
            self.is_stream = False
        else:
            self.is_stream = True

        self.video_fps = self.stream.get(cv2.CAP_PROP_FPS)
        
        self.feeding_fps = feed_FPS
        self.skipping_coefficient = self.video_fps/self.feeding_fps
        self.counting_pool = self.skipping_coefficient        
        self.int_skip = int(self.counting_pool)
        self.counting_pool = self.counting_pool - self.int_skip
        self.frame_count = 0

        self.frame_queue = frame_queue
        self.real_video_FPS_deque = deque(maxlen=1)
        self.real_feeding_FPS_deque = deque(maxlen=1)

        self.real_feeding_fps = 0.0
        self.real_video_fps = 0.0
        self.status ={
            'Feeding FPS': self.real_feeding_fps,
            'Capture FPS': self.real_video_fps
        }

    def start(self):
        if self.is_stream:
            Thread(target=self.get_stream, args=()).start()
        else:
            Thread(target=self.get_video, args=()).start()
        return self

    def get_stream(self):
        print("Starting stream reading thread.")
        start_t = time.perf_counter()
        next_t = start_t
        frames_sent = 0

        frame_idx = 0
        next_frame_idx = 0.0

        reading_last = time.perf_counter()
        reading_count = 0        
        feeding_last = reading_last
        feeding_count = 0        
        while not self.stopped:
            # now = time.perf_counter()
            # sleep_for = next_t - now
            # if sleep_for > 0:
            #     time.sleep(sleep_for)
            (self.grabbed, self.frame) = self.stream.read()
            # frames_sent += 1
            # period = 1.0 / float(self.video_fps)
            # next_t = start_t + frames_sent * period
            if self.grabbed:
                if frame_idx >= round(next_frame_idx):
                    rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    self.frame_queue.put(rgb_frame)
                    next_frame_idx += self.real_video_fps / self.feeding_fps
                    feeding_count += 1
                    feeding_new = time.perf_counter()
                    feeding_duration = feeding_new - feeding_last
                    self.real_feeding_FPS_deque.append(1.0 / feeding_duration)
                    feeding_last = feeding_new
                    self.real_feeding_fps = sum(self.real_feeding_FPS_deque) / len(self.real_feeding_FPS_deque)
                
                frame_idx += 1
                
                reading_count += 1
                reading_new = time.perf_counter()
                reading_duration = reading_new - reading_last
                self.real_video_FPS_deque.append(1.0 / reading_duration)
                reading_last = reading_new
                self.real_video_fps = sum(self.real_video_FPS_deque) / len(self.real_video_FPS_deque)
                
                
                print(f'real video fps: {self.real_video_fps:0.05f}. real feeding fps: {self.real_feeding_fps:0.05f}')
                self.status ={
                    'Feeding FPS': self.real_feeding_fps,
                    'Capture FPS': self.real_video_fps
                }
            else:
                self.stop()



    def get_video(self):
        print('Video mode activated.')
        
        start_t = time.perf_counter()
        next_t = start_t
        frames_sent = 0

        frame_idx = 0
        next_frame_idx = 0.0

        reading_last = time.perf_counter()
        reading_count = 0        
        feeding_last = reading_last
        feeding_count = 0        
        while not self.stopped:
            now = time.perf_counter()
            sleep_for = next_t - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            (self.grabbed, self.frame) = self.stream.read()
            frames_sent += 1
            period = 1.0 / float(self.video_fps)
            next_t = start_t + frames_sent * period
            if self.grabbed:
                if frame_idx >= round(next_frame_idx):
                    rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    self.frame_queue.put(rgb_frame)
                    next_frame_idx += self.real_video_fps / self.feeding_fps
                    feeding_count += 1
                    feeding_new = time.perf_counter()
                    feeding_duration = feeding_new - feeding_last
                    self.real_feeding_FPS_deque.append(1.0 / feeding_duration)
                    feeding_last = feeding_new
                    self.real_feeding_fps = sum(self.real_feeding_FPS_deque) / len(self.real_feeding_FPS_deque)
                
                frame_idx += 1
                
                reading_count += 1
                reading_new = time.perf_counter()
                reading_duration = reading_new - reading_last
                self.real_video_FPS_deque.append(1.0 / reading_duration)
                reading_last = reading_new
                self.real_video_fps = sum(self.real_video_FPS_deque) / len(self.real_video_FPS_deque)
                
                
                print(f'real video fps: {self.real_video_fps:0.05f}. real feeding fps: {self.real_feeding_fps:0.05f}')
                self.status ={
                    'Feeding FPS': self.real_feeding_fps,
                    'Capture FPS': self.real_video_fps
                }
            else:
                self.stop()


                
        
    def stop(self):
        self.stopped = True

    
        
