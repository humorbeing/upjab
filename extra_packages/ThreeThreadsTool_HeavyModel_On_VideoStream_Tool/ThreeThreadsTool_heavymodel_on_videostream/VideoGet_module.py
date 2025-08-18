from threading import Thread
import cv2
import time

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, frame_queue, src=0, feed_FPS=15):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False


        self.video_fps = self.stream.get(cv2.CAP_PROP_FPS)        
        self.feeding_fps = feed_FPS
        self.skipping_coefficient = self.video_fps/self.feeding_fps
        self.counting_pool = self.skipping_coefficient        
        self.int_skip = int(self.counting_pool)
        self.counting_pool = self.counting_pool - self.int_skip
        self.frame_count = 0

        self.frame_queue = frame_queue
        

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        reading_start = time.time()
        reading_count = 0
        reading_fps_check = 0
        feeding_start = reading_start
        feeding_count = 0
        feeding_fps_check = 0
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
            
            if self.grabbed:
                reading_count += 1
                reading_duration = time.time() - reading_start
                reading_fps_check = reading_count / reading_duration
                # print(f'reading {self.frame_count}. {self.video_fps}')
                self.frame_count += 1
                if self.frame_count % self.int_skip == 0:
                    feeding_count += 1 
                    feeding_duration = time.time() - feeding_start
                    feeding_fps_check = feeding_count / feeding_duration
                    rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    self.frame_queue.put(rgb_frame)

                    self.counting_pool = self.counting_pool + self.skipping_coefficient
                    self.int_skip = int(self.counting_pool)
                    self.counting_pool = self.counting_pool - self.int_skip
                    self.frame_count = 0
                # print(f'real reading fps: {reading_fps_check:0.05f}. real feeding fps: {feeding_fps_check:0.05f}')
                self.skipping_coefficient = reading_fps_check/self.feeding_fps
            else:
                self.stop()



                
                
        
    def stop(self):
        self.stopped = True
