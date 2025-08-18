from threading import Thread
import cv2
import time

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None, output_FPS=60, do_time=0.0033279):
        self.frame = frame
        self.stopped = False
        self.one_frame_time = 1 / output_FPS
        self.do_time = do_time
        if self.one_frame_time > self.do_time:
            self.sleep_time = self.one_frame_time - self.do_time
        else:
            self.sleep_time = 0

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        start_time = time.time()
        init_time = start_time
        count = 0
        while not self.stopped:
            count += 1
            # print("show image")
            cv2.imshow("Video", self.frame)
            
            # if self.sleep_time == 0:
            #     pass
            # else:
            #     time.sleep(self.sleep_time)
            on_time = time.time() - init_time
            if self.one_frame_time > on_time:
                time.sleep(self.one_frame_time - on_time)
            new_init_time = time.time()
            duration = new_init_time - start_time
            this_time = new_init_time - init_time
            init_time = new_init_time            
            # print(f'Average 1F duration: {duration / count:0.05f}. Average FPS: {count / duration:0.05f}. FPS: {1/this_time:0.05f}')
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True
