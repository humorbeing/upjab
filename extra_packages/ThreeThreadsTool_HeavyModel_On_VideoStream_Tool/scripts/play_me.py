import cv2

from demo_model import detection_model_fn

from ThreeThreadsTool_heavymodel_on_videostream import VideoGet
from ThreeThreadsTool_heavymodel_on_videostream import VideoShow
from ThreeThreadsTool_heavymodel_on_videostream import Queue
from ThreeThreadsTool_heavymodel_on_videostream import ModelThread

import time


def show_on_frame(frame, results):    
    for key, value in results.items():
        cv2.putText(frame, f"{key}: {value:.03f}", (10, 30 + 30 * list(results.keys()).index(key)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))     
    return frame



def threadBoth(source=0):    
    frame_queue = Queue(30)
    video_getter = VideoGet(frame_queue, source, feed_FPS=15)
    video_getter.start()
    video_shower = VideoShow(video_getter.frame)
    video_shower.start()
    model_thread = ModelThread(frame_queue, detection_model_fn)
    model_thread.start() 
    

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            model_thread.stop()
            break
        time.sleep(1/120)
        frame = video_getter.frame
        results = model_thread.results        
        frame = show_on_frame(frame, results)
        video_shower.frame = frame
        

 
if __name__ == "__main__":
    # video_source = 'rtp://192.168.0.4:5004'
    video_source = 0
    # video_source = 'rtmp://115.22.172.73/live/stream'
    # video_source = 'data/KakaoTalk_20250226_121532508.mp4'
    video_source = 'data/video/v01.avi'
    threadBoth(video_source)
