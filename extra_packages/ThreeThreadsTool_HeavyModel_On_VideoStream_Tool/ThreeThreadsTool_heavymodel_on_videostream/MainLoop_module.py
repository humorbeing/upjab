import cv2


from ThreeThreadsTool_heavymodel_on_videostream import VideoGet
from ThreeThreadsTool_heavymodel_on_videostream import VideoShow
from ThreeThreadsTool_heavymodel_on_videostream import Queue
from ThreeThreadsTool_heavymodel_on_videostream import ModelThread

import time
# from demo_model import detection_model_fn

def show_on_frame(frame, results):    
    for key, value in results.items():
        if isinstance(value, float):
            svalue = f"{value:.03f}"
        else:
            svalue = str(value)
        cv2.putText(frame, f"{key}: {svalue}", (10, 30 + 30 * list(results.keys()).index(key)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))     
    return frame



def ThreeThreadsTool(
    source,
    task_fn,
    frame_queue_size=30,
    feed_FPS=15,
    output_FPS=45,
    main_loop_FPS=120,
    stream_frame_width=1000,
):
    frame_queue = Queue(frame_queue_size)
    video_getter = VideoGet(frame_queue, source, feed_FPS=feed_FPS)
    video_getter.start()
    video_shower = VideoShow(video_getter.frame, output_FPS=output_FPS)
    video_shower.start()
    model_thread = ModelThread(frame_queue, task_fn)
    model_thread.start()
    
    init_time = time.perf_counter()
    main_loop_period = 1.0 / main_loop_FPS
    main_loop_status = {
        'Main Loop FPS': main_loop_FPS
    }
    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            model_thread.stop()
            break
        # time.sleep(1/120)
        frame = video_getter.frame
        
        height, width = frame.shape[0:2]
        aspect_ratio = height / width
        new_height = int(stream_frame_width * aspect_ratio)
        new_dimensions = (stream_frame_width, new_height)

        # Resize the image (INTER_AREA is good for downsampling, INTER_CUBIC for upsampling)
        frame = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_AREA)

        capture_status = video_getter.status
        # capture_fps = capture_status['Capture FPS']
        # feeding_fps = capture_status['Feeding FPS']
        
        results = model_thread.results
        streaming_status = video_shower.status
        show_resutls = {
            **capture_status, 
            **results, 
            **streaming_status,
            **main_loop_status
        }
        frame = show_on_frame(frame, show_resutls)
        video_shower.frame = frame

        on_time = time.perf_counter() - init_time
        if  main_loop_period > on_time:
            time.sleep(main_loop_period - on_time)

        this_time = time.perf_counter()
        duration = this_time - init_time
        init_time = this_time
        main_loop_status = {
            'Main Loop FPS': 1.0 / duration
        }
        

 

