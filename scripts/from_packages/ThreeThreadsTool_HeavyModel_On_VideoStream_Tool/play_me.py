


from ThreeThreadsTool_heavymodel_on_videostream import ThreeThreadsTool

from demo_model import detection_model_fn

   

 
if __name__ == "__main__":
    
    # video_source = 'rtp://192.168.0.4:5004'
    video_source = 0
    # video_source = 'rtmp://115.22.172.73/live/stream'
    
    
    # is_extended_video_needed = True    
    is_extended_video_needed = False
    if is_extended_video_needed:
        from upjab.video.extend_video_by_repetition_module import extend_video_by_repetition
        extend_video_by_repetition(
            "data/video/v01.avi",
            target_length_sec=3600
        )
    video_source = 'data/video/v01_extended.mp4'
    
    
    # is_timer_video_needed = True
    is_timer_video_needed = False
    if is_timer_video_needed:
        from create_timer_video import make_timer_video
        make_timer_video(
            output_path="saves/timer_SS_ms_0_to_300.mp4",
            width=800,
            height=600,
            fps=60,
            total_seconds=300
        )
    
    video_source = 'saves/timer_SS_ms_0_to_300.mp4'


    ThreeThreadsTool(
        source=video_source,
        frame_queue_size=30,
        feed_FPS=13,
        task_fn=detection_model_fn
    )
