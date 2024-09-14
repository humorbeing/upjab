import numpy as np

def get_numpy_video(video_path, is_cv2=True):
    if is_cv2:
        import cv2

        stream = cv2.VideoCapture(video_path)
        fps = stream.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            (grabbed, frame) = stream.read()
            if grabbed:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
            else:
                break
        # import numpy as np
        video_ = np.array(frames)
        return video_, fps
    else:
        from torchvision.io.video import read_video
        # pip install av  # IF needed
        rgb, audio, info = read_video(video_path, pts_unit='sec')
        fps = info['video_fps']
        video_ = rgb.numpy()
        return video_, fps

def video_to_need_fps(THWC_numpy_video, video_fps, feeding_fps=15, echo=True):   

    if feeding_fps > video_fps:
        if echo:
            print('----------From: video_to_need_fps')
            print(f'[feeding_fps > video_fps] Can not get FPS needed')
            print('----------END: video_to_need_fps')
        # print(f'[feeding_fps > video_fps] Can not get FPS needed')
        
        return THWC_numpy_video
    elif feeding_fps == video_fps:
        if echo:
            print('----------From: video_to_need_fps')
            print('[feeding_fps == video_fps] FPS is good')
            print('----------END: video_to_need_fps')
        return THWC_numpy_video
    else:
        
        skipping_coefficient = video_fps/feeding_fps
        counting_pool = skipping_coefficient        
        int_skip = int(counting_pool)
        counting_pool = counting_pool - int_skip
        frame_count = 0

        new_frames = []
        for frame in THWC_numpy_video:
            frame_count += 1
            if frame_count % int_skip == 0:
                new_frames.append(frame)

                counting_pool = counting_pool + skipping_coefficient
                int_skip = int(counting_pool)
                counting_pool = counting_pool - int_skip
                frame_count = 0      

        # import numpy as np

        new_frames_np = np.array(new_frames)
        if echo:
            print('----------From: video_to_need_fps')
            print(f'FPS is changed to {feeding_fps}')
            print('----------END: video_to_need_fps')
        return new_frames_np
    

def read_video(
    video_path,
    is_cv2=True,
    feeding_fps=None,
    echo=False
    ):

    vid, video_fps = get_numpy_video(
        video_path=video_path,
        is_cv2=is_cv2
        )
    if echo:
        print(f'read video: {video_path}')
        
    if feeding_fps is None:
        if echo:
            print(f'No. of video frames: {len(vid)}, video_fps: {video_fps}')
        return vid
    else:
        video_ = video_to_need_fps(
            THWC_numpy_video=vid,
            video_fps=video_fps,
            feeding_fps=feeding_fps,
            echo=echo
        )
        if echo:
            print(f'Original No. of video frames: {len(vid)}, video fps: {video_fps}')
            print(f'Return No. of video frames: {len(video_)}, estimate video fps: {feeding_fps}')
        return video_


if __name__ == '__main__':
    video_path = 'example_data/videos/fishes/crowd/00000001.mp4'
    vid = read_video(video_path=video_path)
    vid = read_video(video_path=video_path, echo=True)
    # vid = np.random.randint(0, 256, size=(105,360,480,3))
    

    video_path = 'example_data/videos/fishes/crowd/00000103_notcrowd.mp4'
    vid = read_video(
        video_path=video_path,
        is_cv2=True,
        feeding_fps=None,
        echo=True)
    
    vid = read_video(
        video_path=video_path,
        is_cv2=False,
        feeding_fps=None,
        echo=True)
    
    vid = read_video(
        video_path=video_path,
        is_cv2=True,
        feeding_fps=15,
        echo=True)
    
    print('end')