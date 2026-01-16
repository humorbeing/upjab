import cv2
import os

def extend_video_by_repetition(
    input_path,
    target_length_sec=360
):
    # ---- Build output path ----
    dir_name, base_name = os.path.split(input_path)
    name, ext = os.path.splitext(base_name)
    ext = '.mp4'  # Force mp4 output
    output_path = os.path.join(dir_name, f"{name}_extended{ext}")

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    video_duration = len(frames) / fps
    repeat_count = int(target_length_sec / video_duration) + 1

    total_written = 0
    max_frames = int(target_length_sec * fps)

    for _ in range(repeat_count):
        for frame in frames:
            if total_written >= max_frames:
                break
            writer.write(frame)
            total_written += 1

    writer.release()
    print(f"Saved extended video to {output_path}")

    return output_path


# Example
if __name__ == "__main__":
    from upjab.video.extend_video_by_repetition_module import extend_video_by_repetition
    extend_video_by_repetition(
        "data/video/v01.avi",        
        target_length_sec=360
    )
    extend_video_by_repetition(
        "data/video/v02.mp4",        
        target_length_sec=360
    )
