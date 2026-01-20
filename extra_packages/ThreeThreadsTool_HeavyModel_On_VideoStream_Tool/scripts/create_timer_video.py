import cv2
import numpy as np

def make_timer_video(
    output_path="timer_SS_ms_0_to_300.mp4",
    width=800,
    height=600,
    fps=60,
    total_seconds=300
):
    total_frames = int(total_seconds * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to create VideoWriter")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4.0
    thickness = 8
    text_color = (0, 0, 0)  # black

    for frame_idx in range(total_frames + 1):
        # Video-time (exact, no sleep)
        t = frame_idx / fps
        if t > total_seconds:
            t = total_seconds

        seconds = int(t)
        milliseconds = int((t - seconds) * 100)  # 00–99

        timer_text = f"{seconds}:{milliseconds:02d}"  # SS:ms

        # White background
        frame = np.full((height, width, 3), 255, dtype=np.uint8)

        # Center text
        (tw, th), _ = cv2.getTextSize(timer_text, font, font_scale, thickness)
        x = (width - tw) // 2
        y = (height + th) // 2

        cv2.putText(
            frame,
            timer_text,
            (x, y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )

        writer.write(frame)

    writer.release()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    from upjab import AP
    make_timer_video(
        output_path=AP("saves/timer_SS_ms_0_to_300.mp4"),
        width=800,
        height=600,
        fps=60,
        total_seconds=300
    )
