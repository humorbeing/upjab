
import math
import warnings
from typing import Any, Callable,  Dict, List, Optional, Tuple, TypeVar, Union

import torch
from torchvision.io import read_video, read_video_timestamps


def unfold(tensor: torch.Tensor, size: int, step: int, dilation: int = 1) -> torch.Tensor:
    """
    similar to tensor.unfold, but with the dilation
    and specialized for 1d tensors

    Returns all consecutive windows of `size` elements, with
    `step` between windows. The distance between each element
    in a window is given by `dilation`.
    """
    if tensor.dim() != 1:
        raise ValueError(f"tensor should have 1 dimension instead of {tensor.dim()}")
    o_stride = tensor.stride(0)
    numel = tensor.numel()
    new_stride = (step * o_stride, dilation * o_stride)
    new_size = ((numel - (dilation * (size - 1) + 1)) // step + 1, size)
    if new_size[0] < 1:
        new_size = (0, size)
    return torch.as_strided(tensor, new_size, new_stride)


def compute_clips_for_video(
    video_pts: torch.Tensor, num_frames: int, step: int, fps: Optional[float], frame_rate: Optional[float] = None
) -> Tuple[torch.Tensor, Union[List[slice], torch.Tensor]]:
    if fps is None:
        # if for some reason the video doesn't have fps (because doesn't have a video stream)
        # set the fps to 1. The value doesn't matter, because video_pts is empty anyway
        fps = 1
    if frame_rate is None:
        frame_rate = fps
    total_frames = len(video_pts) * frame_rate / fps
    _idxs = _resample_video_idx(int(math.floor(total_frames)), fps, frame_rate)
    video_pts = video_pts[_idxs]
    clips = unfold(video_pts, num_frames, step)
    if not clips.numel():
        warnings.warn(
            "There aren't enough frames in the current video to get a clip for the given clip length and "
            "frames between clips. The video (and potentially others) will be skipped."
        )
    idxs: Union[List[slice], torch.Tensor]
    if isinstance(_idxs, slice):
        idxs = [_idxs] * len(clips)
    else:
        idxs = unfold(_idxs, num_frames, step)
    return clips, idxs


def _resample_video_idx(num_frames: int, original_fps: float, new_fps: float) -> Union[slice, torch.Tensor]:
    step = original_fps / new_fps
    if step.is_integer():
        # optimization: if step is integer, don't need to perform
        # advanced indexing
        step = int(step)
        return slice(None, None, step)
    idxs = torch.arange(num_frames, dtype=torch.float32) * step
    idxs = idxs.floor().to(torch.int64)
    return idxs


class VideoClips:   

    def __init__(
        self,
        video_path,
        clip_length_in_frames: int = 16,
        frames_between_clips: int = 1,
        frame_rate=15,        
        output_format: str = "TCHW",
    ) -> None:

        self.video_path = video_path

        self.output_format = output_format.upper()
        if self.output_format not in ("THWC", "TCHW"):
            raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

        
        self._compute_frame_pts()
        
        self.compute_clips(clip_length_in_frames, frames_between_clips, frame_rate)


    def _compute_frame_pts(self) -> None:
        
        pts, self.video_fps = read_video_timestamps(self.video_path)
        self.video_pts = torch.as_tensor(pts, dtype=torch.long)      

    
    def compute_clips(self, num_frames: int, step: int, frame_rate: Optional[float] = None) -> None:
        
        self.num_frames = num_frames
        self.step = step
        self.frame_rate = frame_rate        
        # self.clips, self.resampling_idxs = self.compute_clips_for_video(self.video_pts, num_frames, step, self.video_fps, frame_rate)
        self.clips, self.resampling_idxs = compute_clips_for_video(self.video_pts, num_frames, step, self.video_fps, frame_rate)
        

    def __len__(self) -> int:
        return len(self.clips)

    
    def get_clip(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], int]:
        
        if idx >= len(self.clips):
            raise IndexError(f"Index {idx} out of range ({self.num_clips()} number of clips)")
        
        video_path = self.video_path
        clip_pts = self.clips[idx]       

        
        start_pts = clip_pts[0].item()
        end_pts = clip_pts[-1].item()
        video, audio, info = read_video(video_path, start_pts, end_pts)
        
        if self.frame_rate is not None:
            resampling_idx = self.resampling_idxs[idx]
            if isinstance(resampling_idx, torch.Tensor):
                resampling_idx = resampling_idx - resampling_idx[0]
            video = video[resampling_idx]
            info["video_fps"] = self.frame_rate
        assert len(video) == self.num_frames, f"{video.shape} x {self.num_frames}"

        if self.output_format == "TCHW":
            # [T,H,W,C] --> [T,C,H,W]
            video = video.permute(0, 3, 1, 2)

        return video, audio, info, video_path


if __name__ == '__main__':
    
    video_path = 'example_data/videos/fishes/crowd/00000103_notcrowd.mp4'
    vc = VideoClips(
        video_path,
        frame_rate=15
    )
    temp11 = vc.get_clip(0)
    print(f'{len(vc)=}')
    
    from upjab.tool.timer import timer
    with timer() as t:
        read_video(video_path)
    
    with timer() as t:
        read_video_timestamps(video_path)
    print('done')