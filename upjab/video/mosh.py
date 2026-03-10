# Problem:
# libGL.so.1: cannot open shared object file: No such file or directory
# Solution: (SSH connection)
# sudo apt update
# sudo apt install libgl1


# Problem:
# ffmpeg: not found
# Solution: (SSH connection)
# sudo apt update
# sudo apt install ffmpeg

import os
import subprocess
import cv2
import random

def datamosh(input_video, start_point=0.0, end_point=1.0, random_seed=2024):
    random.seed(random_seed)    

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()


    start_frame = int(frame_count * start_point)
    end_frame = int(frame_count * end_point)
    video_name = input_video[:-4]
    output_video = f'{video_name}_datamosh.mp4'

    input_avi = 'datamoshing_input.avi'  # must be an AVI so i-frames can be located in binary file
    output_avi = 'datamoshing_output.avi'

    # convert original file to avi
    subprocess.call('ffmpeg -loglevel error -y -i ' + input_video + ' ' +
                    ' -crf 0 -pix_fmt yuv420p -bf 0 -b 10000k -r ' + str(fps) + ' ' +
                    input_avi, shell=True)

    # open up the new files so we can read and write bytes to them
    in_file = open(input_avi, 'rb')
    out_file = open(output_avi, 'wb')    


    # because we used 'rb' above when the file is read the output is in byte format instead of Unicode strings
    in_file_bytes = in_file.read()

    # 0x30306463 which is ASCII 00dc signals the end of a frame.
    frame_start = bytes.fromhex('30306463')

    # get all frames of video
    frames = in_file_bytes.split(frame_start)

    # write header
    out_file.write(frames[0])
    frames = frames[1:]

    # 0x0001B0 signals the beginning of an i-frame, 0x0001B6 signals a p-frame
    iframe = bytes.fromhex('0001B0')
    pframe = bytes.fromhex('0001B6')


    def write_frame(frame):
        out_file.write(frame_start + frame)

    frame_count = 0
    first_iframe = None
    iframe_list = []
    pframe_list = []
    other_list = []
    # pframe_repeat_gap = end_frame - start_frame
    randint_min = 4
    randint_max = 8
    pc = 0  # pframe counter
    
    randi = random.randint(randint_min, randint_max)
    last_pframe = None
    for frame in frames:
        if frame[5:8] == iframe:
            frame_count = frame_count + 1
            iframe_list.append([frame, frame_count])
            if first_iframe is None:
                first_iframe = frame
            if (start_frame <= frame_count) and (frame_count <= end_frame):
                try:
                    if random.random() > 0.2:
                        temp_frame = iframe_list[-2][0]
                    else:
                        temp_frame = random.sample(iframe_list, 1)[0][0]
                except:
                    temp_frame = first_iframe
                write_frame(temp_frame)
            else:
                write_frame(frame)
        

        elif frame[5:8] == pframe:
            frame_count = frame_count + 1
            pframe_list.append([frame, frame_count])
            if last_pframe is None:
                last_pframe = frame
            
            if random.random() > 0.7:
                if (start_frame <= frame_count) and (frame_count <= end_frame):
                    write_frame(last_pframe)
                    pc = pc + 1
                    if pc == randi:
                        randi = random.randint(randint_min, randint_max)
                        pc = 0
                        last_pframe = frame
                    else:
                        pass
                else:
                    last_pframe = frame
                    write_frame(frame)
            else:
                randi = random.randint(randint_min, randint_max)
                pc = 0
                last_pframe = frame
                write_frame(frame)
        else:
            write_frame(frame)
            other_list.append([frame, frame_count])


    # export the video
    subprocess.call('ffmpeg -loglevel error -y -i ' + output_avi + ' ' +
                    ' -crf 18 -pix_fmt yuv420p -vcodec libx264 -acodec aac -b 10000k -r ' + str(fps) + ' ' +
                    output_video, shell=True)

    in_file.close()
    out_file.close()
    os.remove(input_avi)
    os.remove(output_avi)

if __name__ == '__main__':
    from upjab.video.mosh import datamosh

    datamosh('data/video/v01.avi')
    datamosh('data/video/v02.mp4')
    print('end')

