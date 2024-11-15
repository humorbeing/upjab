import cv2
import os



def video_to_image(video_path, folder_name=None):    
    file_name = os.path.basename(video_path)
    if folder_name is None:
        folder_name = os.path.dirname(video_path)
    save_folder = folder_name + '/' + file_name[:-4]
    os.makedirs(save_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count= 0 
    retaining = True
    while retaining:
        retaining, bgr_frame = cap.read()
        if retaining:
            count += 1
            cv2.imwrite(save_folder+f'/{count:08d}.png', bgr_frame)
        

if __name__ == '__main__':
    video_path = 'example_data/videos/fishes/crowd/00000001.mp4'
    video_to_image(video_path)