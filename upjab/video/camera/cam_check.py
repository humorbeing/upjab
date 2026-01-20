import cv2

def print_camera_check():
    # Try to open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video stream or camera.")
        print("Please check the camera index or if another application is using the camera.")
    else:
        print("Camera successfully opened.")
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame.")
                break

            # Display the resulting frame
            cv2.imshow('Camera Feed', frame)

            # Press 'q' on the keyboard to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    from upjab.video.camera.cam_check import print_camera_check
    print_camera_check()
