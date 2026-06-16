import cv2

def show_camera_stream(camera_index=0):
    """
    Displays the live camera feed using OpenCV.
    
    :param camera_index: Index of the camera (0 = default webcam)
    """
    # Open the camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}")
        return

    print("Press 'q' to quit the camera stream.")

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame from camera.")
            break

        # Display the frame
        cv2.imshow("Camera Stream", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_camera_stream()
