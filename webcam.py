import cv2

# Initialize the camera (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully. Press 'q' to close the window.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting...")
        break

    # Display the resulting frame
    cv2.imshow('Simple Camera Test', frame)

    # Use a 1ms delay; this is vital for the GUI to refresh
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()